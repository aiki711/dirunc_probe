import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import sys
import importlib.util
from sklearn.metrics import f1_score

def load_03_train_probe():
    path = Path(__file__).parent / "03_train_probe.py"
    spec = importlib.util.spec_from_file_location("train_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_probe"] = module
    spec.loader.exec_module(module)
    return module

train_probe = load_03_train_probe()
ProbeModelBase = train_probe.ProbeModelBase
extract_activations = train_probe.extract_activations
JsonlDirUncDataset = train_probe.JsonlDirUncDataset
FilterSpec = train_probe.FilterSpec
collate_batch = train_probe.collate_batch

from common import DIRS, SPECIAL_TOKENS, QUERY_LABEL_STR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Evaluation dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained probe .pt file")
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--summary_json", type=str, required=True, help="Path to summary.json with thresholds")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--mode", type=str, default="query")
    parser.add_argument("--dataset_names", nargs="+", default=["sgd", "multiwoz", "qasrl"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    
    shared_model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    shared_model.resize_token_embeddings(len(tokenizer))
    shared_model.to(device)
    shared_model.eval()

    with open(args.data_jsonl, "r") as f:
        rows = [json.loads(line) for line in f]

    fs = FilterSpec(None, None, None)
    ds = JsonlDirUncDataset(rows, fs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                    collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length))

    base = ProbeModelBase(
        args.model_name,
        vocab_size=None,
        train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
        pretrained_model=shared_model,
    ).to(device)

    print(f"Extracting activations for layer {args.layer_idx}...")
    X_dict, Y, _ = extract_activations(dl, base, [args.layer_idx], args.mode, device, tokenizer)
    X = X_dict[args.layer_idx]

    hidden_size = shared_model.config.hidden_size
    num_labels = len(DIRS)
    probe = train_probe.QueryHead(hidden_size, num_labels, torch.float32).to(device)
    probe.load_state_dict(torch.load(args.model_path, map_location=device))
    probe.eval()

    with open(args.summary_json, "r") as f:
        summary = json.load(f)
    best_metrics = summary["best_overall"]["best"]
    t_dict = best_metrics["per_class_tuned"]["threshold_dict"]
    thresholds = torch.tensor([t_dict[d] for d in DIRS], device=device)
    print(f"Loaded optimized thresholds: {t_dict}")

    print("Evaluating...")
    with torch.no_grad():
        logits = probe(X)
        probs = torch.sigmoid(logits)
        # Apply thresholds manually since apply_thresholds is missing
        preds = (probs >= thresholds).float()

    Y_np = Y.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # Eval overall
    macro_f1 = f1_score(Y_np, preds_np, average="macro", zero_division=0)
    print(f"\n[Overall] Macro F1: {macro_f1:.4f}")

    # Eval per dataset
    for ds_name in args.dataset_names:
        indices = [i for i, r in enumerate(rows) if ds_name.lower() in r.get("id", "").lower()]
        if not indices:
            print(f"No samples found for dataset {ds_name}")
            continue
        
        y_ds = Y_np[indices]
        preds_ds = preds_np[indices]
        macro_f1_ds = f1_score(y_ds, preds_ds, average="macro", zero_division=0)
        
        # Calculate per-class F1 for this dataset
        per_class_f1 = f1_score(y_ds, preds_ds, average=None, zero_division=0)
        
        print(f"\n[{ds_name.upper()}] (N={len(indices)})")
        print(f"Macro F1: {macro_f1_ds:.4f}")
        for j, cls in enumerate(DIRS):
            print(f"  - {cls}: {per_class_f1[j]:.4f}")

if __name__ == "__main__":
    main()
