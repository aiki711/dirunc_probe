import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, precision_recall_fscore_support

from common import DIRS, SPECIAL_TOKENS, QUERY_LABEL_STR
import importlib.util
import sys

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
evaluate_cached = train_probe.evaluate_cached
JsonlDirUncDataset = train_probe.JsonlDirUncDataset
FilterSpec = train_probe.FilterSpec
collate_batch = train_probe.collate_batch
strip_query_tokens = train_probe.strip_query_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--data_jsonl", type=str, required=True, help="Evaluation dataset (e.g. dev_balanced.jsonl)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained probe .pt file")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--mode", type=str, default="query", choices=["baseline", "query"])
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None, help="If None, uses 0.5 (or per-class best from summary)")
    parser.add_argument("--summary_json", type=str, default=None, help="If provided, loads optimized thresholds")
    parser.add_argument("--strip_query", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    
    # Load base model
    shared_model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    shared_model.resize_token_embeddings(len(tokenizer))
    shared_model.to(device)
    shared_model.eval()

    # Prepare dataset
    with open(args.data_jsonl, "r") as f:
        rows = [json.loads(line) for line in f]
    
    if args.strip_query:
        rows = [{"text": strip_query_tokens(r["text"]), **{k:v for k,v in r.items() if k!="text"}} for r in rows]
    
    fs = FilterSpec(None, None, None)
    ds = JsonlDirUncDataset(rows, fs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                    collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length))

    # Wrap model
    base = ProbeModelBase(
        args.model_name,
        vocab_size=None,
        train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
        pretrained_model=shared_model,
    ).to(device)

    # Extract activations
    print(f"Extracting activations for layer {args.layer_idx}...")
    X_dict, Y, _ = extract_activations(dl, base, [args.layer_idx], args.mode, device, tokenizer)
    X = X_dict[args.layer_idx]

    # Load probe
    hidden_size = shared_model.config.hidden_size
    num_labels = len(DIRS)
    
    if args.mode == "baseline":
        probe = train_probe.BaselineHead(hidden_size, num_labels, torch.float32).to(device)
    else:
        probe = train_probe.QueryHead(hidden_size, num_labels, torch.float32).to(device)
        
    probe.load_state_dict(torch.load(args.model_path, map_location=device))
    probe.eval()

    # Determine thresholds
    thresholds = torch.full((len(DIRS),), 0.5, device=device)
    if args.threshold is not None:
        thresholds = torch.full((len(DIRS),), args.threshold, device=device)
    elif args.summary_json:
        # Load from summary.json (optimized per-class)
        with open(args.summary_json, "r") as f:
            summary = json.load(f)
        key = f"{args.mode}/layer_{args.layer_idx}"
        best_metrics = summary.get(key, {}).get("best", {})
        ts = best_metrics.get("per_label_thresholds", None)
        if ts:
            thresholds = torch.tensor(ts, device=device)
            print(f"Loaded optimized thresholds: {ts}")
        else:
            t = best_metrics.get("threshold", 0.5)
            thresholds = torch.full((len(DIRS),), t, device=device)
            print(f"Loaded single threshold: {t}")

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_cached(probe, X, Y, thresholds)
    
    # Save results
    result_file = out_dir / "eval_metrics.json"
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {result_file}")
    print(f"Macro F1 (posonly): {metrics['macro_f1_posonly']:.4f}")

if __name__ == "__main__":
    main()
