import torch
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import importlib.util
import sys

from common import DIRS, SPECIAL_TOKENS, QUERY_LABEL_STR

def load_03_train_probe():
    path = Path(__file__).parent / "03_train_probe.py"
    spec = importlib.util.spec_from_file_location("train_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_probe"] = module
    spec.loader.exec_module(module)
    return module

train_probe = load_03_train_probe()
ProbeModelBase = train_probe.ProbeModelBase
MultiLayerQueryHead = train_probe.MultiLayerQueryHead
extract_activations = train_probe.extract_activations
evaluate_cached = train_probe.evaluate_cached
JsonlDirUncDataset = train_probe.JsonlDirUncDataset
FilterSpec = train_probe.FilterSpec
collate_batch = train_probe.collate_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-9b")
    parser.add_argument("--data_jsonl", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--layers", type=str, default="10,15,20,25")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--summary_json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [int(l) for l in args.layers.split(",")]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens(SPECIAL_TOKENS)
    
    # Load base model
    shared_model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    shared_model.resize_token_embeddings(len(tokenizer))
    shared_model.to(device)
    shared_model.eval()

    print(f"Loading dataset from {args.data_jsonl}...")
    with open(args.data_jsonl, "r") as f:
        rows = [json.loads(line) for line in f]
    
    fs = FilterSpec(None, None, None)
    ds = JsonlDirUncDataset(rows, fs)
    dataset = ds
    
    # === Experiment Configuration Print ===
    print(f"\n{'='*40}")
    print(f" Experiment Configuration")
    print(f"{'='*40}")
    print(f"Script: {sys.argv[0]}")
    exp_type = "balanced" if "balanced" in args.out_dir or "balanced" in args.data_jsonl else "imbalanced"
    print(f"Experiment Type: {exp_type} (inferred from path)")
    print(f"Model Name: {args.model_name}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Directory: {args.out_dir}")
    print(f"Target Layers: {args.layers}")
    print(f"Dataset Path: {args.data_jsonl}")
    print(f"Dataset Size (rows): {len(rows)}")
    print(f"Batch Size: {args.batch_size}")
    print(f"{'='*40}\n")
    # ======================================

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                    collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length))

    # Wrap model
    base = ProbeModelBase(
        args.model_name,
        vocab_size=None,
        train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
        pretrained_model=shared_model,
    ).to(device)

    # Extract activations
    print(f"Extracting activations for layers {layers}...")
    # extract_activations returns X_dict: {layer_idx: tensor(N, C, H)}
    X_dict, Y, _ = extract_activations(dl, base, layers, "query", device, tokenizer)
    
    # Stack layers: (N, L, C, H)
    X_stacked = torch.stack([X_dict[l] for l in layers], dim=1)

    # Load multi-layer probe
    hidden_size = shared_model.config.hidden_size
    num_labels = len(DIRS)
    probe = MultiLayerQueryHead(hidden_size, len(layers), num_labels, torch.float32).to(device)
    
    print(f"Loading checkpoint from {args.model_path}...")
    probe.load_state_dict(torch.load(args.model_path, map_location=device))
    probe.eval()

    # Determine thresholds (default 0.5 or from summary)
    thresholds = torch.full((len(DIRS),), 0.5, device=device)
    if args.summary_json:
        with open(args.summary_json, "r") as f:
            summary = json.load(f)
        ml_best = summary.get("multilayer", {}).get("best", {})
        ts = ml_best.get("per_label_thresholds", None)
        if ts:
            thresholds = torch.tensor(ts, device=device)
            print(f"Loaded optimized thresholds: {ts}")

    # Evaluate
    print("Evaluating...")
    with torch.no_grad():
        logits = probe(X_stacked)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_true = Y.cpu().numpy()
        
    metrics = train_probe.eval_with_threshold(y_true, probs, 0.5) # Basic eval
    # If we have per-class thresholds
    if args.summary_json:
        metrics = train_probe.eval_with_per_class_threshold(y_true, probs, thresholds.cpu().numpy())

    # Save results
    result_file = out_dir / "eval_metrics_balanced.json"
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {result_file}")
    print(f"Macro F1 (posonly): {metrics['macro_f1_posonly']:.4f}")

if __name__ == "__main__":
    main()
