import torch
import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys
import os

# Add the project root to sys.path to allow importing scripts.common
sys.path.append(os.getcwd())
from scripts.common import DIRS

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="runs/balanced/experiment6_lodo")
    parser.add_argument("--out_json", type=str, default="runs/balanced/experiment7_neurons/neurons_report.json")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_paths = list(model_dir.glob("*.pt"))
    
    if not model_paths:
        print(f"No models found in {model_dir}")
        return

    print(f"Analyzing {len(model_paths)} models from {model_dir}...")

    # label -> neuron_idx -> list of weight values
    label_neuron_stats = defaultdict(lambda: defaultdict(list))
    
    for path in model_paths:
        weights = torch.load(path, map_location="cpu")
        W = weights["W"] # [num_labels, hidden_size]
        
        for i, label in enumerate(DIRS):
            w_label = W[i]
            # Get top k values and indices (only positive weights are relevant for "activation")
            # We take more than top_k initially to ensure we find intersection
            vals, idxs = torch.topk(w_label, k=args.top_k * 2) 
            for val, idx in zip(vals.tolist(), idxs.tolist()):
                if val > 0:
                    label_neuron_stats[label][idx].append(val)

    # Aggregate: find neurons that are consistent across models
    report = {}
    for label in DIRS:
        neurons = []
        for idx, vals in label_neuron_stats[label].items():
            # Consensus: must appear in at least 70% of the models
            if len(vals) >= (len(model_paths) * 0.7):
                avg_val = sum(vals) / len(vals)
                neurons.append({
                    "index": int(idx),
                    "avg_weight": float(avg_val),
                    "frequency": int(len(vals))
                })
        
        # Sort by average weight descending
        neurons.sort(key=lambda x: x["avg_weight"], reverse=True)
        report[label] = neurons[:args.top_k]

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Neuron report saved to {out_path}")
    for label in DIRS:
        top_ns = report[label][:5]
        indices = [n["index"] for n in top_ns]
        print(f"  {label:<7}: Top 5 neurons: {indices}")

if __name__ == "__main__":
    main()
