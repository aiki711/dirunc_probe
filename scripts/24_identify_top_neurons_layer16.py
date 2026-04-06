import torch
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())
from scripts.common import DIRS

def main():
    probe_path = "runs/contrastive/contrastive_ft_layer16.pt"
    output_path = "runs/contrastive/layer16_neurons_report.json"
    
    if not os.path.exists(probe_path):
        print(f"Error: Probe not found at {probe_path}")
        return

    print(f"Loading probe from {probe_path}...")
    weights = torch.load(probe_path, map_location="cpu")
    W = weights["W"] # [n_dirs, hidden_size]
    b = weights["b"] # [n_dirs]
    
    report = {}
    
    for i, label in enumerate(DIRS):
        # W[i] is the vector for this slot
        vec = W[i]
        
        # Find top 10 positive and 10 negative contributors
        # positive means "Missing", negative means "Filled" (due to how we trained contrastive)
        # Actually, in our training (19b):
        # A (Missing) -> Target 1
        # B (Filled) -> Target 0
        # So positive weights contribute to "Missing" prediction.
        
        values, indices = torch.sort(vec, descending=True)
        
        top_positive = []
        for idx, val in zip(indices[:10], values[:10]):
            top_positive.append({
                "index": int(idx),
                "weight": float(val)
            })
            
        top_negative = []
        for idx, val in zip(indices[-10:], values[-10:]):
            top_negative.append({
                "index": int(idx),
                "weight": float(val)
            })
            
        report[label] = {
            "top_missing_neurons": top_positive,  # Positive weights (A=Missing)
            "top_filled_neurons": top_negative[::-1] # Most negative weights (B=Filled)
        }
        
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Saved neuron report to {output_path}")
    
    # Display top neurons for key slots
    for label in ["who", "what", "where"]:
        if label in report:
            print(f"\n--- Top neurons for {label} ---")
            print("  [Missing side (Pos weight)]")
            for n in report[label]["top_missing_neurons"][:3]:
                print(f"    n{n['index']}: {n['weight']:.4f}")
            print("  [Filled side (Neg weight)]")
            for n in report[label]["top_filled_neurons"][:3]:
                print(f"    n{n['index']}: {n['weight']:.4f}")

if __name__ == "__main__":
    main()
