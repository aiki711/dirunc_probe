import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", type=str, default="runs/layer_sweep_gemma2")
    parser.add_argument("--out_path", type=str, default="runs/layer_sweep_gemma2/sweep_plot.png")
    args = parser.parse_args()

    results = []
    runs_path = Path(args.runs_dir)
    
    # Load all layer results
    for layer_dir in sorted(runs_path.glob("layer_*"), key=lambda d: int(d.name.split("_")[1])):
        layer_idx = int(layer_dir.name.split("_")[1])
        log_file = layer_dir / "log.jsonl"
        if log_file.exists():
            with log_file.open("r") as f:
                lines = f.readlines()
                if not lines: continue
                # Get the best epoch based on pair_accuracy_standard + macro_f1 (as in 32_train_contrastive_probe.py)
                best_epoch_data = None
                best_score = -1.0
                for line in lines:
                    data = json.loads(line)
                    score = data.get("pair_accuracy_standard", 0.0) + data.get("macro_f1", 0.0)
                    if score > best_score:
                        best_score = score
                        best_epoch_data = data
                
                if best_epoch_data:
                    best_epoch_data["layer"] = layer_idx
                    results.append(best_epoch_data)

    if not results:
        print("No results found in", args.runs_dir)
        return

    df = pd.DataFrame(results).sort_values("layer")
    print("Summary of results:")
    print(df[["layer", "pair_accuracy_standard", "pair_accuracy_strict", "macro_f1"]])

    # Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(df["layer"], df["pair_accuracy_standard"], marker='o', label="Standard Accuracy")
    plt.plot(df["layer"], df["pair_accuracy_strict"],   marker='s', label="Strict Accuracy")
    plt.plot(df["layer"], df["macro_f1"],              marker='^', label="Macro F1")
    
    plt.xlabel("Layer Index")
    plt.ylabel("Score")
    plt.title(f"Contrastive Probing Performance across Layers\n(Gemma-2-2b-it, SGD+MultiWOZ)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(df["layer"])
    
    plt.tight_layout()
    plt.savefig(args.out_path)
    print(f"Plot saved to {args.out_path}")

    # Optional: Per-case-role plot
    if "by_case_role" in df.columns:
        plt.figure(figsize=(12, 7), dpi=150)
        roles = set()
        for res in results:
            roles.update(res["by_case_role"].keys())
        
        for role in sorted(roles):
            role_accs = []
            layers = []
            for res in results:
                if role in res["by_case_role"]:
                    role_accs.append(res["by_case_role"][role]["pair_acc_standard"])
                    layers.append(res["layer"])
            if role_accs:
                plt.plot(layers, role_accs, marker='.', label=role)
        
        plt.xlabel("Layer Index")
        plt.ylabel("Standard Pair Accuracy")
        plt.title("Per-Case-Role Performance across Layers")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        role_plot_path = args.out_path.replace(".png", "_per_role.png")
        plt.savefig(role_plot_path)
        print(f"Per-role plot saved to {role_plot_path}")

if __name__ == "__main__":
    main()
