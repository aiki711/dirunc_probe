import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_summary(path):
    with open(path, "r") as f:
        return json.load(f)

def extract_best_metrics(summary):
    # Find best_overall or equivalent
    if "best_overall" in summary:
        best_data = summary["best_overall"]["best"]
        name = summary.get("best_overall_key", "best")
    else:
        # Fallback to query/layer_25 for Exp 1 if best_overall missing
        keys = ["query/layer_25", "query/best"]
        best_data = None
        name = "best"
        for k in keys:
            if k in summary:
                best_data = summary[k]["best"]
                name = k
                break
    
    if best_data is None:
        return None
        
    per_label = best_data["per_label_f1"]
    macro = best_data["macro_f1_posonly"]
    return {
        "macro": macro,
        "per_label": per_label,
        "name": name
    }

def main():
    base_path = Path("runs/imbalanced")
    exp_files = {
        "Exp 1 (Baseline)": base_path / "experiment1_baseline/summary.json",
        "Exp 2 (Per-class)": base_path / "experiment2_perclass/summary.json",
        "Exp 3 (Multilayer)": base_path / "experiment3_multilayer/summary.json"
    }
    
    results = {}
    for label, path in exp_files.items():
        if path.exists():
            data = load_summary(path)
            metrics = extract_best_metrics(data)
            if metrics:
                results[label] = metrics
    
    labels = ["who", "what", "when", "where", "why", "how", "which"]
    
    # --- B2-style: Best Macro F1 Comparison ---
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    names = list(results.keys())
    macros = [results[n]["macro"] for n in names]
    
    bars = sns.barplot(x=names, y=macros, palette="viridis")
    plt.ylim(0, 1.0)
    plt.title("Comparison of Best Macro F1 (Pos-only) across Experiments", fontsize=14)
    plt.ylabel("Macro F1 score")
    
    for i, v in enumerate(macros):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(base_path / "B2_comparison_macro_f1.png")
    print(f"Saved B2 comparison to {base_path / 'B2_comparison_macro_f1.png'}")

    # --- B4-style: Per-label F1 Comparison ---
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.25
    
    for i, (name, metrics) in enumerate(results.items()):
        plt.bar(x + (i - 1) * width, metrics["per_label"], width, label=name)
        
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.title("Per-label F1 Score Comparison across Experiments", fontsize=14)
    plt.ylabel("F1 Score")
    plt.xlabel("Category")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(base_path / "B4_comparison_per_label_f1.png")
    print(f"Saved B4 comparison to {base_path / 'B4_comparison_per_label_f1.png'}")

if __name__ == "__main__":
    main()
