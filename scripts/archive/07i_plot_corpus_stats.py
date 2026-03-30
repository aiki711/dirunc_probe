import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def plot_corpus_shifts(data_path, out_dir, title_prefix="Standard"):
    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot Delta Activations
    summary_data = []
    
    for label, data in all_results.items():
        valid_pairs = [p for p in data.get("pairs", []) if p.get("valid")]
        top_neurons = data.get("neuron_indices", [])
        
        if not valid_pairs: continue
        
        for i, nid in enumerate(top_neurons):
            acts_a = [p["A_activations"][i] for p in valid_pairs]
            acts_b = [p["B_activations"][i] for p in valid_pairs]
            deltas = [b - a for a, b in zip(acts_a, acts_b)]
            
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            _, p_val = stats.ttest_rel(acts_b, acts_a)
            
            # Keep only the top 1 neuron for simplicity in the main plot
            if i == 0:
                summary_data.append({
                    "Label": label.upper(),
                    "Neuron": f"n{nid}",
                    "Mean_Delta": mean_delta,
                    "Std_Delta": std_delta,
                    "P_val": p_val
                })

    if not summary_data:
        print(f"No valid data to plot for {data_path}")
        return

    labels = [d["Label"] for d in summary_data]
    means = [d["Mean_Delta"] for d in summary_data]
    errs = [d["Std_Delta"] for d in summary_data]

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Create barplot
    colors = ["#e74c3c" if m < 0 else "#2ecc71" for m in means]
    bars = plt.bar(labels, means, yerr=errs, capsize=5, color=colors, alpha=0.8)
    
    plt.axhline(0, color="black", linestyle="--", linewidth=1.5)
    plt.title(f"Experiment 7: Corpus Shift Activation Δ ({title_prefix})\nTop Neuron per Category", fontsize=14)
    plt.ylabel("Mean Δ Activation (Filled - Missing)", fontsize=12)
    plt.xlabel("Target Category", fontsize=12)
    
    # Add p-value annotations
    for bar, d in zip(bars, summary_data):
        height = bar.get_height()
        y_pos = height - (0.5 if height < 0 else -0.5)
        sig = "***" if d["P_val"] < 0.001 else ("**" if d["P_val"] < 0.01 else ("*" if d["P_val"] < 0.05 else ""))
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                 f'{sig}\n{d["Neuron"]}',
                 ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

    plt.tight_layout()
    out_file = out_dir / "exp7_corpus_shift_plot.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    # Standard
    plot_corpus_shifts(
        "runs/balanced/experiment7_neurons/corpus_shift_data.json",
        "runs/balanced/experiment7_neurons",
        title_prefix="Standard LODO"
    )
    # Final Token
    plot_corpus_shifts(
        "runs/balanced/experiment7_neurons_ft/corpus_shift_data.json",
        "runs/balanced/experiment7_neurons_ft",
        title_prefix="Final Token"
    )
