import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def generate_visuals(data_path, out_dir, title_prefix=""):
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Dist Data
    dist_records = []
    paired_records = []
    scatter_records = []
    
    for label, d in data.items():
        pairs = [p for p in d.get("pairs", []) if p.get("valid")]
        top_neurons = d.get("neuron_indices", [])
        if not pairs or not top_neurons: continue
        
        nid = top_neurons[0] # Focus on top 1
        
        for k, p in enumerate(pairs):
            act_a = p["A_activations"][0]
            act_b = p["B_activations"][0]
            prob_a = p["A_prob"]
            prob_b = p["B_prob"]
            
            dist_records.append({"Label": label.upper(), "Condition": "Missing (A)", "Activation": act_a})
            dist_records.append({"Label": label.upper(), "Condition": "Filled (B)", "Activation": act_b})
            
            # for paired, just take first 30 per label to avoid clutter
            if k < 30:
                paired_records.append({"Label": label.upper(), "PairID": k, "Missing": act_a, "Filled": act_b})
                
            scatter_records.append({
                "Label": label.upper(),
                "Δ_Activation": act_b - act_a,
                "Δ_Probability": prob_b - prob_a
            })

    if not dist_records: 
        print(f"No records parsed for {title_prefix}")
        return

    # --- STEP 1: Violin Plot ---
    df_dist = pd.DataFrame(dist_records)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    sns.violinplot(data=df_dist, x="Label", y="Activation", hue="Condition", split=True, inner="quart", palette={"Missing (A)": "#ff9999", "Filled (B)": "#99ccff"})
    plt.title(f"Step 1: Distribution of Neuron Activations ({title_prefix})\nMissing State vs. Filled State", fontsize=14)
    plt.ylabel("Neuron Activation Value")
    plt.tight_layout()
    plt.savefig(out_dir / "step1_distribution.png", dpi=300)
    plt.close()

    # --- STEP 2: Dumbbell Paired Plot ---
    df_paired = pd.DataFrame(paired_records)
    labels = df_paired["Label"].unique()
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 5), sharey=True)
    if len(labels) == 1: axes = [axes]
    
    for ax, lbl in zip(axes, labels):
        sub = df_paired[df_paired["Label"] == lbl]
        for _, row in sub.iterrows():
            ax.plot([0, 1], [row["Missing"], row["Filled"]], color="gray", alpha=0.5)
            ax.scatter(0, row["Missing"], color="#ff9999", zorder=5, s=40)
            ax.scatter(1, row["Filled"], color="#99ccff", zorder=5, s=40)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Missing", "Filled"])
        ax.set_title(lbl)
    
    fig.suptitle(f"Step 2: Paired Activation Shifts (Sample of 30) - {title_prefix}", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "step2_paired_shifts.png", dpi=300)
    plt.close()

    # --- STEP 3: Scatter Plot ---
    df_scatter = pd.DataFrame(scatter_records)
    g = sns.lmplot(data=df_scatter, x="Δ_Activation", y="Δ_Probability", col="Label", col_wrap=3, height=4, scatter_kws={"alpha":0.5, "color":"#8e44ad"}, line_kws={"color":"red"})
    g.fig.suptitle(f"Step 3: Causal Correlation [ΔActivation vs ΔProbability] ({title_prefix})", y=1.05, fontsize=14)
    g.set_axis_labels("Δ Activation (Filled - Missing)", "Δ Probability")
    
    # Add R and p annotations
    for ax, lbl in zip(g.axes.flat, g.col_names):
        sub = df_scatter[df_scatter["Label"] == lbl]
        if len(sub) > 1:
            # Need to filter out exact identical deltas preventing variance
            std_act = np.std(sub["Δ_Activation"])
            std_prob = np.std(sub["Δ_Probability"])
            if std_act > 1e-6 and std_prob > 1e-6:
                r, p = stats.pearsonr(sub["Δ_Activation"], sub["Δ_Probability"])
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                ax.annotate(f"r = {r:.3f}{sig}\np = {p:.2e}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
            
    plt.savefig(out_dir / "step3_scatter_correlation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comprehensive visuals to {out_dir}")

def main():
    plot_args = [
        ("runs/balanced/experiment7_neurons/corpus_shift_data.json", "runs/balanced/experiment7_neurons", "Standard LODO"),
        ("runs/balanced/experiment7_neurons_ft/corpus_shift_data.json", "runs/balanced/experiment7_neurons_ft", "Final Token")
    ]
    for dp, od, pfx in plot_args:
        generate_visuals(dp, od, pfx)

if __name__ == "__main__":
    main()
