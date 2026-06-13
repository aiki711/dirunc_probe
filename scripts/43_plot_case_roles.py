#!/usr/bin/env python3
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Configurations covering all 4 sweep runs
CONFIGS = {
    "Query (Aligned)": {
        "dir_name": "layer_sweep_gemini_nq_aligned",
        "color": "#1565C0",       # Dark Blue
        "linestyle": "-",         # Solid
        "marker": "o",
    },
    "Query (Unaligned)": {
        "dir_name": "layer_sweep_gemini_nq_unaligned",
        "color": "#64B5F6",       # Light Blue
        "linestyle": "--",        # Dashed
        "marker": "o",
    },
    "Final (Aligned)": {
        "dir_name": "layer_sweep_gemini_final_token_aligned",
        "color": "#BF360C",       # Dark Red-Orange
        "linestyle": "-",         # Solid
        "marker": "s",
    },
    "Final (Unaligned)": {
        "dir_name": "layer_sweep_gemini_final_token_unaligned",
        "color": "#FF8A65",       # Light Orange
        "linestyle": "--",        # Dashed
        "marker": "s",
    }
}

OMISSIONS = ["soft", "strong"]
LAYERS = [0, 4, 8, 12, 16, 20, 24, 26]
CASE_ROLES = ["Agent", "Goal", "Location", "Manner", "Source", "Theme", "Time"]
RUNS_DIR = Path("runs")

def load_data():
    # Structure: role -> omission -> config -> list of (layer, acc_std, acc_strict, n)
    data = {role: {om: {cfg: [] for cfg in CONFIGS} for om in OMISSIONS} for role in CASE_ROLES}
    overall = {om: {cfg: {"std": [], "str": []} for cfg in CONFIGS} for om in OMISSIONS}
    
    for omission in OMISSIONS:
        for config_label, cfg_info in CONFIGS.items():
            dir_path = RUNS_DIR / cfg_info["dir_name"]
            
            for layer in LAYERS:
                log_file = dir_path / f"{omission}_layer_{layer}" / "log.jsonl"
                if not log_file.exists():
                    continue
                
                # Read epochs
                epochs = []
                with log_file.open("r") as f:
                    for line in f:
                        if line.strip():
                            epochs.append(json.loads(line))
                
                if not epochs:
                    continue
                
                # Find best epoch based on macro_f1
                best_ep_idx = np.argmax([ep["macro_f1"] for ep in epochs])
                best_ep_data = epochs[best_ep_idx]
                
                # Get role-wise metrics
                role_metrics = best_ep_data.get("by_case_role", {})
                for role in CASE_ROLES:
                    r_data = role_metrics.get(role, {})
                    n = r_data.get("n", 0)
                    acc_std = r_data.get("pair_acc_standard", 0.0)
                    acc_strict = r_data.get("pair_acc_strict", 0.0)
                    data[role][omission][config_label].append((layer, acc_std, acc_strict, n))
                
                # Overall average
                overall[omission][config_label]["std"].append((layer, best_ep_data.get("pair_accuracy_standard", 0.0)))
                overall[omission][config_label]["str"].append((layer, best_ep_data.get("pair_accuracy_strict", 0.0)))
                
    return data, overall

def plot_aligned_vs_unaligned(data, overall, omission_type="soft", metric_type="strict", filename=""):
    metric_display = "Strict Pair Accuracy" if metric_type == "strict" else "Standard Pair Accuracy"
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 20), sharex=True)
    axes = axes.flatten()
    
    for i, role in enumerate(CASE_ROLES):
        ax = axes[i]
        sample_size = 0
        
        for config_label, cfg_info in CONFIGS.items():
            points = data[role][omission_type][config_label]
            if not points:
                continue
            
            x = [p[0] for p in points]
            y = [p[2] if metric_type == "strict" else p[1] for p in points]
            sample_size = points[0][3]
            
            ax.plot(
                x, y,
                label=config_label,
                color=cfg_info["color"],
                linestyle=cfg_info["linestyle"],
                marker=cfg_info["marker"],
                linewidth=2.2,
                markersize=6
            )
            
        ax.set_title(f"{role} (n={sample_size})", fontsize=14, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xticks(LAYERS)
        if metric_type == "strict":
            ax.set_ylim(0.2, 0.95)
        else:
            ax.set_ylim(0.5, 1.01)
            
        if i % 2 == 0:
            ax.set_ylabel(metric_display, fontsize=12)
            
    # Subplot 8: Overall Average
    ax_avg = axes[7]
    for config_label, cfg_info in CONFIGS.items():
        points = overall[omission_type][config_label]["str" if metric_type == "strict" else "std"]
        if not points:
            continue
        
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        ax_avg.plot(
            x, y,
            label=config_label,
            color=cfg_info["color"],
            linestyle=cfg_info["linestyle"],
            marker=cfg_info["marker"],
            linewidth=2.5,
            markersize=7
        )
        
    ax_avg.set_title("Overall Average (All Slots)", fontsize=14, fontweight="bold", pad=8)
    ax_avg.grid(True, linestyle=":", alpha=0.6)
    ax_avg.set_xticks(LAYERS)
    if metric_type == "strict":
        ax_avg.set_ylim(0.2, 0.95)
    else:
        ax_avg.set_ylim(0.5, 1.01)
        
    axes[6].set_xlabel("Layer", fontsize=12)
    axes[7].set_xlabel("Layer", fontsize=12)
    
    # Single Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=4, fontsize=12, frameon=True)
    
    plt.suptitle(
        f"Aligned vs Unaligned Comparison: {omission_type.upper()} OMISSION ({metric_display})",
        fontsize=18,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    out_path = RUNS_DIR / filename
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")

def main():
    print("Loading data from logs...")
    data, overall = load_data()
    
    # We will generate 4 comparison plots:
    # 1. Soft Omission - Strict Accuracy (Aligned vs Unaligned)
    # 2. Strong Omission - Strict Accuracy (Aligned vs Unaligned)
    # 3. Soft Omission - Standard Accuracy (Aligned vs Unaligned)
    # 4. Strong Omission - Standard Accuracy (Aligned vs Unaligned)
    
    print("Generating Strict Accuracy Plots...")
    plot_aligned_vs_unaligned(data, overall, "soft", "strict", "case_roles_soft_strict_aligned_vs_unaligned.png")
    plot_aligned_vs_unaligned(data, overall, "strong", "strict", "case_roles_strong_strict_aligned_vs_unaligned.png")
    
    print("Generating Standard Accuracy Plots...")
    plot_aligned_vs_unaligned(data, overall, "soft", "standard", "case_roles_soft_standard_aligned_vs_unaligned.png")
    plot_aligned_vs_unaligned(data, overall, "strong", "standard", "case_roles_strong_standard_aligned_vs_unaligned.png")

if __name__ == "__main__":
    main()
