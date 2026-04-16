import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
from pathlib import Path

DIRS = ["who", "what", "when", "where", "why", "how", "which"]
COLORS = {
    "who": "#1f77b4", "what": "#ff7f0e", "when": "#2ca02c",
    "where": "#d62728", "why": "#9467bd", "how": "#8c564b", "which": "#e377c2"
}

def plot_trajectory_line(target_id, data, out_dir):
    traj = data["trajectory"]
    tokens = [step["token_str"] for step in traj]
    x_indices = np.arange(len(tokens))
    
    plt.figure(figsize=(12, 6))
    
    for d in DIRS:
        probs = [step["probs"][d] * 100 for step in traj]
        
        # Highlight WHERE and HOW for exp9_how_shared_ride
        lw = 3 if d in ["where", "how"] and "shared_ride" in target_id else 1
        alpha = 1.0 if d in ["where", "how"] and "shared_ride" in target_id else 0.4
        
        plt.plot(x_indices, probs, label=d.upper(), color=COLORS[d], 
                 linewidth=lw, alpha=alpha, marker='o', markersize=4)

    plt.xticks(x_indices, tokens, rotation=45, ha="right", fontsize=10)
    plt.ylim(0, 105)
    plt.ylabel("Missing Probability (%)", fontsize=12)
    plt.title(f"Dynamic Checklist Update: '{data['text']}'", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    
    plt.savefig(out_dir / f"{target_id}_line.png", dpi=300)
    plt.savefig(out_dir / f"{target_id}_line.pdf")
    plt.close()

def plot_heatmap(target_id, data, out_dir):
    traj = data["trajectory"]
    tokens = [step["token_str"] for step in traj]
    
    # Create matrix: [Labels, Tokens]
    matrix = np.zeros((len(DIRS), len(tokens)))
    for j, step in enumerate(traj):
        for i, d in enumerate(DIRS):
            matrix[i, j] = step["probs"][d] * 100
            
    plt.figure(figsize=(14, len(DIRS) * 0.8 + 2))
    ax = sns.heatmap(matrix, cmap="YlOrRd", annot=True, fmt=".0f", 
                     xticklabels=tokens, yticklabels=[d.upper() for d in DIRS],
                     cbar_kws={'label': 'Missing Probability (%)'})
    
    plt.xticks(rotation=45, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.title(f"Checklist Alert State Matrix: '{data['text']}'", fontsize=14, pad=20)
    plt.tight_layout()
    
    plt.savefig(out_dir / f"{target_id}_heatmap.png", dpi=300)
    plt.savefig(out_dir / f"{target_id}_heatmap.pdf")
    plt.close()

def plot_radar(target_id, data, out_dir):
    traj = data["trajectory"]
    if len(traj) < 6: return
    
    idx_early = 5  # e.g., "Could you book me a"
    idx_late = len(traj) - 1 # End of phrase
    
    early_step = traj[idx_early]
    late_step = traj[idx_late]
    
    early_text = "".join([t["token_str"] for t in traj[:idx_early+1]])
    
    labels = [d.upper() for d in DIRS]
    early_vals = [early_step["probs"][d] * 100 for d in DIRS]
    late_vals = [late_step["probs"][d] * 100 for d in DIRS]
    
    # Close the polygons
    early_vals += [early_vals[0]]
    late_vals += [late_vals[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot Early state
    ax.plot(angles, early_vals, color='#1f77b4', linewidth=2, linestyle='solid', label=f"Early: '{early_text}...'")
    ax.fill(angles, early_vals, color='#1f77b4', alpha=0.25)
    
    # Plot Late state
    ax.plot(angles, late_vals, color='#d62728', linewidth=2, linestyle='solid', label=f"Final: '{data['text']}'")
    ax.fill(angles, late_vals, color='#d62728', alpha=0.25)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="grey", size=10)
    ax.set_ylim(0, 100)

    plt.title(f"Checklist Morphing (Radar): Status Shift", size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    
    plt.savefig(out_dir / f"{target_id}_radar.png", dpi=300)
    plt.savefig(out_dir / f"{target_id}_radar.pdf")
    plt.close()

def main():
    out_dir = Path("runs/balanced/experiment9_visuals")
    data_file = out_dir / "trajectory_data.json"
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
        
    with open(data_file, "r") as f:
        results = json.load(f)
        
    for target_id, data in results.items():
        print(f"Generating visual proofs for: {target_id}")
        plot_trajectory_line(target_id, data, out_dir)
        plot_heatmap(target_id, data, out_dir)
        plot_radar(target_id, data, out_dir)
        
    print(f"All visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
