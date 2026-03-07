import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Fix plotting issues without X server
import matplotlib
matplotlib.use('Agg')

DIRS = ["who", "what", "when", "where", "why", "how", "which"]

def plot_comparison(success_id, failure_id, success_data, failure_data, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharey=True)
    
    # 1. Plot Success Case
    s_traj = success_data[success_id]["trajectory"]
    s_tokens = [step["token_str"] for step in s_traj]
    s_label = success_data[success_id].get("label", "where")
    s_indices = np.arange(len(s_tokens))
    
    ax0 = axes[0]
    for d in DIRS:
        probs = [step["probs"][d] * 100 for step in s_traj]
        alpha = 1.0 if d == s_label else 0.3
        linewidth = 3 if d == s_label else 1
        ax0.plot(s_indices, probs, label=d.upper(), alpha=alpha, linewidth=linewidth, marker='o' if d == s_label else None)
    
    ax0.set_xticks(s_indices)
    ax0.set_xticklabels(s_tokens, rotation=45, ha='right')
    ax0.set_title(f"SUCCESS Case: {success_id}\nTarget: {s_label.upper()} (Probability drops as info is filled)")
    ax0.set_ylabel("Missing Prob (%)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='upper right', fontsize='small', ncol=2)

    # 2. Plot Failure Case
    f_traj = failure_data[failure_id]["trajectory"]
    f_tokens = [step["token_str"] for step in f_traj]
    f_label = failure_data[failure_id]["label"]
    f_indices = np.arange(len(f_tokens))
    
    ax1 = axes[1]
    for d in DIRS:
        probs = [step["probs"][d] * 100 for step in f_traj]
        alpha = 1.0 if d == f_label else 0.3
        linewidth = 3 if d == f_label else 1
        ax1.plot(f_indices, probs, label=d.upper(), alpha=alpha, linewidth=linewidth, marker='x' if d == f_label else None)
    
    ax1.set_xticks(f_indices)
    ax1.set_xticklabels(f_tokens, rotation=45, ha='right')
    ax1.set_title(f"FAILURE Case: {failure_id}\nTarget: {f_label.upper()} (Probability stays high despite info presence)")
    ax1.set_ylabel("Missing Prob (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize='small', ncol=2)

    plt.tight_layout()
    plot_path = out_dir / f"comparison_{success_id}_vs_{failure_id}.png"
    plt.savefig(plot_path, dpi=150)
    plt.savefig(plot_path.with_suffix(".pdf"))
    plt.close()
    print(f"Saved comparison plot to {plot_path}")

def main():
    out_dir = Path("runs/balanced/experiment9_visuals")
    success_file = out_dir / "trajectory_data.json"
    failure_file = out_dir / "failure_trajectory_data.json"
    
    if not success_file.exists() or not failure_file.exists():
        print("Required data files missing.")
        return

    with open(success_file, "r") as f:
        success_data = json.load(f)
    with open(failure_file, "r") as f:
        failure_data = json.load(f)

    # Manually map labels for success cases
    success_data["exp9_where_airport"]["label"] = "where"
    success_data["exp9_how_shared_ride"]["label"] = "how"
    success_data["exp9_time_tonight"]["label"] = "when"

    # Generate comparisons
    plot_comparison("exp9_where_airport", "fail_who_12_00017", success_data, failure_data, out_dir)
    
if __name__ == "__main__":
    main()
