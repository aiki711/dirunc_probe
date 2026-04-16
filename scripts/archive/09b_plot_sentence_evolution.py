import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path

# Common label ordering
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

def main():
    input_dir = Path("runs/balanced/experiment9_sentence")
    data_file = input_dir / "sentence_trajectory.json"
    
    if not data_file.exists():
        print(f"Error: {data_file} not found.")
        return
        
    with open(data_file, "r") as f:
        data = json.load(f)
        
    turns = data["turns"]
    evolution = data["macro_evolution"]
    
    # Prepare data for Seaborn
    records = []
    labels = []
    text_labels = []
    
    for item in evolution:
        turn_id = item["turn_id"]
        speaker = item["speaker"]
        added_text = item["added_text"]
        
        # Determine the label for the X-axis (Speaker + Truncated Text)
        short_text = added_text[:30] + "..." if len(added_text) > 30 else added_text
        axis_label = f"Sentence {turn_id}\n'{short_text}'"
        labels.append(axis_label)
        
        # To show the text clearly, we'll also collect it separately
        text_labels.append(f"Sentence {turn_id}: {added_text}")
        
        probs = item["probs"]
        for d in DIRS:
            if d in probs:
                records.append({
                    "Turn": turn_id,
                    "Axis_Label": axis_label,
                    "Label": d.upper(),
                    "Missing_Probability": probs[d]
                })
                
    df = pd.DataFrame(records)
    
    # Target labels to highlight
    TARGET_LABELS = ["WHO", "WHEN", "WHERE"]
    
    # Custom palette and alpha control
    palette = {}
    default_palette = sns.color_palette("tab10", n_colors=len(DIRS))
    for i, d in enumerate(DIRS):
        label = d.upper()
        palette[label] = default_palette[i]

    # Plotting
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create the line plot with custom alpha for non-targets
    for label in df["Label"].unique():
        label_df = df[df["Label"] == label]
        is_target = label in TARGET_LABELS
        alpha = 1.0 if is_target else 0.2
        linewidth = 4 if is_target else 1.5
        linestyle = "-" if is_target else "--"
        
        sns.lineplot(
            data=label_df,
            x="Axis_Label",
            y="Missing_Probability",
            marker="o",
            markersize=12 if is_target else 6,
            linewidth=linewidth,
            linestyle=linestyle,
            color=palette[label],
            alpha=alpha,
            label=label,
            ax=ax
        )
    
    # Customize axes and title
    ax.set_title("Checklist Evolution Across Dialogue Turns\n(Probability vs Empirical Decision Thresholds)", fontsize=18, fontweight="bold", pad=20)
    ax.set_ylabel("Probability of Missing Information", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dialogue Turn", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    
    # Empirical Thresholds
    # (WHEN is set to 0.40 to reflect the restaurant probe's range; WHO 0.15 provisional)
    THRESHOLDS = {
        "WHO": 0.15,
        "WHAT": 0.55,
        "WHEN": 0.40,
        "WHERE": 0.35,
        "HOW": 0.25,
        "WHICH": 0.45,
        "WHY": 0.00,
    }
    
    # Get colors used in lineplot to match threshold lines
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    color_map = {}
    for handle, label in zip(legend_handles, legend_labels):
        if hasattr(handle, 'get_color'):
            color_map[label] = handle.get_color()
        elif hasattr(handle, 'get_edgecolor'):
            color_map[label] = handle.get_edgecolor()
            
    # Add horizontal lines for thresholds
    for label, thresh in THRESHOLDS.items():
        if label in palette:
            is_target = label in TARGET_LABELS
            ax.axhline(thresh, ls=':', color=palette[label], alpha=0.3 if is_target else 0.1, linewidth=2)
            # Add a small text label for the threshold value near the right edge
            ax.text(len(labels) - 0.5, thresh + 0.01, f"Thresh:{thresh}", 
                    color=palette[label], fontsize=9, alpha=0.6 if is_target else 0.2, fontweight='bold')

    # Improve X-axis labels readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Enhance the legend
    plt.legend(title="Checklist Item (Highlighted: Targets, Faded: Others)", 
               title_fontsize=13, fontsize=12, loc="upper right", bbox_to_anchor=(1.25, 1))
    
    # Add the full dialogue text as a legend/box at the bottom or side
    dialogue_text = "\n".join(text_labels)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, dialogue_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    out_file = input_dir / "sentence_evolution_line.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved Sentence Evolution Line Chart to {out_file}")
    
    plt.close()

if __name__ == "__main__":
    main()
