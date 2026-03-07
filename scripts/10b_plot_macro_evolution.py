import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# Common label ordering
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

def main():
    input_dir = Path("runs/balanced/experiment10_macro")
    data_file = input_dir / "macro_trajectory.json"
    
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
        axis_label = f"T{turn_id}: {speaker}\n'{short_text}'"
        labels.append(axis_label)
        
        # To show the text clearly, we'll also collect it separately
        text_labels.append(f"Turn {turn_id} ({speaker}): {added_text}")
        
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
    
    # Plotting
    sns.set_theme(style="whitegrid")
    # Increase figure size significantly to accommodate labels and text
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Create the line plot
    sns.lineplot(
        data=df,
        x="Axis_Label",
        y="Missing_Probability",
        hue="Label",
        marker="o",
        markersize=10,
        linewidth=3,
        palette="tab10",
        ax=ax
    )
    
    # Customize axes and title
    ax.set_title("Checklist Evolution Across Dialogue Turns\n(Probability of Missing Information)", fontsize=18, fontweight="bold", pad=20)
    ax.set_ylabel("Probability of Missing Information", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dialogue Turn", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    
    # Improve X-axis labels readability
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    # Enhance the legend
    plt.legend(title="Checklist Item", title_fontsize=12, fontsize=11, loc="upper right", bbox_to_anchor=(1.15, 1))
    
    # Add a horizontal line at 0.5 (often considered a threshold)
    ax.axhline(0.5, ls='--', color='gray', alpha=0.5)
    
    # Add the full dialogue text as a legend/box at the bottom or side
    dialogue_text = "\n".join(text_labels)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, dialogue_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    
    out_file = input_dir / "macro_evolution_line.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved Macro Evolution Line Chart to {out_file}")
    
    plt.close()

if __name__ == "__main__":
    main()
