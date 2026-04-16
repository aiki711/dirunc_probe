import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    base_dir = "runs/exp4_static/query"
    layers = [2, 4, 6, 8, 10, 15, 20, 25]
    
    records = []
    
    for layer in layers:
        log_file = os.path.join(base_dir, f"layer_{layer}", "log.jsonl")
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} does not exist. Skipping layer {layer}.")
            continue
            
        best_record = None
        best_macro_f1 = -1.0
        
        with open(log_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if "macro_f1_posonly" in data:
                    if data["macro_f1_posonly"] > best_macro_f1:
                        best_macro_f1 = data["macro_f1_posonly"]
                        best_record = data
        
        if best_record:
            labels = best_record.get("macro_posonly_labels", ["who", "what", "when", "where", "why", "how", "which"])
            per_label_f1 = best_record.get("per_label_f1", [])
            
            # Record Macro F1
            records.append({
                "Layer": layer,
                "Metric": "Macro F1",
                "F1 Score": best_macro_f1,
                "Type": "Overall"
            })
            
            # Record Per-Class F1
            if len(per_label_f1) == len(labels):
                for label, f1 in zip(labels, per_label_f1):
                    records.append({
                        "Layer": layer,
                        "Metric": f"{label}",
                        "F1 Score": f1,
                        "Type": "Class"
                    })
    
    if not records:
        print("No valid records found.")
        return
        
    df = pd.DataFrame(records)
    
    plt.figure(figsize=(12, 8))
    
    # Plot classes first
    class_df = df[df["Type"] == "Class"]
    sns.lineplot(data=class_df, x="Layer", y="F1 Score", hue="Metric", marker="o", dashes=False, palette="tab10", alpha=0.7)
    
    # Plot Marco F1 on top with thicker line and different style
    overall_df = df[df["Type"] == "Overall"]
    plt.plot(overall_df["Layer"], overall_df["F1 Score"], color="black", linewidth=3, marker="s", markersize=8, label="Macro F1 (Overall)", zorder=10)
    
    plt.title("6W1H Prediction F1 Score across Gemma-2-2b-it Layers", fontsize=16)
    plt.xlabel("Layer Depth", fontsize=14)
    plt.ylabel("F1 Score (Pos-only)", fontsize=14)
    
    plt.xticks(layers)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.ylim(0.0, 1.0)
    
    # Customize legend
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, title="Metrics", loc="lower right", fontsize=12, title_fontsize=13)
    
    out_path = "runs/exp4_static/layer_trajectory.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
