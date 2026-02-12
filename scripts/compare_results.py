#!/usr/bin/env python3
"""
3つの実験結果（ベースライン、クラスごと閾値、マルチレイヤー）を比較分析するスクリプト

summary.jsonを読み込み、F1スコアの改善度や閾値の変化、レイヤー重みなどを比較します。
"""

import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

DIRS = ["who", "what", "when", "where", "why", "how", "which"]

def load_summary(run_dir: str) -> Dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def extract_metrics(summary: Dict[str, Any], exp_type: str) -> Dict[str, Any]:
    if not summary:
        return {}
    
    metrics = {}
    
    # Best overall result
    best = summary.get("best_overall", {})
    if exp_type == "multilayer":
        # summary.jsonの構造が少し異なる場合に対応
        if "multilayer" in summary:
            best = summary["multilayer"].get("best", {})
            metrics["layer_weights"] = summary["multilayer"].get("best", {}).get("layer_weights", {})
        elif "best" in summary:
             best = summary["best"]
             metrics["layer_weights"] = summary.get("layer_weights", {})
    elif exp_type == "perclass":
         # Per-class experiment usually has best_overall
         if "best_overall" in summary:
             best = summary["best_overall"].get("best", {}) # Nested best in best_overall?
             # Actually scripts/03_train_probe.py structure:
             # results["best_overall"] = {score, mode, layer_idx, best: {...}}
             if "best" in summary["best_overall"]:
                 best = summary["best_overall"]["best"]
    else: # baseline
         if "best_overall" in summary:
             if "best" in summary["best_overall"]:
                 best = summary["best_overall"]["best"]

    metrics["macro_f1"] = best.get("macro_f1_posonly", 0.0)
    metrics["per_label_f1"] = best.get("per_label_f1", [0]*len(DIRS))
    metrics["thresholds"] = best.get("threshold_dict", {})
    
    # Get optimal layer if available
    if "best_overall" in summary:
        metrics["best_layer"] = summary["best_overall"].get("layer_idx")
    
    return metrics

def plot_comparison(results: Dict[str, Dict[str, Any]], out_dir: Path):
    # 1. Macro F1 Comparison
    f1_data = []
    for name, res in results.items():
        if res:
            f1_data.append({"Experiment": name, "Macro F1": res["macro_f1"]})
    
    if f1_data:
        df_f1 = pd.DataFrame(f1_data)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_f1, x="Experiment", y="Macro F1")
        plt.title("Macro F1 Score Comparison")
        plt.ylim(0, 1.0)
        for i, v in enumerate(df_f1["Macro F1"]):
            plt.text(i, v + 0.02, f"{v:.4f}", ha='center')
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_macro_f1.png")
        plt.close()

    # 2. Per-Class F1 Comparison
    class_data = []
    for name, res in results.items():
        if res and "per_label_f1" in res:
            for i, label in enumerate(DIRS):
                if i < len(res["per_label_f1"]):
                    class_data.append({
                        "Experiment": name,
                        "Class": label,
                        "F1 Score": res["per_label_f1"][i]
                    })
    
    if class_data:
        df_class = pd.DataFrame(class_data)
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_class, x="Class", y="F1 Score", hue="Experiment")
        plt.title("Per-Class F1 Score Comparison")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(out_dir / "comparison_per_class_f1.png")
        plt.close()

    # 3. Layer Weights (Multilayer only)
    if "Multilayer" in results and "layer_weights" in results["Multilayer"]:
        weights = results["Multilayer"]["layer_weights"]
        if weights: # could be list or dict
             if isinstance(weights, dict):
                 # extract layer numbers
                 w_data = [{"Layer": int(k.split('_')[-1]) if '_' in k else k, "Weight": v} for k, v in weights.items()]
             elif isinstance(weights, list):
                  # Assuming order matches experimentation
                  pass # hard to know layer ids easily without more info
                  w_data = [] 

             if w_data:
                 df_w = pd.DataFrame(w_data)
                 # Sort by layer index if possible
                 try:
                     df_w["LayerInt"] = pd.to_numeric(df_w["Layer"])
                     df_w = df_w.sort_values("LayerInt")
                 except:
                     pass
                 
                 plt.figure(figsize=(8, 5))
                 sns.barplot(data=df_w, x="Layer", y="Weight")
                 plt.title("Learned Layer Weights (Multilayer)")
                 plt.tight_layout()
                 plt.savefig(out_dir / "multilayer_weights.png")
                 plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, required=True)
    parser.add_argument("--perclass", type=str, required=True)
    parser.add_argument("--multilayer", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="analysis_results/comparison")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load summaries
    s_base = load_summary(args.baseline)
    s_perc = load_summary(args.perclass)
    s_multi = load_summary(args.multilayer)
    
    # Extract metrics
    results = {
        "Baseline": extract_metrics(s_base, "baseline"),
        "Per-Class": extract_metrics(s_perc, "perclass"),
        "Multilayer": extract_metrics(s_multi, "multilayer"),
    }
    
    # Print Text Report
    report = []
    report.append("# Experiment Comparison Report\n")
    
    report.append("## 1. Macro F1 Scores")
    for name, res in results.items():
        if res:
            report.append(f"- **{name}**: {res['macro_f1']:.4f}")
            if "best_layer" in res and res["best_layer"] is not None:
                report.append(f"  - Best Single Layer: {res['best_layer']}")
    
    report.append("\n## 2. Improvements")
    base_f1 = results["Baseline"].get("macro_f1", 0)
    if base_f1 > 0:
        if results["Per-Class"]:
            imp = (results["Per-Class"]["macro_f1"] - base_f1) / base_f1 * 100
            report.append(f"- Per-Class vs Baseline: **{imp:+.2f}%**")
        if results["Multilayer"]:
            imp = (results["Multilayer"]["macro_f1"] - base_f1) / base_f1 * 100
            report.append(f"- Multilayer vs Baseline: **{imp:+.2f}%**")
        if results["Per-Class"] and results["Multilayer"]:
             perc_f1 = results["Per-Class"]["macro_f1"]
             imp = (results["Multilayer"]["macro_f1"] - perc_f1) / perc_f1 * 100
             report.append(f"- Multilayer vs Per-Class: **{imp:+.2f}%**")

    report.append("\n## 3. Multilayer Weights")
    if results["Multilayer"] and "layer_weights" in results["Multilayer"]:
         w = results["Multilayer"]["layer_weights"]
         report.append("Learned weights:")
         if isinstance(w, dict):
             for k, v in w.items():
                 report.append(f"- {k}: {v:.4f}")
    
    report_text = "\n".join(report)
    print(report_text)
    (out_dir / "report.md").write_text(report_text)
    
    # Generate Plots
    plot_comparison(results, out_dir)
    print(f"\nAnalysis saved to {out_dir}")

if __name__ == "__main__":
    main()
