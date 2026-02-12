#!/usr/bin/env python3
"""
プローブモデル結果の可視化スクリプト

ベースライン、クラスごと閾値最適化、マルチレイヤー融合の結果を比較可視化します。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use Agg backend for headless environments
matplotlib.use('Agg')

# Direction labels
DIRS = ["who", "what", "when", "where", "why", "how", "which"]


def load_summary(path: Path) -> Dict[str, Any]:
    """Load summary.json from a run directory"""
    summary_file = path / "summary.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_layer_scores(summary: Dict[str, Any], mode: str = "query") -> Dict[int, float]:
    """Extract macro_f1_posonly scores for each layer"""
    scores = {}
    for key, value in summary.items():
        if key.startswith(f"{mode}/layer_"):
            layer_idx = int(key.split("_")[-1])
            best_data = value.get("best", {})
            score = best_data.get("macro_f1_posonly", 0.0)
            scores[layer_idx] = score
    return scores


def extract_class_f1(summary: Dict[str, Any], mode: str = "query", layer: Optional[int] = None) -> Dict[str, float]:
    """Extract per-class F1 scores for the best or specified layer"""
    if layer is None:
        # Use best layer
        best_key = f"{mode}/best"
        if best_key in summary:
            layer = summary[best_key].get("layer_idx")
    
    if layer is None:
        return {}
    
    key = f"{mode}/layer_{layer}"
    if key not in summary:
        return {}
    
    best_data = summary[key].get("best", {})
    per_label_f1 = best_data.get("per_label_f1", [])
    labels = best_data.get("macro_posonly_labels", DIRS)
    
    return {label: f1 for label, f1 in zip(labels, per_label_f1)}


def extract_thresholds(summary: Dict[str, Any], mode: str = "query", layer: Optional[int] = None) -> Dict[str, float]:
    """Extract per-class thresholds"""
    if layer is None:
        best_key = f"{mode}/best"
        if best_key in summary:
            layer = summary[best_key].get("layer_idx")
    
    if layer is None:
        return {}
    
    key = f"{mode}/layer_{layer}"
    if key not in summary:
        return {}
    
    best_data = summary[key].get("best", {})
    per_class_tuned = best_data.get("per_class_tuned", {})
    threshold_dict = per_class_tuned.get("threshold_dict", {})
    
    return threshold_dict


def plot_layer_comparison(baseline_summary: Dict[str, Any], 
                         perclass_summary: Dict[str, Any],
                         output_path: Path):
    """Plot layer-wise macro F1 comparison"""
    baseline_scores = extract_layer_scores(baseline_summary)
    perclass_scores = extract_layer_scores(perclass_summary)
    
    layers = sorted(set(list(baseline_scores.keys()) + list(perclass_scores.keys())))
    
    baseline_values = [baseline_scores.get(l, 0.0) for l in layers]
    perclass_values = [perclass_scores.get(l, 0.0) for l in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, baseline_values, marker='o', label='Baseline (単一閾値)', linewidth=2)
    plt.plot(layers, perclass_values, marker='s', label='Per-Class (クラスごと閾値)', linewidth=2)
    
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Macro F1 (pos only)', fontsize=12)
    plt.title('Layer-wise Macro F1 Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_class_f1_comparison(baseline_summary: Dict[str, Any],
                            perclass_summary: Dict[str, Any],
                            output_path: Path):
    """Plot per-class F1 comparison"""
    baseline_f1 = extract_class_f1(baseline_summary)
    perclass_f1 = extract_class_f1(perclass_summary)
    
    labels = DIRS
    x = np.arange(len(labels))
    width = 0.35
    
    baseline_values = [baseline_f1.get(l, 0.0) for l in labels]
    perclass_values = [perclass_f1.get(l, 0.0) for l in labels]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, perclass_values, width, label='Per-Class', alpha=0.8)
    
    ax.set_xlabel('Direction Class', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_threshold_heatmap(perclass_summary: Dict[str, Any], output_path: Path):
    """Plot threshold heatmap for per-class optimization"""
    thresholds = extract_thresholds(perclass_summary)
    
    if not thresholds:
        print("No per-class thresholds found, skipping heatmap")
        return
    
    labels = DIRS
    values = [thresholds.get(l, 0.5) for l in labels]
    
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow([values], cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks([0])
    ax.set_yticklabels(['Threshold'])
    
    # Add text annotations
    for i, val in enumerate(values):
        text = ax.text(i, 0, f'{val:.2f}',
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax.set_title('Optimized Thresholds per Direction Class', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Threshold Value')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize probe model results')
    parser.add_argument('--baseline', type=str, required=True,
                       help='Path to baseline results directory')
    parser.add_argument('--perclass', type=str, required=True,
                       help='Path to per-class threshold results directory')
    parser.add_argument('--multilayer', type=str, default=None,
                       help='Path to multi-layer results directory (optional)')
    parser.add_argument('--output_dir', type=str, default='visualization',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Load summaries
    baseline_path = Path(args.baseline)
    perclass_path = Path(args.perclass)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading summaries...")
    baseline_summary = load_summary(baseline_path)
    perclass_summary = load_summary(perclass_path)
    
    print("\n=== Generating Visualizations ===\n")
    
    # Generate plots
    plot_layer_comparison(
        baseline_summary,
        perclass_summary,
        output_dir / "layer_comparison.png"
    )
    
    plot_class_f1_comparison(
        baseline_summary,
        perclass_summary,
        output_dir / "class_f1_comparison.png"
    )
    
    plot_threshold_heatmap(
        perclass_summary,
        output_dir / "threshold_heatmap.png"
    )
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
