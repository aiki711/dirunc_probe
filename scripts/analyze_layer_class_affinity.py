#!/usr/bin/env python3
"""
レイヤーとクラスの親和性分析スクリプト

各Transformerレイヤーがどのdirectional uncertaintyクラス（who, what, when, where, why, how, which）
に強いかを分析し、可視化します。
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

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


def extract_layer_class_performance(summary: Dict[str, Any], mode: str = "query") -> pd.DataFrame:
    """
    Extract per-class F1 scores for each layer
    
    Returns:
        DataFrame with layers as rows and classes as columns
    """
    data = {}
    
    for key, value in summary.items():
        if key.startswith(f"{mode}/layer_"):
            layer_idx = int(key.split("_")[-1])
            best_data = value.get("best", {})
            per_label_f1 = best_data.get("per_label_f1", [])
            labels = best_data.get("macro_posonly_labels", DIRS)
            
            if per_label_f1:
                data[layer_idx] = {
                    label: f1 for label, f1 in zip(labels, per_label_f1)
                }
    
    # Convert to DataFrame
    df = pd.DataFrame(data).T
    df.index.name = 'Layer'
    df = df.sort_index()
    
    return df


def analyze_class_best_layers(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each class, find the best and second-best layers
    
    Returns:
        DataFrame with class analysis
    """
    analysis = []
    
    for class_name in df.columns:
        scores = df[class_name].sort_values(ascending=False)
        
        best_layer = scores.index[0]
        best_score = scores.iloc[0]
        
        second_layer = scores.index[1] if len(scores) > 1 else None
        second_score = scores.iloc[1] if len(scores) > 1 else 0.0
        
        worst_layer = scores.index[-1]
        worst_score = scores.iloc[-1]
        
        improvement = best_score - worst_score
        
        analysis.append({
            'Class': class_name,
            'Best Layer': best_layer,
            'Best F1': best_score,
            'Second Layer': second_layer,
            'Second F1': second_score,
            'Worst Layer': worst_layer,
            'Worst F1': worst_score,
            'Improvement': improvement,
        })
    
    return pd.DataFrame(analysis)


def plot_layer_class_heatmap(df: pd.DataFrame, output_path: Path):
    """Plot heatmap of Layer × Class F1 scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(df.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels([f'Layer {i}' for i in df.index])
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            color = "white" if val > 0.5 else "black"
            text = ax.text(j, i, f'{val:.3f}',
                          ha="center", va="center", color=color, fontsize=9)
    
    ax.set_title('Layer-Class F1 Score Heatmap', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Direction Class', fontsize=12)
    ax.set_ylabel('Transformer Layer', fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('F1 Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_class_layer_profiles(df: pd.DataFrame, output_path: Path):
    """Plot line graph showing each class's performance across layers"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for class_name in df.columns:
        ax.plot(df.index, df[class_name], marker='o', label=class_name, linewidth=2)
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class Performance Across Layers', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df.index)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_layer_specialization(df: pd.DataFrame, output_path: Path):
    """Plot which class each layer is best at (relative to other classes)"""
    # Normalize each layer's scores to show relative strength
    df_norm = df.div(df.max(axis=1), axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Stacked bar chart
    bottom = np.zeros(len(df_norm))
    colors = plt.cm.Set3(np.linspace(0, 1, len(df_norm.columns)))
    
    for idx, class_name in enumerate(df_norm.columns):
        ax.bar(df_norm.index, df_norm[class_name], bottom=bottom, 
               label=class_name, color=colors[idx], alpha=0.8)
        bottom += df_norm[class_name]
    
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Relative Strength (Normalized)', fontsize=12)
    ax.set_title('Layer Specialization Profile', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    ax.set_xticks(df_norm.index)
    ax.set_xticklabels([f'L{i}' for i in df_norm.index])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_analysis_report(df: pd.DataFrame, class_analysis: pd.DataFrame, output_path: Path):
    """Generate a markdown report with analysis insights"""
    
    report = []
    report.append("# レイヤー-クラス親和性分析レポート\n")
    report.append("## 概要\n")
    report.append("各Transformerレイヤーがどのdirectional uncertaintyクラスに強いかを分析しました。\n")
    
    # Overall best layer
    layer_avg = df.mean(axis=1).sort_values(ascending=False)
    best_layer = layer_avg.index[0]
    best_avg = layer_avg.iloc[0]
    
    report.append(f"\n### 全体的な最良レイヤー\n")
    report.append(f"- **Layer {best_layer}**: 平均F1 = {best_avg:.4f}\n")
    
    # Class-specific insights
    report.append(f"\n## クラスごとの分析\n")
    
    for _, row in class_analysis.iterrows():
        report.append(f"\n### {row['Class'].upper()}\n")
        report.append(f"- **最良レイヤー**: Layer {row['Best Layer']} (F1 = {row['Best F1']:.4f})\n")
        if row['Second Layer'] is not None:
            report.append(f"- **次点レイヤー**: Layer {row['Second Layer']} (F1 = {row['Second F1']:.4f})\n")
        report.append(f"- **最悪レイヤー**: Layer {row['Worst Layer']} (F1 = {row['Worst F1']:.4f})\n")
        report.append(f"- **改善幅**: +{row['Improvement']:.4f} ({row['Improvement']/row['Worst F1']*100:.1f}%)\n")
    
    # Layer characterization
    report.append(f"\n## レイヤーの特性\n")
    
    for layer in df.index:
        layer_scores = df.loc[layer].sort_values(ascending=False)
        best_class = layer_scores.index[0]
        best_score = layer_scores.iloc[0]
        
        report.append(f"\n### Layer {layer}\n")
        report.append(f"- **最も強いクラス**: {best_class} (F1 = {best_score:.4f})\n")
        report.append(f"- **全クラスの平均**: F1 = {df.loc[layer].mean():.4f}\n")
        
        # Top 3 classes
        top3 = layer_scores.head(3)
        report.append(f"- **トップ3**: {', '.join([f'{c} ({s:.3f})' for c, s in top3.items()])}\n")
    
    # Insights
    report.append(f"\n## 重要な知見\n")
    
    # Identify shallow/deep preferences
    shallow_classes = []
    deep_classes = []
    
    for _, row in class_analysis.iterrows():
        if row['Best Layer'] <= 10:
            shallow_classes.append(row['Class'])
        elif row['Best Layer'] >= 20:
            deep_classes.append(row['Class'])
    
    if shallow_classes:
        report.append(f"\n### 浅層レイヤーに強いクラス\n")
        report.append(f"- {', '.join(shallow_classes)}\n")
        report.append(f"- **解釈**: 表面的・構文的な5W1H情報は浅層で捉えられる\n")
    
    if deep_classes:
        report.append(f"\n### 深層レイヤーに強いクラス\n")
        report.append(f"- {', '.join(deep_classes)}\n")
        report.append(f"- **解釈**: 抽象的・意味的な5W1H情報は深層で捉えられる\n")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze layer-class affinity')
    parser.add_argument('--summary', type=str, required=True,
                       help='Path to summary.json (e.g., runs/mixed_llm_gemma2b_perclass)')
    parser.add_argument('--output_dir', type=str, default='analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--mode', type=str, default='query',
                       help='Mode to analyze (query or baseline)')
    
    args = parser.parse_args()
    
    # Load data
    summary_path = Path(args.summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading summary from: {summary_path}")
    summary = load_summary(summary_path)
    
    # Extract layer-class performance
    print("\nExtracting layer-class performance...")
    df = extract_layer_class_performance(summary, args.mode)
    
    print(f"\nLayer-Class F1 Scores:")
    print(df.to_string())
    
    # Analyze best layers per class
    print("\nAnalyzing best layers per class...")
    class_analysis = analyze_class_best_layers(df)
    
    print(f"\nClass Analysis:")
    print(class_analysis.to_string(index=False))
    
    # Save raw data
    df.to_csv(output_dir / "layer_class_f1_matrix.csv")
    class_analysis.to_csv(output_dir / "class_best_layers.csv", index=False)
    print(f"\nSaved raw data to {output_dir}/")
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===\n")
    
    plot_layer_class_heatmap(df, output_dir / "layer_class_heatmap.png")
    plot_class_layer_profiles(df, output_dir / "class_layer_profiles.png")
    plot_layer_specialization(df, output_dir / "layer_specialization.png")
    
    # Generate report
    print("\n=== Generating Analysis Report ===\n")
    generate_analysis_report(df, class_analysis, output_dir / "layer_class_analysis.md")
    
    print(f"\nAll analysis results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
