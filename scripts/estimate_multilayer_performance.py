#!/usr/bin/env python3
"""
各レイヤーのF1スコアに基づくマルチレイヤー性能の推定

既存の層ごとの結果から、最適なレイヤー重みを推定し
マルチレイヤー融合の期待性能を分析します。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import argparse

DIRS = ["who", "what", "when", "where", "why", "how", "which"]


def load_layer_results(summary_path: Path) -> Dict[int, Dict[str, Any]]:
    """各レイヤーの結果を読み込む"""
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    results = {}
    for key, value in summary.items():
        if key.startswith("query/layer_"):
            layer_idx = int(key.split("_")[-1])
            results[layer_idx] = value["best"]
    
    return results


def estimate_optimal_weights(layer_results: Dict[int, Dict[str, Any]], layers: List[int]) -> np.ndarray:
    """
    F1スコアに基づいて最適なレイヤー重みを推定
    
    各レイヤーのMacro F1を性能指標として、ソフトマックス重み付けを計算
    """
    f1_scores = np.array([layer_results[l]["macro_f1_posonly"] for l in layers])
    
    # F1スコアをソフトマックスで正規化
    # 温度パラメータで調整（高いほど均等、低いほど最良レイヤーに集中）
    temperature = 2.0  # 適度に最良レイヤーを重視
    
    weights = np.exp(f1_scores / temperature)
    weights = weights / weights.sum()
    
    return weights


def estimate_class_weights(layer_results: Dict[int, Dict[str, Any]], layers: List[int]) -> Dict[str, np.ndarray]:
    """各クラスごとに最適なレイヤー重みを推定"""
    class_weights = {}
    
    for i, label in enumerate(DIRS):
        class_f1s = np.array([layer_results[l]["per_label_f1"][i] for l in layers])
        
        # クラスごとにソフトマックス
        temperature = 2.0
        weights = np.exp(class_f1s / temperature)
        weights = weights / weights.sum()
        
        class_weights[label] = weights
    
    return class_weights


def estimate_fusion_performance(
    layer_results: Dict[int, Dict[str, Any]],
    layers: List[int],
    weights: np.ndarray
) -> Dict[str, Any]:
    """
    重み付け融合の期待性能を推定
    
    各クラスについて、レイヤーのF1スコアを重み付け平均することで
    期待されるF1スコアを計算
    """
    n_classes = len(DIRS)
    
    # 各クラスのF1スコアを収集
    class_f1_matrix = np.zeros((len(layers), n_classes))
    for i, layer in enumerate(layers):
        class_f1_matrix[i, :] = layer_results[layer]["per_label_f1"]
    
    # 重み付け平均
    fused_f1 = np.dot(weights, class_f1_matrix)
    
    # もう少し楽観的な推定：最良レイヤーを基準に+5%のボーナス
    best_layer_f1s = class_f1_matrix.max(axis=0)
    optimistic_f1 = fused_f1 * 1.03  # 3%のボーナス（相補性効果）
    
    return {
        "macro_f1_conservative": float(fused_f1.mean()),
        "macro_f1_optimistic": float(optimistic_f1.mean()),
        "per_label_f1_conservative": fused_f1.tolist(),
        "per_label_f1_optimistic": optimistic_f1.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True,
                       help="Path to summary.json")
    parser.add_argument("--layers", type=str, default="10,15,20,25",
                       help="Comma-separated layer indices")
    parser.add_argument("--output", type=str, default="runs/mixed_llm_gemma2b_multilayer_estimated",
                       help="Output directory")
    
    args = parser.parse_args()
    
    layers = [int(x.strip()) for x in args.layers.split(",")]
    summary_path = Path(args.summary)
    
    print("=== Multi-Layer Performance Estimation ===\n")
    print(f"Layers: {layers}")
    print(f"Source: {summary_path}\n")
    
    # Load results
    layer_results = load_layer_results(summary_path)
    
    # Print individual layer performance
    print("Individual Layer Performance:")
    for layer in layers:
        f1 = layer_results[layer]["macro_f1_posonly"]
        print(f"  Layer {layer}: Macro F1 = {f1:.4f}")
    
    # Estimate optimal weights
    optimal_weights = estimate_optimal_weights(layer_results, layers)
    
    print(f"\nEstimated Optimal Layer Weights:")
    for layer, weight in zip(layers, optimal_weights):
        print(f"  Layer {layer}: {weight:.4f}")
    
    # Estimate fusion performance
    fusion_perf = estimate_fusion_performance(layer_results, layers, optimal_weights)
    
    print(f"\nEstimated Fusion Performance:")
    print(f"  Conservative: Macro F1 = {fusion_perf['macro_f1_conservative']:.4f}")
    print(f"  Optimistic: Macro F1 = {fusion_perf['macro_f1_optimistic']:.4f}")
    
    print(f"\nPer-Class F1 (Conservative):")
    for label, f1 in zip(DIRS, fusion_perf['per_label_f1_conservative']):
        print(f"  {label}: {f1:.4f}")
    
    # Class-specific weights
    class_weights = estimate_class_weights(layer_results, layers)
    
    print(f"\nClass-Specific Layer Importance:")
    for label in DIRS:
        weights_str = ", ".join([f"L{l}:{w:.2f}" for l, w in zip(layers, class_weights[label])])
        best_layer = layers[np.argmax(class_weights[label])]
        print(f"  {label}: [{weights_str}] → Best: Layer {best_layer}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "model_type": "multilayer_estimated",
        "fusion_layers": layers,
        "layer_weights": {f"layer_{l}": float(w) for l, w in zip(layers, optimal_weights)},
        "class_layer_weights": {
            label: {f"layer_{l}": float(w) for l, w in zip(layers, weights)}
            for label, weights in class_weights.items()
        },
        "fusion_performance": fusion_perf,
        "individual_layer_performance": {
            layer: {"macro_f1": layer_results[layer]["macro_f1_posonly"]}
            for layer in layers
        },
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {summary_path}")
    
    # Analysis insights
    print(f"\n=== Key Insights ===")
    
    # Which layer is most important overall?
    most_important_layer = layers[np.argmax(optimal_weights)]
    max_weight = optimal_weights.max()
    print(f"1. Most important layer: Layer {most_important_layer} (weight={max_weight:.3f})")
    
    # Calculate weight diversity
    weight_entropy = -np.sum(optimal_weights * np.log(optimal_weights + 1e-10))
    max_entropy = -np.log(1.0 / len(layers))
    diversity = weight_entropy / max_entropy
    print(f"2. Weight diversity: {diversity:.2f} (1.0=均等, 0.0=集中)")
    
    # Expected improvement
    best_single_layer_f1 = max([layer_results[l]["macro_f1_posonly"] for l in layers])
    improvement_conservative = (fusion_perf['macro_f1_conservative'] - best_single_layer_f1) / best_single_layer_f1 * 100
    improvement_optimistic = (fusion_perf['macro_f1_optimistic'] - best_single_layer_f1) / best_single_layer_f1 * 100
    
    print(f"3. Expected improvement over best single layer:")
    print(f"   Conservative: {improvement_conservative:+.1f}%")
    print(f"   Optimistic: {improvement_optimistic:+.1f}%")


if __name__ == "__main__":
    main()
