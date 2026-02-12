#!/usr/bin/env python3
"""
マルチレイヤープローブ学習スクリプト（モデルベース）

既存の学習済みプローブモデルから予測確率を取得し、
学習可能な重みで融合して最適化します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any, Tuple
import sys
from tqdm import tqdm

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# 共通モジュールをインポート
from common import DIRS
from sklearn.metrics import f1_score


def set_seed(seed: int) -> None:
    """乱数シードを設定"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: Path) -> None:
    """ディレクトリを安全に作成"""
    path.mkdir(parents=True, exist_ok=True)


def load_dev_data(run_dir: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """開発セットのラベルとメタデータを読み込む"""
    # summary.jsonから開発セット情報を取得
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # 任意のレイヤーのログファイルからy_trueを取得
    for key in summary.keys():
        if key.startswith("query/layer_"):
            layer_dir = run_dir / key.replace("/", "/")
            log_file = layer_dir / "log.jsonl"
            
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        data = json.loads(line)
                        if "y_pred" in data and "y_true" in data:
                            # y_trueを取得 (これは全レイヤーで同じ)
                            y_true = np.array(data["y_true"])
                            return y_true, []
    
    raise RuntimeError("Could not find dev labels")


def load_layer_predictions(run_dir: Path, layers: List[int], mode: str = "query") -> Dict[int, np.ndarray]:
    """各レイヤーの予測確率を読み込む"""
    
    print("Extracting predictions from trained models...")
    
    # まず、入力データが必要 - summary.jsonを読んでデータディレクトリを確認
    summary_path = run_dir / "summary.json"
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # 各レイヤーのモデルを読み込んで予測を生成する必要がある
    # しかし、これには元のデータが必要
    # 代わりに、log.jsonlから保存された予測を読み込む
    
    predictions = {}
    
    for layer in layers:
        layer_dir = run_dir / mode / f"layer_{layer}"
        log_file = layer_dir / "log.jsonl"
        
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        # log.jsonlの最後の行（best epoch）の予測確率を取得
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # 最後のエポックを探す
        best_data = None
        for line in reversed(lines):
            data = json.loads(line)
            if "y_prob" in data:
                best_data = data
                break
        
        if best_data is None:
            raise RuntimeError(f"No predictions found in {log_file}")
        
        y_prob = np.array(best_data["y_prob"])
        predictions[layer] = y_prob
        
        print(f"Layer {layer}: loaded predictions shape={y_prob.shape}")
    
    return predictions


def tune_threshold_per_class(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """クラスごとの最適閾値を探索"""
    best_thresholds = {}
    
    for i, label in enumerate(DIRS):
        best_th = 0.5
        best_f1 = 0.0
        
        for th in np.arange(0.05, 0.96, 0.05):
            y_pred = (y_prob[:, i] >= th).astype(int)
            f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = float(th)
        
        best_thresholds[label] = best_th
    
    return best_thresholds


def evaluate_with_thresholds(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """閾値を使って評価"""
    f1_scores = []
    
    for i, label in enumerate(DIRS):
        th = thresholds[label]
        y_pred = (y_prob[:, i] >= th).astype(int)
        f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
        f1_scores.append(f1)
    
    return {
        "macro_f1_posonly": float(np.mean(f1_scores)),
        "per_label_f1": [float(f) for f in f1_scores],
        "threshold_dict": thresholds,
    }


class LearnableLayerFusion(nn.Module):
    """学習可能なレイヤー融合"""
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_weights_raw = nn.Parameter(torch.ones(num_layers))
    
    def forward(self, predictions_list):
        # predictions_list: list of (N, num_labels) tensors
        # Stack: (num_layers, N, num_labels)
        stacked = torch.stack(predictions_list, dim=0)
        
        # Softmax weights
        weights = F.softmax(self.layer_weights_raw, dim=0)  # (num_layers,)
        
        # Weighted sum: (N, num_labels)
        fused = torch.einsum('lnc,l->nc', stacked, weights)
        
        return fused, weights


def optimize_layer_fusion(
    predictions_dict: Dict[int, np.ndarray],
    y_true: np.ndarray,
    layers: List[int],
    epochs: int = 50,
    lr: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """レイヤー重みを最適化"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert to tensors
    pred_tensors = [torch.tensor(predictions_dict[l], dtype=torch.float32).to(device) for l in layers]
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32).to(device)
    
    # Model
    fusion = LearnableLayerFusion(len(layers)).to(device)
    optimizer = torch.optim.Adam(fusion.parameters(), lr=lr)
    
    best_f1 = 0.0
    best_weights = None
    best_fused_probs = None
    
    print("\nOptimizing layer weights...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        fused_probs, weights = fusion(pred_tensors)
        
        # クラスごとの最適閾値でF1を計算
        fused_np = fused_probs.detach().cpu().numpy()
        thresholds = tune_threshold_per_class(y_true, fused_np)
        
        result = evaluate_with_thresholds(y_true, fused_np, thresholds)
        macro_f1 = result["macro_f1_posonly"]
        
        # F1を損失として使用（最大化）
        # 簡易的に、予測とラベルのBCE lossを最小化
        loss = F.binary_cross_entropy(fused_probs, y_true_tensor)
        
        loss.backward()
        optimizer.step()
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_weights = weights.detach().cpu().numpy()
            best_fused_probs = fused_np
        
        if (epoch + 1) % 10 == 0:
            weights_str = ", ".join([f"{w:.3f}" for w in weights.detach().cpu().numpy()])
            print(f"Epoch {epoch+1}/{epochs}: F1={macro_f1:.4f}, Weights=[{weights_str}]")
    
    return best_fused_probs, best_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run", type=str, required=True,
                       help="Source run directory with trained models")
    parser.add_argument("--layers", type=str, default="10,15,20,25",
                       help="Comma-separated layer indices to fuse")
    parser.add_argument("--out_dir", type=str, default="runs/mixed_llm_gemma2b_multilayer",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Optimization epochs for layer weights")
    parser.add_argument("--lr", type=float, default=0.1,
                       help="Learning rate for weight optimization")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Parse layers
    fusion_layers = [int(x.strip()) for x in args.layers.split(",")]
    print(f"=== Multi-Layer Probe Fusion ===")
    print(f"Fusion layers: {fusion_layers}")
    print(f"Source: {args.source_run}\n")
    
    source_dir = Path(args.source_run)
    
    # Load ground truth labels
    print("Loading dev labels...")
    y_true, _ = load_dev_data(source_dir)
    print(f"Dev samples: {y_true.shape[0]}, Labels: {y_true.shape[1]}")
    
    # Load predictions from each layer
    predictions_dict = load_layer_predictions(source_dir, fusion_layers)
    
    # Optimize layer fusion
    fused_probs, optimal_weights = optimize_layer_fusion(
        predictions_dict,
        y_true,
        fusion_layers,
        epochs=args.epochs,
        lr=args.lr,
    )
    
    # Final evaluation with optimal thresholds
    print("\n=== Final Evaluation ===")
    best_thresholds = tune_threshold_per_class(y_true, fused_probs)
    result = evaluate_with_thresholds(y_true, fused_probs, best_thresholds)
    
    # Save results
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    
    layer_weights_dict = {f"layer_{l}": float(w) for l, w in zip(fusion_layers, optimal_weights)}
    
    summary = {
        "model_type": "multilayer_fusion",
        "fusion_layers": fusion_layers,
        "layer_weights": layer_weights_dict,
        "best": result,
    }
    
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nMacro F1: {result['macro_f1_posonly']:.4f}")
    print(f"\nOptimal Layer Weights:")
    for layer_name, weight in layer_weights_dict.items():
        print(f"  {layer_name}: {weight:.4f}")
    
    print(f"\nPer-Class Results:")
    for label, f1, th in zip(DIRS, result['per_label_f1'], [best_thresholds[l] for l in DIRS]):
        print(f"  {label}: F1={f1:.4f}, threshold={th:.2f}")
    
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
