#!/usr/bin/env python3
"""
スタンドアロンのマルチレイヤープローブ学習スクリプト

train_multilayer_probe_from_cache関数を直接呼び出してマルチレイヤー学習を実行します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional
import sys

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

# 共通モジュールをインポート
from common import DIRS, QUERY_LABEL_STR
from sklearn.metrics import f1_score, classification_report


def set_seed(seed: int) -> None:
    """乱数シードを設定"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: Path) -> None:
    """ディレクトリを安全に作成"""
    path.mkdir(parents=True, exist_ok=True)


def load_cached_features(run_dir: Path, layers: List[int], mode: str = "query") -> Dict[int, Dict[str, Any]]:
    """各レイヤーのキャッシュ済み特徴量を読み込む"""
    features = {}
    
    for layer in layers:
        layer_dir = run_dir / mode / f"layer_{layer}"
        
        # 特徴量ファイルを探す
        train_cache = layer_dir / "train_cached.pt"
        dev_cache = layer_dir / "dev_cached.pt"
        
        if not train_cache.exists() or not dev_cache.exists():
            raise FileNotFoundError(f"Cached features not found for layer {layer}")
        
        features[layer] = {
            "train": torch.load(train_cache),
            "dev": torch.load(dev_cache),
        }
    
    return features


class LearnableLayerWeights(nn.Module):
    """学習可能なレイヤー重み"""
    def __init__(self, num_layers: int):
        super().__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
    
    def forward(self):
        return F.softmax(self.layer_weights, dim=0)


class MultiLayerQueryHead(nn.Module):
    """マルチレイヤー対応のQuery Head"""
    def __init__(self, hidden_size: int, num_layers: int, num_labels: int):
        super().__init__()
        self.num_layers = num_layers
        self.layer_fusion = LearnableLayerWeights(num_layers)
        self.classifier = nn.Linear(hidden_size, num_labels)
    
    def forward(self, x):
        # x shape: (batch, num_layers, hidden_size)
        weights = self.layer_fusion()  # (num_layers,)
        
        # 重み付け和
        weighted_x = torch.einsum('bld,l->bd', x, weights)
        
        # 分類
        logits = self.classifier(weighted_x)
        return logits, weights


def train_multilayer_probe_simple(
    X_train_dict: Dict[int, torch.Tensor],
    Y_train: torch.Tensor,
    X_dev_dict: Dict[int, torch.Tensor],
    Y_dev: torch.Tensor,
    layers: List[int],
    hidden_size: int,
    num_labels: int,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-4,
    device: str = "cuda",
) -> Dict[str, Any]:
    """マルチレイヤープローブの学習"""
    
    # データ準備
    X_train_list = [X_train_dict[l] for l in layers]
    X_dev_list = [X_dev_dict[l] for l in layers]
    
    X_train_stacked = torch.stack(X_train_list, dim=1)  # (N, num_layers, hidden)
    X_dev_stacked = torch.stack(X_dev_list, dim=1)
    
    train_ds = TensorDataset(X_train_stacked, Y_train)
    dev_ds = TensorDataset(X_dev_stacked, Y_dev)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    
    # モデル
    model = MultiLayerQueryHead(hidden_size, len(layers), num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_f1 = 0.0
    best_weights = None
    best_thresholds = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_dl:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_dl)
        
        # Evaluation
        model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in dev_dl:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                logits, _ = model(batch_x)
                probs = torch.sigmoid(logits)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
        
        y_true = np.concatenate(all_labels, axis=0)
        y_prob = np.concatenate(all_probs, axis=0)
        
        # Per-class threshold optimization
        best_thresholds_epoch = {}
        f1_scores = []
        
        for i, label in enumerate(DIRS):
            best_th = 0.5
            best_f1_class = 0.0
            
            for th in np.arange(0.05, 0.95, 0.05):
                y_pred = (y_prob[:, i] >= th).astype(int)
                f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
                
                if f1 > best_f1_class:
                    best_f1_class = f1
                    best_th = th
            
            best_thresholds_epoch[label] = best_th
            f1_scores.append(best_f1_class)
        
        macro_f1 = np.mean(f1_scores)
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Macro F1={macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_thresholds = best_thresholds_epoch
            # レイヤー重みを取得
            with torch.no_grad():
                best_weights = model.layer_fusion().cpu().numpy().tolist()
    
    return {
        "macro_f1_posonly": best_f1,
        "per_label_f1": f1_scores,
        "layer_weights": {f"layer_{l}": w for l, w in zip(layers, best_weights)},
        "threshold_dict": best_thresholds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run", type=str, required=True,
                       help="Source run directory with cached features")
    parser.add_argument("--layers", type=str, default="10,15,20,25",
                       help="Comma-separated layer indices to fuse")
    parser.add_argument("--out_dir", type=str, default="runs/mixed_llm_gemma2b_multilayer",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse layers
    fusion_layers = [int(x.strip()) for x in args.layers.split(",")]
    print(f"Fusion layers: {fusion_layers}")
    
    # Load cached features
    print(f"Loading cached features from: {args.source_run}")
    source_dir = Path(args.source_run)
    
    # Load from first layer to get metadata
    first_layer_dir = source_dir / "query" / f"layer_{fusion_layers[0]}"
    train_cache = torch.load(first_layer_dir / "train_cached.pt")
    dev_cache = torch.load(first_layer_dir / "dev_cached.pt")
    
    X_train_dict = {}
    X_dev_dict = {}
    
    for layer in fusion_layers:
        layer_dir = source_dir / "query" / f"layer_{layer}"
        train_data = torch.load(layer_dir / "train_cached.pt")
        dev_data = torch.load(layer_dir / "dev_cached.pt")
        
        X_train_dict[layer] = train_data["X"]
        X_dev_dict[layer] = dev_data["X"]
        
        print(f"Layer {layer}: train={train_data['X'].shape}, dev={dev_data['X'].shape}")
    
    # Use Y from first layer (same for all)
    Y_train = train_cache["Y"]
    Y_dev = dev_cache["Y"]
    
    hidden_size = X_train_dict[fusion_layers[0]].shape[1]
    num_labels = Y_train.shape[1]
    
    print(f"\nHidden size: {hidden_size}, Num labels: {num_labels}")
    print(f"Training samples: {Y_train.shape[0]}, Dev samples: {Y_dev.shape[0]}")
    
    # Train
    print("\n=== Training Multi-Layer Probe ===\n")
    result = train_multilayer_probe_simple(
        X_train_dict=X_train_dict,
        Y_train=Y_train,
        X_dev_dict=X_dev_dict,
        Y_dev=Y_dev,
        layers=fusion_layers,
        hidden_size=hidden_size,
        num_labels=num_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    
    # Save results
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)
    
    summary = {
        "model_type": "multilayer",
        "fusion_layers": fusion_layers,
        "best": result,
    }
    
    summary_path = out_dir / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Macro F1: {result['macro_f1_posonly']:.4f}")
    print(f"\nLayer Weights:")
    for layer_name, weight in result['layer_weights'].items():
        print(f"  {layer_name}: {weight:.4f}")
    
    print(f"\nPer-Class F1:")
    for label, f1 in zip(DIRS, result['per_label_f1']):
        th = result['threshold_dict'][label]
        print(f"  {label}: F1={f1:.4f}, threshold={th:.2f}")
    
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
