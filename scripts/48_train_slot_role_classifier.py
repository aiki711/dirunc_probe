#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]

class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def build_probe_dataset(data):
    f_hs = data["f_hs"]          # Should be [N, D]
    metadata = data["metadata"]
    
    xs, ys = [], []
    for i, meta in enumerate(metadata):
        role = meta.get("case_role", "")
        if not role or role not in CASE_ROLES:
            continue
            
        x_vec = f_hs[i, :]
        xs.append(x_vec)
        ys.append(CASE_ROLES.index(role))
        
    if not xs:
        return None
        
    xs_tensor = torch.stack(xs)
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(xs_tensor, ys_tensor)

def train_probe_layer(train_ds, dev_ds, device, epochs=15, batch_size=64, lr=1e-3):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    
    input_dim = train_ds[0][0].shape[0]
    num_classes = len(CASE_ROLES)
    
    model = LinearProbe(input_dim, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device).to(torch.float32)
            y_batch = y_batch.to(device)
            
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in dev_dl:
                x_batch = x_batch.to(device).to(torch.float32)
                y_batch = y_batch.to(device)
                
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            
    return best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, default="slot_aligned_soft")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,26")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_path = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Case Role Classifiers using Slot-aligned states '{args.prefix}'...")
    print(f"Device: {device}, Layers to sweep: {layers}")
    
    role_accs = []
    
    for L in layers:
        train_file = cache_path / f"{args.prefix}_layer{L}_train.pt"
        dev_file = cache_path / f"{args.prefix}_layer{L}_dev.pt"
        
        if not train_file.exists() or not dev_file.exists():
            print(f"Cache files for layer {L} missing. Skipping.")
            continue
            
        print(f"  Processing Layer {L}...", end="")
        # Load tensors using cpu mapping to avoid GPU loading overhead
        train_data = torch.load(train_file, map_location="cpu")
        dev_data = torch.load(dev_file, map_location="cpu")
        
        train_ds = build_probe_dataset(train_data)
        dev_ds = build_probe_dataset(dev_data)
        
        if train_ds is None or dev_ds is None:
            print(" No valid samples found.")
            continue
            
        acc = train_probe_layer(train_ds, dev_ds, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print(f" Best Dev Acc: {acc:.4f}")
        role_accs.append((L, acc))
        
    if not role_accs:
        print("No layers were trained successfully.")
        return
        
    # Read existing missingness detection metrics for comparison (from final_token_aligned_soft log)
    runs_base_path = Path("runs/layer_sweep_gemini_final_token_aligned")
    
    missing_f1s = []
    missing_pair_accs = []
    
    print(f"Reading missingness detection logs from '{runs_base_path}'...")
    for L in layers:
        log_file = runs_base_path / f"soft_layer_{L}" / "log.jsonl"
        if log_file.exists():
            best_f1 = 0.0
            best_pair_acc = 0.0
            with log_file.open("r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        score = data.get("pair_accuracy_standard", 0.0) + data.get("macro_f1", 0.0)
                        f1 = data.get("macro_f1", 0.0)
                        pair_acc = data.get("pair_accuracy_standard", 0.0)
                        if 'best_score' not in locals() or score > best_score:
                            best_score = score
                            best_f1 = f1
                            best_pair_acc = pair_acc
            del best_score
            missing_f1s.append((L, best_f1))
            missing_pair_accs.append((L, best_pair_acc))
        else:
            print(f"  Log file not found: {log_file}")
            
    # Convert to arrays for plotting
    layers_role, accs_role = zip(*role_accs)
    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(layers_role, accs_role, marker='o', color="#1B5E20", linewidth=2.5, label="Slot Role Classification Accuracy (Filled)")
    
    if missing_f1s:
        layers_f1, f1s = zip(*missing_f1s)
        plt.plot(layers_f1, f1s, marker='s', color="#BF360C", linewidth=2.5, linestyle="--", label="Missingness Detection Macro F1 (Final Token)")
        
    if missing_pair_accs:
        layers_pacc, paccs = zip(*missing_pair_accs)
        plt.plot(layers_pacc, paccs, marker='^', color="#1565C0", linewidth=2.5, linestyle=":", label="Missingness Standard Pair Accuracy")
        
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(f"True Depth Gradient: Slot Classification vs Missingness Detection\n(Gemma-2-2b-it, Slot-aligned vs Final Token)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xticks(layers)
    plt.ylim(0.0, 1.05)
    plt.legend(fontsize=10, loc="lower left")
    
    plot_path = out_dir / "depth_gradient_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Depth gradient comparison plot saved to {plot_path}")
    
    # Save raw results to JSON
    raw_results = {
        "prefix": args.prefix,
        "layers": layers,
        "slot_classification_accuracy": {str(L): acc for L, acc in role_accs},
        "missingness_detection_macro_f1": {str(L): f1 for L, f1 in missing_f1s} if missing_f1s else {},
        "missingness_detection_pair_accuracy": {str(L): pacc for L, pacc in missing_pair_accs} if missing_pair_accs else {}
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Raw results saved to {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()
