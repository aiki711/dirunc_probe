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

def build_diff_dataset(data, mean=None, std=None):
    f_hs = data["f_hs"]          # [B, D] or [B, 7, D]
    m_hs = data["m_hs"]          # [B, D] or [B, 7, D]
    metadata = data["metadata"]
    
    d_hs = m_hs - f_hs
    if len(d_hs.shape) == 3:
        d_hs_flat = d_hs.flatten(start_dim=1)
    else:
        d_hs_flat = d_hs
        
    xs, ys = [], []
    for i, meta in enumerate(metadata):
        role = meta.get("case_role", "")
        if not role or role not in CASE_ROLES:
            continue
            
        x_vec = d_hs_flat[i, :]
        xs.append(x_vec)
        ys.append(CASE_ROLES.index(role))
        
    if not xs:
        return None, None, None
        
    xs_tensor = torch.stack(xs).float()
    ys_tensor = torch.tensor(ys, dtype=torch.long)
    
    if mean is None or std is None:
        mean = xs_tensor.mean(dim=0, keepdim=True)
        std = xs_tensor.std(dim=0, keepdim=True) + 1e-8
        
    xs_tensor = (xs_tensor - mean) / std
    return TensorDataset(xs_tensor, ys_tensor), mean, std

def train_probe_layer(train_ds, dev_ds, device, lambda_l1, active_threshold=1e-4, epochs=15, batch_size=64, lr=1e-3):
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    
    input_dim = train_ds[0][0].shape[0]
    num_classes = len(CASE_ROLES)
    
    model = LinearProbe(input_dim, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0) # We use explicit L1 regularization instead of standard weight decay
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            logits = model(x_batch)
            loss_ce = criterion(logits, y_batch)
            
            # Explicit L1 norm penalty on weights to drive them to zero
            l1_penalty = model.linear.weight.abs().sum()
            loss = loss_ce + lambda_l1 * l1_penalty
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_batch, y_batch in dev_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(x_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            best_weights = model.linear.weight.detach().cpu().clone()
            
    # Calculate active dimensions on the best model weights
    # threshold active_threshold is used to distinguish active features
    active_mask = best_weights.abs() > active_threshold
    active_dims = active_mask.any(dim=0).sum().item()
    
    return best_acc, active_dims

def parse_prefix(prefix):
    prefix_lower = prefix.lower()
    if "soft" in prefix_lower:
        omission = "soft"
    else:
        omission = "strong"
        
    if "nq_aligned" in prefix_lower:
        runs_dir = "layer_sweep_gemini_nq_aligned"
    elif "nq_unaligned" in prefix_lower:
        runs_dir = "layer_sweep_gemini_nq_unaligned"
    elif "final_token_aligned" in prefix_lower:
        runs_dir = "layer_sweep_gemini_final_token_aligned"
    elif "final_token_unaligned" in prefix_lower:
        runs_dir = "layer_sweep_gemini_final_token_unaligned"
    else:
        runs_dir = "layer_sweep_gemini_final_token_aligned"
        
    return runs_dir, omission

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, required=True, help="e.g. final_token_aligned_soft")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,26")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_l1", type=float, default=5e-4, help="Lasso regularization strength")
    parser.add_argument("--active_threshold", type=float, default=1e-4, help="Weight threshold to be considered active")
    parser.add_argument("--max_active_dims", type=int, default=2000, help="Upper limit for the active dimensions Y-axis in the plot")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_path = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training Sparse Case Role Classifiers using Difference vectors from '{args.prefix}'...")
    print(f"L1 regularization strength (lambda_l1): {args.lambda_l1}")
    print(f"Device: {device}, Layers to sweep: {layers}")
    
    results_list = []
    
    for L in layers:
        train_file = cache_path / f"{args.prefix}_layer{L}_train.pt"
        dev_file = cache_path / f"{args.prefix}_layer{L}_dev.pt"
        
        if not train_file.exists() or not dev_file.exists():
            print(f"Cache files for layer {L} missing. Skipping.")
            continue
            
        print(f"  Processing Layer {L}...", end="")
        train_data = torch.load(train_file, map_location="cpu")
        dev_data = torch.load(dev_file, map_location="cpu")
        
        train_ds, train_mean, train_std = build_diff_dataset(train_data)
        dev_ds, _, _ = build_diff_dataset(dev_data, mean=train_mean, std=train_std)
        
        if train_ds is None or dev_ds is None:
            print(" No valid samples found.")
            continue
            
        acc, active_dims = train_probe_layer(
            train_ds, dev_ds, device, 
            lambda_l1=args.lambda_l1, 
            active_threshold=args.active_threshold,
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr
        )
        print(f" Best Dev Acc: {acc:.4f} | Active Dims: {active_dims}/{train_ds[0][0].shape[0]}")
        results_list.append((L, acc, active_dims))
        
    if not results_list:
        print("No layers were trained successfully.")
        return
        
    # Read existing missingness detection metrics for comparison
    runs_dir, omission = parse_prefix(args.prefix)
    runs_base_path = Path("runs") / runs_dir
    
    missing_f1s = []
    
    print(f"Reading missingness detection logs from '{runs_base_path}' for Omission Type '{omission}'...")
    for L in layers:
        log_file = runs_base_path / f"{omission}_layer_{L}" / "log.jsonl"
        if log_file.exists():
            best_f1 = 0.0
            with log_file.open("r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        score = data.get("pair_accuracy_standard", 0.0) + data.get("macro_f1", 0.0)
                        f1 = data.get("macro_f1", 0.0)
                        if 'best_score' not in locals() or score > best_score:
                            best_score = score
                            best_f1 = f1
            del best_score
            missing_f1s.append((L, best_f1))
            
    # Convert to arrays for plotting
    layers_role, accs_role, active_dims = zip(*results_list)
    total_dims = train_ds[0][0].shape[0]
    
    # Plot double Y-axis chart
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Left axis for scores
    color_role = "#1B5E20" # Green
    ax1.set_xlabel("Layer Index", fontsize=12)
    ax1.set_ylabel("Score (Accuracy / F1)", color="black", fontsize=12)
    p1, = ax1.plot(layers_role, accs_role, marker="o", color=color_role, linewidth=2.5, label="Sparse Classifier Accuracy")
    
    p2 = None
    if missing_f1s:
        layers_f1, f1s = zip(*missing_f1s)
        color_f1 = "#BF360C" # Red-Orange
        p2, = ax1.plot(layers_f1, f1s, marker="s", color=color_f1, linewidth=2.5, linestyle="--", label="Missingness Detection Macro F1")
        
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(0.0, 1.05)
    
    # Right axis for active dimensions count
    ax2 = ax1.twinx()
    color_dim = "#E65100" # Orange
    ax2.set_ylabel(f"Active Probing Dimensions (out of {total_dims})", color=color_dim, fontsize=12)
    p3, = ax2.plot(layers_role, active_dims, marker="x", color=color_dim, linewidth=2.0, linestyle=":", label="Active Dimensions")
    ax2.tick_params(axis="y", labelcolor=color_dim)
    ax2.set_ylim(0, args.max_active_dims)
    
    # Build legend
    plots = [p1]
    if p2 is not None:
        plots.append(p2)
    plots.append(p3)
    labs = [p.get_label() for p in plots]
    ax1.legend(plots, labs, loc="lower left", fontsize=10)
    
    plt.title(f"Sparse Probing on Difference Vectors (L1 regularized)\n(Gemma-2-2b-it, {args.prefix.upper()}, $\\lambda_{{l1}}$={args.lambda_l1})", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plot_path = out_dir / "depth_gradient_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Depth gradient comparison plot saved to {plot_path}")
    
    # Save raw results to JSON
    raw_results = {
        "prefix": args.prefix,
        "lambda_l1": args.lambda_l1,
        "active_threshold": args.active_threshold,
        "layers": layers,
        "sparse_classification_accuracy": {str(L): acc for L, acc in zip(layers_role, accs_role)},
        "active_dimensions": {str(L): dim for L, dim in zip(layers_role, active_dims)},
        "total_dimensions": total_dims,
        "missingness_detection_macro_f1": {str(L): f1 for L, f1 in missing_f1s} if missing_f1s else {}
    }
    with (out_dir / "results.json").open("w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Raw results saved to {out_dir / 'results.json'}")

if __name__ == "__main__":
    main()
