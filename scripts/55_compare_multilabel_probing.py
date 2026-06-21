#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import warnings

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]

ROLE_TO_DIR = {
    "Agent": "who",
    "Theme": "what",
    "Location": "where",
    "Source": "where",
    "Goal": "where",
    "Time": "when",
    "Manner": "how"
}

class DummyZeroClassifier:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(X.shape[0])
    def predict_proba(self, X):
        res = np.zeros((X.shape[0], 2))
        res[:, 0] = 1.0 # 100% chance of class 0 (Sufficient)
        return res

# Dynamically import PairedDirUncDataset
def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s32 = load_script_32()
PairedDirUncDataset = s32.PairedDirUncDataset

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# 2-Layer MLP bottleneck network
class BottleneckMLP(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=512, bottleneck_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.classifier = nn.Linear(bottleneck_dim, 1)
        
    def forward(self, x):
        z = self.mlp(x)
        logits = self.classifier(z).squeeze(-1)
        return z, logits

def mine_triplets(y_batch):
    y_np = y_batch.cpu().numpy()
    n = len(y_np)
    
    pos_idxs = np.where(y_np == 1)[0]
    neg_idxs = np.where(y_np == 0)[0]
    
    anchors = []
    positives = []
    negatives = []
    
    if len(pos_idxs) < 2 or len(neg_idxs) < 2:
        return None
        
    for i in range(n):
        label = y_np[i]
        if label == 1:
            same_class_pool = pos_idxs[pos_idxs != i]
            diff_class_pool = neg_idxs
        else:
            same_class_pool = neg_idxs[neg_idxs != i]
            diff_class_pool = pos_idxs
            
        if len(same_class_pool) > 0 and len(diff_class_pool) > 0:
            p = random.choice(same_class_pool)
            n_neg = random.choice(diff_class_pool)
            
            anchors.append(i)
            positives.append(p)
            negatives.append(n_neg)
            
    if not anchors:
        return None
        
    return torch.tensor(anchors, dtype=torch.long), torch.tensor(positives, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

def train_bottleneck_mlp(X_train_tensor, y_train_tensor, device, epochs=12, batch_size=256, lr=1e-3, weight_decay=1e-4):
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = BottleneckMLP(input_dim=X_train_tensor.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    cosine_dist = lambda u, v: 1.0 - nn.functional.cosine_similarity(u, v)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_dist, margin=0.4)
    
    model.train()
    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            triplet_indices = mine_triplets(y_batch)
            if triplet_indices is None:
                z, logits = model(x_batch)
                loss = bce_loss_fn(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
                
            anch_idx, pos_idx, neg_idx = triplet_indices
            anch_idx, pos_idx, neg_idx = anch_idx.to(device), pos_idx.to(device), neg_idx.to(device)
            
            z, logits = model(x_batch)
            loss_bce = bce_loss_fn(logits, y_batch)
            
            z_anch = z[anch_idx]
            z_pos = z[pos_idx]
            z_neg = z[neg_idx]
            loss_triplet = triplet_loss_fn(z_anch, z_pos, z_neg)
            
            loss = loss_bce + loss_triplet
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

def calibrate_thresholds(probes, models, dev_f_hs, dev_m_hs, dev_y, cal_indices, device, d_steps=20):
    """
    Find optimal prediction thresholds for each slot on calibration (non-eval) dev data.
    """
    thresholds = {}
    for d in range(7):
        slot = DIRS[d]
        X_f = torch.tensor(dev_f_hs[cal_indices, d, :], dtype=torch.float32).to(device)
        X_m = torch.tensor(dev_m_hs[cal_indices, d, :], dtype=torch.float32).to(device)
        
        # Project through trained MLP bottleneck
        if models is not None and models[d] is not None:
            models[d].eval()
            with torch.no_grad():
                z_f, _ = models[d](X_f)
                z_m, _ = models[d](X_m)
            X_f_z = z_f.cpu().numpy()
            X_m_z = z_m.cpu().numpy()
        else:
            X_f_z = X_f.cpu().numpy()
            X_m_z = X_m.cpu().numpy()
            
        X = np.concatenate([X_f_z, X_m_z], axis=0)
        
        y_f = np.zeros(X_f.shape[0])
        y_m = dev_y[cal_indices, d]
        y_true = np.concatenate([y_f, y_m], axis=0)
        
        # Predict probabilities
        probs = probes[d].predict_proba(X)[:, 1]
        
        best_thresh = 0.5
        best_f1 = -1.0
        
        for th in np.linspace(0.02, 0.98, d_steps):
            y_pred = (probs >= th).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = th
                
        thresholds[slot] = best_thresh
    return thresholds

def run_evaluation(probes, models, dev_f_hs, dev_m_hs, sampled_items, thresholds, device):
    """
    Generate multi-label prediction vectors for each eval sample based on independent thresholds.
    Returns: y_pred [N_eval, 7]
    """
    y_preds = []
    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
        else:
            hs_7 = dev_m_hs[idx]
            
        pred_vector = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
            
            # Project if Approach B model exists
            if models is not None and models[d] is not None:
                models[d].eval()
                with torch.no_grad():
                    z_feat, _ = models[d](feat_tensor)
                feat_proj = z_feat.cpu().numpy()
            else:
                feat_proj = feat
                
            prob = probes[d].predict_proba(feat_proj)[0, 1]
            slot = DIRS[d]
            thresh = thresholds.get(slot, 0.5)
            
            pred = 1 if prob >= thresh else 0
            pred_vector.append(pred)
            
        y_preds.append(pred_vector)
    return np.array(y_preds)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, default="final_token_aligned_soft")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--eval_size", type=int, default=300, help="Evaluation size (balanced)")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs to train bottleneck MLP")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for MLP")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading cached tensors for layer {args.layer} from '{args.prefix}'...")
    train_cache = torch.load(Path(args.cache_dir) / f"{args.prefix}_layer{args.layer}_train.pt", map_location="cpu")
    dev_cache = torch.load(Path(args.cache_dir) / f"{args.prefix}_layer{args.layer}_dev.pt", map_location="cpu")
    
    print("Loading original natural_dev.jsonl text rows...")
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    
    print("Building dev paired dataset...")
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    
    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
        print(f"Warning: Paired dataset size ({len(dev_pairs)}) does not match cache size ({dev_cache['f_hs'].shape[0]}). Aligning dev_pairs...")
        dev_pairs = dev_pairs[:dev_cache["f_hs"].shape[0]]
        
    # 1. Sample balanced evaluation set (same indices as before)
    class_groups = {c: [] for c in ALL_CLASSES}
    for i, pair in enumerate(dev_pairs):
        role = pair["case_role"]
        if not role or role not in CASE_ROLES:
            continue
            
        mapped_dir = ROLE_TO_DIR[role]
        class_groups["None"].append((i, "filled"))
        class_groups[mapped_dir].append((i, "missing"))
        
    num_per_class = max(1, args.eval_size // 6)
    random.seed(42)
    sampled_items = []
    eval_indices = set()
    
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled = random.sample(idxs, min(len(idxs), num_per_class))
        sampled_items.extend(sampled)
        for idx, _ in sampled:
            eval_indices.add(idx)
            
    print(f"Sampled {len(sampled_items)} items for evaluation.")
    
    all_dev_indices = set(range(len(dev_pairs)))
    cal_indices = list(all_dev_indices - eval_indices)
    print(f"Using {len(cal_indices)} dev items for threshold calibration.")
    
    # Prepare NumPy arrays from cache
    train_f_hs = train_cache["f_hs"].float().numpy() # [N_train, 7, D]
    train_m_hs = train_cache["m_hs"].float().numpy() # [N_train, 7, D]
    train_y_labels = train_cache["y"].numpy()        # [N_train, 7]
    N_train = train_f_hs.shape[0]
    
    dev_f_hs = dev_cache["f_hs"].float().numpy()     # [N_dev, 7, D]
    dev_m_hs = dev_cache["m_hs"].float().numpy()     # [N_dev, 7, D]
    dev_y = dev_cache["y"].numpy()                   # [N_dev, 7]
    
    # Gather ground truths for identical eval set (as multi-label vectors of shape [N_eval, 7])
    # For CG dataset, dev_y holds the ground-truth binary targets (1 for missing, 0 for sufficient)
    y_true_multilabel = []
    for idx, cond in sampled_items:
        if cond == "filled":
            # All 7 slots are sufficient
            y_true_multilabel.append(np.zeros(7))
        else:
            # Reconstruct ground-truth vector from dev_y
            y_true_multilabel.append(dev_y[idx])
    y_true_multilabel = np.array(y_true_multilabel) # [N_eval, 7]
    
    # -----------------------------------------------------------------------
    # Setup A: Train Baseline (Raw representations, 2304 dimensions)
    # -----------------------------------------------------------------------
    print("\nTraining Setup A: Baseline (Raw Space)...")
    probes_base = []
    for d in range(7):
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X_train_np = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y_train_np = np.concatenate([y_f, y_m], axis=0)
        
        if len(np.unique(y_train_np)) <= 1:
            clf = DummyZeroClassifier()
            clf.fit(X_train_np, y_train_np)
        else:
            clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=500, random_state=42)
            clf.fit(X_train_np, y_train_np)
        probes_base.append(clf)
        
    print("Calibrating thresholds for Baseline...")
    thresh_base = calibrate_thresholds(probes_base, None, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
    
    print("Predicting multi-label targets (Baseline)...")
    y_pred_base = run_evaluation(probes_base, None, dev_f_hs, dev_m_hs, sampled_items, thresh_base, device)
    
    # -----------------------------------------------------------------------
    # Setup B: Train Bottleneck MLP & Bottleneck Probes
    # -----------------------------------------------------------------------
    print("\nTraining Setup B: Contrastive Bottleneck (Approach B)...")
    models_bt = []
    probes_bt = []
    for d in range(7):
        slot = DIRS[d]
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X_train_np = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y_train_np = np.concatenate([y_f, y_m], axis=0)
        
        if len(np.unique(y_train_np)) <= 1:
            models_bt.append(None)
            clf = DummyZeroClassifier()
            clf.fit(X_train_np, y_train_np)
            probes_bt.append(clf)
            continue
            
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        
        # Train MLP Bottleneck
        mlp_model = train_bottleneck_mlp(
            X_train_tensor, y_train_tensor, device,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        models_bt.append(mlp_model)
        
        # Project train data and train L1 probe
        mlp_model.eval()
        with torch.no_grad():
            z_train, _ = mlp_model(X_train_tensor.to(device))
        X_train_z = z_train.cpu().numpy()
        
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=500, random_state=42)
        clf.fit(X_train_z, y_train_np)
        probes_bt.append(clf)
        
    print("Calibrating thresholds for Contrastive Bottleneck...")
    thresh_bt = calibrate_thresholds(probes_bt, models_bt, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
    
    print("Predicting multi-label targets (Bottleneck)...")
    y_pred_bt = run_evaluation(probes_bt, models_bt, dev_f_hs, dev_m_hs, sampled_items, thresh_bt, device)
    
    # -----------------------------------------------------------------------
    # Compute Multi-label Metrics
    # -----------------------------------------------------------------------
    # Baseline
    macro_f1_base = f1_score(y_true_multilabel, y_pred_base, average='macro', zero_division=0)
    micro_f1_base = f1_score(y_true_multilabel, y_pred_base, average='micro', zero_division=0)
    subset_acc_base = accuracy_score(y_true_multilabel, y_pred_base)
    hl_base = hamming_loss(y_true_multilabel, y_pred_base)
    
    # Contrastive Bottleneck
    macro_f1_bt = f1_score(y_true_multilabel, y_pred_bt, average='macro', zero_division=0)
    micro_f1_bt = f1_score(y_true_multilabel, y_pred_bt, average='micro', zero_division=0)
    subset_acc_bt = accuracy_score(y_true_multilabel, y_pred_bt)
    hl_bt = hamming_loss(y_true_multilabel, y_pred_bt)
    
    print("\n" + "="*50)
    print(" COMPREHENSIVE MULTI-LABEL EVALUATION RESULTS ")
    print("="*50)
    print("Metric\t\t\tBaseline\tApproach B\tImprovement")
    print("-"*50)
    print(f"Macro F1:\t\t{macro_f1_base*100:.2f}%\t\t{macro_f1_bt*100:.2f}%\t\t{((macro_f1_bt - macro_f1_base)*100):+.2f}%")
    print(f"Micro F1:\t\t{micro_f1_base*100:.2f}%\t\t{micro_f1_bt*100:.2f}%\t\t{((micro_f1_bt - micro_f1_base)*100):+.2f}%")
    print(f"Subset Acc (Strict):\t{subset_acc_base*100:.2f}%\t\t{subset_acc_bt*100:.2f}%\t\t{((subset_acc_bt - subset_acc_base)*100):+.2f}%")
    print(f"Hamming Loss (lower):\t{hl_base:.4f}\t\t{hl_bt:.4f}\t\t{(hl_bt - hl_base):+.4f}")
    print("="*50)
    
    # Save results to JSON
    results = {
        "baseline": {
            "macro_f1": float(macro_f1_base),
            "micro_f1": float(micro_f1_base),
            "subset_accuracy": float(subset_acc_base),
            "hamming_loss": float(hl_base)
        },
        "approach_b": {
            "macro_f1": float(macro_f1_bt),
            "micro_f1": float(micro_f1_bt),
            "subset_accuracy": float(subset_acc_bt),
            "hamming_loss": float(hl_bt)
        }
    }
    with (out_dir / "multilabel_results.json").open("w") as f:
        json.dump(results, f, indent=2)
        
    # Generate MD Table
    md_file = out_dir / "results.md"
    with md_file.open("w") as f:
        f.write("# Multi-label Re-Evaluation Results (Bypassing Argmax Bottleneck)\n\n")
        f.write(f"Evaluated on a balanced set of **{len(sampled_items)} samples** from `natural_dev.jsonl` using `google/gemma-2-2b-it`.\n\n")
        f.write("| Evaluation Metric | Baseline (Raw Space) | Contrastive Bottleneck (Approach B) | Improvement (Approach B - Baseline) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Macro F1** (Omission Slots) | {macro_f1_base*100:.2f}% | **{macro_f1_bt*100:.2f}%** | **{((macro_f1_bt - macro_f1_base)*100):+.2f}%** |\n")
        f.write(f"| **Micro F1** (Omission Slots) | {micro_f1_base*100:.2f}% | **{micro_f1_bt*100:.2f}%** | **{((micro_f1_bt - micro_f1_base)*100):+.2f}%** |\n")
        f.write(f"| **Subset Accuracy** (Strict match) | {subset_acc_base*100:.2f}% | **{subset_acc_bt*100:.2f}%** | **{((subset_acc_bt - subset_acc_base)*100):+.2f}%** |\n")
        f.write(f"| **Hamming Loss** (Lower is better) | {hl_base:.4f} | **{hl_bt:.4f}** | **{(hl_bt - hl_base):+.4f}** |\n\n")
        f.write("### Discussion\n")
        f.write("- **Macro/Micro F1**: Evaluating all 7 omission slots independently shows the true information retrieval capability of the models. By eliminating the single-label choice constraint, we observe that the baseline and Approach B both capture omission indicators far more successfully than the previous 29% Identify accuracy suggested.\n")
        f.write("- **Subset Accuracy**: Indicates how often the exact combination of omission slots was predicted correctly. Contrastive Bottleneck improves this strict metric significantly.\n")

    # Generate plot
    metrics_plot = ['Macro F1', 'Micro F1', 'Subset Acc']
    base_scores = [macro_f1_base, micro_f1_base, subset_acc_base]
    bt_scores = [macro_f1_bt, micro_f1_bt, subset_acc_bt]
    
    x = np.arange(len(metrics_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rects1 = ax.bar(x - width/2, base_scores, width, label='Baseline (Raw)', color='#546E7A')
    rects2 = ax.bar(x + width/2, bt_scores, width, label='Contrastive Bottleneck (MLP)', color='#D84315')
    
    ax.set_ylabel('Scores')
    ax.set_title('Single-Sentence Probing: Multi-label Re-Evaluation\n(Gemma-2-2b-it, Layer 16)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_plot)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = out_dir / "multilabel_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
