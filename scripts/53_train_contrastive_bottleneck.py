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
from sklearn.metrics import accuracy_score, f1_score
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
    """
    Generate indices for (Anchor, Positive, Negative) triplets from the batch labels.
    """
    y_np = y_batch.cpu().numpy()
    n = len(y_np)
    
    pos_idxs = np.where(y_np == 1)[0]
    neg_idxs = np.where(y_np == 0)[0]
    
    anchors = []
    positives = []
    negatives = []
    
    if len(pos_idxs) < 2 or len(neg_idxs) < 2:
        # Fallback if batch doesn't contain enough positive or negative samples
        return None
        
    for i in range(n):
        label = y_np[i]
        if label == 1:
            # Anchor is positive. Positive must be another positive. Negative must be negative.
            same_class_pool = pos_idxs[pos_idxs != i]
            diff_class_pool = neg_idxs
        else:
            # Anchor is negative. Positive must be another negative. Negative must be positive.
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
    """
    Trains the BottleneckMLP model using dynamic Triplet Margin Loss and BCE Classification Loss.
    """
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = BottleneckMLP(input_dim=X_train_tensor.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # BCE classification loss
    bce_loss_fn = nn.BCEWithLogitsLoss()
    
    # Cosine distance function for triplet loss
    cosine_dist = lambda u, v: 1.0 - nn.functional.cosine_similarity(u, v)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_dist, margin=0.4)
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_bce = 0.0
        total_triplet = 0.0
        steps = 0
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Mine batch-level triplets
            triplet_indices = mine_triplets(y_batch)
            if triplet_indices is None:
                # Fallback to pure classification loss if batch cannot form triplets
                z, logits = model(x_batch)
                loss = bce_loss_fn(logits, y_batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
                
            anch_idx, pos_idx, neg_idx = triplet_indices
            anch_idx, pos_idx, neg_idx = anch_idx.to(device), pos_idx.to(device), neg_idx.to(device)
            
            # Forward pass
            z, logits = model(x_batch)
            
            # BCE Loss on the entire batch
            loss_bce = bce_loss_fn(logits, y_batch)
            
            # Triplet Loss on mined anchors, positives and negatives
            z_anch = z[anch_idx]
            z_pos = z[pos_idx]
            z_neg = z[neg_idx]
            loss_triplet = triplet_loss_fn(z_anch, z_pos, z_neg)
            
            # Combined Loss (equal weighting)
            loss = loss_bce + loss_triplet
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_triplet += loss_triplet.item()
            steps += 1
            
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
        if models[d] is not None:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, default="final_token_aligned_soft")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--eval_size", type=int, default=300, help="Evaluation size (balanced)")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs to train bottleneck MLP")
    parser.add_argument("--batch_size", type=int, default=256, help="Dataloader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for MLP")
    parser.add_argument("--bottleneck_dim", type=int, default=128, help="Dimensionality of bottleneck feature z")
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
        
    # 1. Sample balanced evaluation set (identical to baseline and PCA sweep)
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
    
    # Gather ground truths for identical eval set
    y_true_role = []
    y_true_suff = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        if cond == "filled":
            y_true_role.append("None")
            y_true_suff.append("Sufficient")
        else:
            y_true_role.append(ROLE_TO_DIR[pair["case_role"]])
            y_true_suff.append("Insufficient")
            
    # 2. Train 7 MLP bottleneck networks and fit L1-regularized Logistic Probes
    print("\nTraining Contrastive Bottleneck Networks (Approach B) on CUDA/CPU...")
    
    models = []
    probes = []
    
    for d in range(7):
        slot = DIRS[d]
        print(f"  Processing slot {d} ({slot})...")
        
        # Build training tensors
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X_train_np = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y_train_np = np.concatenate([y_f, y_m], axis=0)
        
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
        
        # Check single class fallback
        if len(np.unique(y_train_np)) <= 1:
            print(f"    Single class detected for slot {slot}. Skipping MLP training.")
            models.append(None)
            clf = DummyZeroClassifier()
            clf.fit(X_train_np, y_train_np)
            probes.append(clf)
            continue
            
        # Fit non-linear MLP projection
        mlp_model = train_bottleneck_mlp(
            X_train_tensor, y_train_tensor, device, 
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
        )
        models.append(mlp_model)
        
        # Extract bottleneck representations z
        mlp_model.eval()
        with torch.no_grad():
            z_train, _ = mlp_model(X_train_tensor.to(device))
        X_train_z = z_train.cpu().numpy()
        
        # Train L1 probe on the 128-dimensional bottleneck space z
        clf = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=1.0, 
            tol=1e-2, 
            max_iter=500, 
            random_state=42
        )
        clf.fit(X_train_z, y_train_np)
        probes.append(clf)
        
    # 3. Calibrate prediction thresholds using non-eval dev data
    print("\nCalibrating prediction thresholds on non-eval dev data...")
    thresholds = calibrate_thresholds(probes, models, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
    print(f"Calibrated Thresholds: {thresholds}")
    
    # 4. Predict on identical eval samples
    print("\nEvaluating Contrastive Bottleneck Probing on evaluation set...")
    pred_roles = []
    pred_suff = []
    
    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
        else:
            hs_7 = dev_m_hs[idx]
            
        scores = []
        preds = []
        
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
            
            # Project through bottleneck MLP
            if models[d] is not None:
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
            score = prob - thresh
            
            preds.append(pred)
            scores.append(score)
            
        if all(p == 0 for p in preds):
            pred_roles.append("None")
            pred_suff.append("Sufficient")
        else:
            valid_slots = [(scores[d], d) for d in range(7) if preds[d] == 1]
            if valid_slots:
                _, d_best = max(valid_slots, key=lambda x: x[0])
                pred_slot = DIRS[d_best]
            else:
                d_best = np.argmax(scores)
                pred_slot = DIRS[d_best]
                
            if pred_slot in ["why", "which"]:
                pred_roles.append("None")
                pred_suff.append("Sufficient")
            else:
                pred_roles.append(pred_slot)
                pred_suff.append("Insufficient")
                
    # 5. Compute metrics
    acc_role = accuracy_score(y_true_role, pred_roles)
    f1_role = f1_score(y_true_role, pred_roles, average="macro")
    acc_suff = accuracy_score(y_true_suff, pred_suff)
    f1_suff = f1_score(y_true_suff, pred_suff, pos_label="Insufficient")
    
    total_weights = 0
    zero_weights = 0
    for clf in probes:
        if hasattr(clf, 'coef_'):
            total_weights += clf.coef_.size
            zero_weights += np.sum(np.abs(clf.coef_) < 1e-5)
    sparsity = zero_weights / max(1, total_weights)
    
    print("\n" + "="*50)
    print(" CONTRASTIVE BOTTLENECK PROBING METRICS ")
    print("="*50)
    print(f"Identify Acc (6-class):\t{acc_role*100:.2f}%")
    print(f"Identify F1 (Macro):\t{f1_role*100:.2f}%")
    print(f"Verify Acc (Binary):\t{acc_suff*100:.2f}%")
    print(f"Verify F1 (Omission):\t{f1_suff*100:.2f}%")
    print(f"Weight Sparsity:\t{sparsity*100:.2f}%")
    print("="*50)
    
    # Save results to JSON
    results = {
        "bottleneck_dim": args.bottleneck_dim,
        "epochs": args.epochs,
        "identify_accuracy": float(acc_role),
        "identify_f1": float(f1_role),
        "sufficiency_accuracy": float(acc_suff),
        "sufficiency_f1": float(f1_suff),
        "sparsity": float(sparsity)
    }
    
    with (out_dir / "bottleneck_results.json").open("w") as f:
        json.dump(results, f, indent=2)
        
    # Baseline raw metrics for direct comparison:
    # Baseline: Identify 28.67%, F1 24.09%, Verify 78.33%, F1 85.46%
    baseline = {
        "identify_accuracy": 0.2867,
        "identify_f1": 0.2409,
        "sufficiency_accuracy": 0.7833,
        "sufficiency_f1": 0.8546,
        "sparsity": 0.0995
    }
    
    # 6. Generate MD Table
    md_file = out_dir / "results.md"
    with md_file.open("w") as f:
        f.write("# Approach B: Contrastive Bottleneck Probing Results\n\n")
        f.write(f"Evaluated on a balanced set of **{len(sampled_items)} samples** from `natural_dev.jsonl` using `google/gemma-2-2b-it`.\n\n")
        f.write("| Metric | Baseline (Raw Single-Sentence) | Contrastive Bottleneck MLP (Approach B) | Improvement |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Identify Accuracy** (6-class) | {baseline['identify_accuracy']*100:.2f}% | **{acc_role*100:.2f}%** | {((acc_role - baseline['identify_accuracy'])*100):+.2f}% |\n")
        f.write(f"| **Identify F1** (Macro) | {baseline['identify_f1']*100:.2f}% | **{f1_role*100:.2f}%** | {((f1_role - baseline['identify_f1'])*100):+.2f}% |\n")
        f.write(f"| **Verify / Sufficiency Acc** | {baseline['sufficiency_accuracy']*100:.2f}% | **{acc_suff*100:.2f}%** | {((acc_suff - baseline['sufficiency_accuracy'])*100):+.2f}% |\n")
        f.write(f"| **Verify / Sufficiency F1** | {baseline['sufficiency_f1']*100:.2f}% | **{f1_suff*100:.2f}%** | {((f1_suff - baseline['sufficiency_f1'])*100):+.2f}% |\n")
        f.write(f"| **Weight Sparsity** | {baseline['sparsity']*100:.2f}% | **{sparsity*100:.2f}%** | {((sparsity - baseline['sparsity'])*100):+.2f}% |\n\n")
        f.write("### Discussion\n")
        f.write("Approach B maps the 2304-dimensional hidden representations into a 128-dimensional space using a non-linear MLP trained via classification loss and cosine triplet loss to force same-omission classes to cluster together.\n")

    # 7. Generate comparison plot
    metrics = ['Identify Acc', 'Identify F1', 'Verify Acc', 'Verify F1']
    base_scores = [baseline['identify_accuracy'], baseline['identify_f1'], baseline['sufficiency_accuracy'], baseline['sufficiency_f1']]
    bt_scores = [acc_role, f1_role, acc_suff, f1_suff]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rects1 = ax.bar(x - width/2, base_scores, width, label='Baseline (Raw)', color='#455A64')
    rects2 = ax.bar(x + width/2, bt_scores, width, label='Contrastive Bottleneck (MLP)', color='#E65100')
    
    ax.set_ylabel('Scores')
    ax.set_title('Single-Sentence Probing: Baseline vs. Contrastive Bottleneck (Approach B)\n(Gemma-2-2b-it, Layer 16)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = out_dir / "bottleneck_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
