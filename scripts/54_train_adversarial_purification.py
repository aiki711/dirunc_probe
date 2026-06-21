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
from sklearn.cluster import KMeans
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

# PyTorch Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Negate gradient and multiply by alpha
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)

# Adversarial purification model
class AdversarialPurifierModel(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=512, bottleneck_dim=128, num_topics=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.classifier = nn.Linear(bottleneck_dim, 1)
        self.topic_classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_topics)
        )
        
    def forward(self, x, alpha=0.5):
        z = self.mlp(x)
        logits_omission = self.classifier(z).squeeze(-1)
        
        # Apply GRL before passing to the topic classifier
        z_reversed = grad_reverse(z, alpha)
        logits_topic = self.topic_classifier(z_reversed)
        
        return z, logits_omission, logits_topic

def train_adversarial_purifier(X_train_tensor, y_omission_tensor, y_topic_tensor, device, num_topics=10, epochs=15, batch_size=256, lr=1e-3, alpha=0.5):
    """
    Trains the purifier using combined BCE (Omission classification) and CrossEntropy (Adversarial Topic Classification).
    """
    dataset = TensorDataset(X_train_tensor, y_omission_tensor, y_topic_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = AdversarialPurifierModel(input_dim=X_train_tensor.shape[1], num_topics=num_topics).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(1, epochs + 1):
        for x_batch, y_om_batch, y_top_batch in dataloader:
            x_batch = x_batch.to(device)
            y_om_batch = y_om_batch.to(device)
            y_top_batch = y_top_batch.to(device)
            
            # Forward pass
            z, logits_om, logits_top = model(x_batch, alpha=alpha)
            
            loss_omission = bce_loss_fn(logits_om, y_om_batch)
            loss_topic = ce_loss_fn(logits_top, y_top_batch)
            
            # Backpropagation (GRL automatically negates topic loss gradient for self.mlp)
            loss = loss_omission + loss_topic
            
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
        
        # Project through trained Adversarial MLP
        if models[d] is not None:
            models[d].eval()
            with torch.no_grad():
                z_f, _, _ = models[d](X_f)
                z_m, _, _ = models[d](X_m)
            X_f_z = z_f.cpu().numpy()
            X_m_z = z_m.cpu().numpy()
        else:
            X_f_z = X_f.cpu().numpy()
            X_m_z = X_m.cpu().numpy()
            
        X = np.concatenate([X_f_z, X_m_z], axis=0)
        
        y_f = np.zeros(X_f.shape[0])
        y_m = dev_y[cal_indices, d]
        y_true = np.concatenate([y_f, y_m], axis=0)
        
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
    parser.add_argument("--epochs", type=int, default=15, help="Epochs to train GRL model")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_topics", type=int, default=10, help="Number of K-Means clusters for pseudo-topics")
    parser.add_argument("--alpha", type=float, default=0.5, help="GRL gradient scaler alpha")
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
        
    # Sample evaluation set
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
    
    train_f_hs = train_cache["f_hs"].float().numpy() # [N_train, 7, D]
    train_m_hs = train_cache["m_hs"].float().numpy() # [N_train, 7, D]
    train_y_labels = train_cache["y"].numpy()        # [N_train, 7]
    N_train = train_f_hs.shape[0]
    
    dev_f_hs = dev_cache["f_hs"].float().numpy()     # [N_dev, 7, D]
    dev_m_hs = dev_cache["m_hs"].float().numpy()     # [N_dev, 7, D]
    dev_y = dev_cache["y"].numpy()                   # [N_dev, 7]
    
    # 1. Fit KMeans clustering to identify 10 pseudo-topics on train f_hs (sufficient contexts)
    print(f"\nClustering sufficient contexts into {args.num_topics} pseudo-topics (K-Means)...")
    # Take mean over the 7 slots to represent the full sentence context
    train_mean_f_hs = train_f_hs.mean(axis=1) # [N_train, D]
    kmeans = KMeans(n_clusters=args.num_topics, random_state=42, n_init="auto")
    train_topic_labels = kmeans.fit_predict(train_mean_f_hs) # [N_train]
    
    # Prepare adversarial topic target labels for train.
    # Topic is context-level, so train_f_hs and train_m_hs of the same context index share the same topic label.
    y_topic_train_np = np.concatenate([train_topic_labels, train_topic_labels], axis=0) # [2 * N_train]
    y_topic_tensor = torch.tensor(y_topic_train_np, dtype=torch.long)
    
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
            
    # 2. Train 7 Adversarial purifiers and L1 Logistic Probes
    print("\nTraining Adversarial purification networks (Approach C)...")
    models = []
    probes = []
    
    for d in range(7):
        slot = DIRS[d]
        print(f"  Processing slot {d} ({slot})...")
        
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X_train_np = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y_om_np = np.concatenate([y_f, y_m], axis=0)
        
        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        y_om_tensor = torch.tensor(y_om_np, dtype=torch.float32)
        
        if len(np.unique(y_om_np)) <= 1:
            print(f"    Single class detected for slot {slot}. Skipping model training.")
            models.append(None)
            clf = DummyZeroClassifier()
            clf.fit(X_train_np, y_om_np)
            probes.append(clf)
            continue
            
        # Fit adversarial GRL model
        adv_model = train_adversarial_purifier(
            X_train_tensor, y_om_tensor, y_topic_tensor, device,
            num_topics=args.num_topics, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, alpha=args.alpha
        )
        models.append(adv_model)
        
        # Extract bottleneck z
        adv_model.eval()
        with torch.no_grad():
            z_train, _, _ = adv_model(X_train_tensor.to(device))
        X_train_z = z_train.cpu().numpy()
        
        # Fit L1 probe on z
        clf = LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=1.0, 
            tol=1e-2, 
            max_iter=500, 
            random_state=42
        )
        clf.fit(X_train_z, y_om_np)
        probes.append(clf)
        
    # 3. Calibrate thresholds
    print("\nCalibrating prediction thresholds on non-eval dev data...")
    thresholds = calibrate_thresholds(probes, models, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
    print(f"Calibrated Thresholds: {thresholds}")
    
    # 4. Predict on eval samples
    print("\nEvaluating Adversarial Probing on evaluation set...")
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
            
            if models[d] is not None:
                models[d].eval()
                with torch.no_grad():
                    z_feat, _, _ = models[d](feat_tensor)
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
    print(" ADVERSARIAL PURIFICATION METRICS (APPROACH C) ")
    print("="*50)
    print(f"Identify Acc (6-class):\t{acc_role*100:.2f}%")
    print(f"Identify F1 (Macro):\t{f1_role*100:.2f}%")
    print(f"Verify Acc (Binary):\t{acc_suff*100:.2f}%")
    print(f"Verify F1 (Omission):\t{f1_suff*100:.2f}%")
    print(f"Weight Sparsity:\t{sparsity*100:.2f}%")
    print("="*50)
    
    # Save results
    results = {
        "num_topics": args.num_topics,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "identify_accuracy": float(acc_role),
        "identify_f1": float(f1_role),
        "sufficiency_accuracy": float(acc_suff),
        "sufficiency_f1": float(f1_suff),
        "sparsity": float(sparsity)
    }
    with (out_dir / "adversarial_results.json").open("w") as f:
        json.dump(results, f, indent=2)
        
    # Baseline & Approach B metrics
    baseline = {
        "identify_accuracy": 0.2867, "identify_f1": 0.2409, "sufficiency_accuracy": 0.7833, "sufficiency_f1": 0.8546, "sparsity": 0.0995
    }
    approach_b = {
        "identify_accuracy": 0.2967, "identify_f1": 0.2512, "sufficiency_accuracy": 0.8100, "sufficiency_f1": 0.8736, "sparsity": 0.0299
    }
    
    # 6. Generate MD table
    md_file = out_dir / "results.md"
    with md_file.open("w") as f:
        f.write("# Approach C: Adversarial Topic Erasure Results\n\n")
        f.write(f"Evaluated on a balanced set of **{len(sampled_items)} samples** using `google/gemma-2-2b-it`.\n\n")
        f.write("| Metric | Baseline (Raw) | Approach B (Contrastive Bottleneck) | Approach C (Adversarial GRL) | Best Method |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        def pick_best(b, ab, ac):
            m = max(b, ab, ac)
            if m == b: return "Baseline"
            if m == ab: return "Approach B"
            return "Approach C"
            
        f.write(f"| **Identify Acc** (6-class) | {baseline['identify_accuracy']*100:.2f}% | {approach_b['identify_accuracy']*100:.2f}% | **{acc_role*100:.2f}%** | {pick_best(baseline['identify_accuracy'], approach_b['identify_accuracy'], acc_role)} |\n")
        f.write(f"| **Identify F1** (Macro) | {baseline['identify_f1']*100:.2f}% | {approach_b['identify_f1']*100:.2f}% | **{f1_role*100:.2f}%** | {pick_best(baseline['identify_f1'], approach_b['identify_f1'], f1_role)} |\n")
        f.write(f"| **Verify / Sufficiency Acc** | {baseline['sufficiency_accuracy']*100:.2f}% | {approach_b['sufficiency_accuracy']*100:.2f}% | **{acc_suff*100:.2f}%** | {pick_best(baseline['sufficiency_accuracy'], approach_b['sufficiency_accuracy'], acc_suff)} |\n")
        f.write(f"| **Verify / Sufficiency F1** | {baseline['sufficiency_f1']*100:.2f}% | {approach_b['sufficiency_f1']*100:.2f}% | **{f1_suff*100:.2f}%** | {pick_best(baseline['sufficiency_f1'], approach_b['sufficiency_f1'], f1_suff)} |\n")
        f.write(f"| **Weight Sparsity** | {baseline['sparsity']*100:.2f}% | {approach_b['sparsity']*100:.2f}% | **{sparsity*100:.2f}%** | - |\n\n")
        f.write("### Discussion\n")
        f.write("Approach C utilizes K-Means to identify semantic context templates as domains, and trains an adversarial network via GRL to actively purge domain/topic signals from the feature space, purifying the structural missingness representation.\n")
        
    # 7. Generate comparison plot
    metrics = ['Identify Acc', 'Identify F1', 'Verify Acc', 'Verify F1']
    b_scores = [baseline['identify_accuracy'], baseline['identify_f1'], baseline['sufficiency_accuracy'], baseline['sufficiency_f1']]
    b_scores_bt = [approach_b['identify_accuracy'], approach_b['identify_f1'], approach_b['sufficiency_accuracy'], approach_b['sufficiency_f1']]
    adv_scores = [acc_role, f1_role, acc_suff, f1_suff]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    rects1 = ax.bar(x - width, b_scores, width, label='Baseline (Raw)', color='#455A64')
    rects2 = ax.bar(x, b_scores_bt, width, label='Approach B (Bottleneck)', color='#E65100')
    rects3 = ax.bar(x + width, adv_scores, width, label='Approach C (Adversarial GRL)', color='#1B5E20')
    
    ax.set_ylabel('Scores')
    ax.set_title('Single-Sentence Probing: Comprehensive Comparison of Mitigation Approaches\n(Gemma-2-2b-it, Layer 16)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = out_dir / "all_approaches_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
