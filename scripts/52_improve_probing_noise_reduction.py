#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS

class DummyZeroClassifier:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(X.shape[0])
    def predict_proba(self, X):
        res = np.zeros((X.shape[0], 2))
        res[:, 0] = 1.0 # 100% chance of class 0 (Sufficient)
        return res

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

def project_out(X, pca, K):
    """
    Subtract the projection onto the top K principal components of the PCA from X.
    X: [N, D] (NumPy array)
    pca: fitted sklearn.decomposition.PCA object
    K: int (number of top components to remove)
    """
    if K == 0:
        return X
    components = pca.components_[:K] # [K, D]
    # Projection = X @ components.T @ components
    proj = np.dot(np.dot(X, components.T), components)
    return X - proj

def calibrate_thresholds(probes, pcas, dev_f_hs, dev_m_hs, dev_y, cal_indices, K, d_steps=20):
    """
    Find optimal prediction thresholds for each slot on calibration (non-eval) dev data.
    """
    thresholds = {}
    for d in range(7):
        slot = DIRS[d]
        X_f = dev_f_hs[cal_indices, d, :]
        X_m = dev_m_hs[cal_indices, d, :]
        
        # Project out PCA components
        X_f_proj = project_out(X_f, pcas[d], K)
        X_m_proj = project_out(X_m, pcas[d], K)
        
        X = np.concatenate([X_f_proj, X_m_proj], axis=0)
        
        y_f = np.zeros(X_f.shape[0])
        y_m = dev_y[cal_indices, d]
        y_true = np.concatenate([y_f, y_m], axis=0)
        
        # Predict probabilities
        probs = probes[d].predict_proba(X)[:, 1]
        
        # Grid search threshold to maximize F1 (Insufficient / positive class)
        best_thresh = 0.5
        best_f1 = -1.0
        
        # We sweep from 0.02 to 0.98
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
    parser.add_argument("--k_sweep", type=str, default="0,1,2,5,10,20,50,100,200,300,500", help="Comma-separated K values to sweep")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    k_sweep = [int(k.strip()) for k in args.k_sweep.split(",")]
    max_k = max(k_sweep)
    
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
    
    # 1. Sample balanced evaluation set (identical to the head-to-head baseline)
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
    
    # Calibration indices are all dev indices except the ones sampled for evaluation
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
    
    # 2. Fit PCA on sufficient representations (train_f_hs) for each of the 7 slots
    print(f"Fitting PCA (max K={max_k}) on sufficient representations to capture context noise...")
    pcas = []
    for d in range(7):
        print(f"  Fitting PCA for slot {d} ({DIRS[d]})...")
        X_pca_input = train_f_hs[:, d, :] # [N_train, D]
        pca = PCA(n_components=max_k, random_state=42)
        pca.fit(X_pca_input)
        pcas.append(pca)
        
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
            
    # Sweep over K
    sweep_results = []
    
    for K in k_sweep:
        print(f"\nEvaluating K = {K} (removing top {K} principal components)...")
        
        # 3. Train 7 independent binary probes on projected train representations
        probes = []
        for d in range(7):
            X_f = train_f_hs[:, d, :]
            X_m = train_m_hs[:, d, :]
            
            # Project out the top K components
            X_f_proj = project_out(X_f, pcas[d], K)
            X_m_proj = project_out(X_m, pcas[d], K)
            
            X_train = np.concatenate([X_f_proj, X_m_proj], axis=0)
            
            y_f = np.zeros(N_train)
            y_m = train_y_labels[:, d]
            y_train = np.concatenate([y_f, y_m], axis=0)
            
            # Train L1-regularized Logistic Regression
            # C=1.0 is standard. Set tol=1e-2 to allow extremely fast convergence.
            if len(np.unique(y_train)) <= 1:
                clf = DummyZeroClassifier()
                clf.fit(X_train, y_train)
            else:
                import warnings
                warnings.simplefilter('ignore')
                clf = LogisticRegression(
                    penalty='l1', 
                    solver='liblinear', 
                    C=1.0, 
                    tol=1e-2, 
                    max_iter=500, 
                    random_state=42
                )
                clf.fit(X_train, y_train)
            probes.append(clf)
            
        # 4. Calibrate prediction thresholds using non-eval dev data
        thresholds = calibrate_thresholds(probes, pcas, dev_f_hs, dev_m_hs, dev_y, cal_indices, K)
        print(f"  Calibrated Thresholds: {thresholds}")
        
        # 5. Predict on identical eval samples
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
                # Project out components for test instance
                feat_proj = project_out(feat, pcas[d], K)
                
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
                # Find valid slots that exceeded threshold
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
                    
        # 6. Calculate accuracy metrics
        acc_role = accuracy_score(y_true_role, pred_roles)
        f1_role = f1_score(y_true_role, pred_roles, average="macro")
        acc_suff = accuracy_score(y_true_suff, pred_suff)
        f1_suff = f1_score(y_true_suff, pred_suff, pos_label="Insufficient")
        
        # Calculate weight sparsity: proportion of zero weights across the 7 probes
        total_weights = 0
        zero_weights = 0
        for clf in probes:
            if hasattr(clf, 'coef_'):
                total_weights += clf.coef_.size
                zero_weights += np.sum(np.abs(clf.coef_) < 1e-5)
        sparsity = zero_weights / max(1, total_weights)
        
        print(f"  Identify Acc (6-class): {acc_role*100:.2f}% | Macro F1: {f1_role*100:.2f}%")
        print(f"  Verify Acc (Binary):    {acc_suff*100:.2f}% | Omission F1: {f1_suff*100:.2f}%")
        print(f"  Weight Sparsity:        {sparsity*100:.2f}%")
        
        sweep_results.append({
            "K": K,
            "identify_accuracy": float(acc_role),
            "identify_f1": float(f1_role),
            "sufficiency_accuracy": float(acc_suff),
            "sufficiency_f1": float(f1_suff),
            "sparsity": float(sparsity)
        })
        
    # Save sweep results to JSON
    results_file = out_dir / "pca_sweep_results.json"
    with results_file.open("w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to {results_file}")
    
    # Generate MD Table
    md_file = out_dir / "results.md"
    with md_file.open("w") as f:
        f.write("# PCA Context Noise Projection: Sweep Results\n\n")
        f.write("Swept over different number of top principal components ($K$) to subtract from single-sentence representations.\n")
        f.write("Classifiers are L1-regularized Logistic Regression. Thresholds calibrated on non-eval dev data.\n\n")
        f.write("| $K$ (Components Removed) | Identify Acc (6-class) | Identify F1 (Macro) | Verify Acc (Binary) | Verify F1 (Omission) | Weight Sparsity |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for res in sweep_results:
            f.write(f"| {res['K']} | {res['identify_accuracy']*100:.2f}% | {res['identify_f1']*100:.2f}% | {res['sufficiency_accuracy']*100:.2f}% | {res['sufficiency_f1']*100:.2f}% | {res['sparsity']*100:.2f}% |\n")
            
    # Generate Plot
    Ks = [res["K"] for res in sweep_results]
    id_accs = [res["identify_accuracy"] for res in sweep_results]
    id_f1s = [res["identify_f1"] for res in sweep_results]
    suff_accs = [res["sufficiency_accuracy"] for res in sweep_results]
    sparsities = [res["sparsity"] for res in sweep_results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
    
    color_id = "#1565C0" # Blue
    color_suff = "#004D40" # Dark Teal
    
    ax1.set_xlabel("Number of Top PCA Components Removed (K)", fontsize=12)
    ax1.set_ylabel("Classification Accuracy / F1", color="black", fontsize=12)
    p1, = ax1.plot(Ks, id_accs, marker="o", color=color_id, linewidth=2.5, label="Identify Acc (6-class)")
    p2, = ax1.plot(Ks, id_f1s, marker="o", linestyle="--", color=color_id, linewidth=2.0, label="Identify F1 (Macro)")
    p3, = ax1.plot(Ks, suff_accs, marker="s", color=color_suff, linewidth=2.5, label="Verify Acc (Binary)")
    
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(0.0, 1.05)
    
    # Second y-axis for Sparsity
    ax2 = ax1.twinx()
    color_sp = "#E65100" # Orange
    ax2.set_ylabel("Classifier Weight Sparsity (Fraction of Zeros)", color=color_sp, fontsize=12)
    p4, = ax2.plot(Ks, sparsities, marker="x", color=color_sp, linewidth=2.0, linestyle=":", label="Weight Sparsity")
    ax2.tick_params(axis="y", labelcolor=color_sp)
    ax2.set_ylim(0.0, 1.05)
    
    # Legend
    plots = [p1, p2, p3, p4]
    labs = [p.get_label() for p in plots]
    ax1.legend(plots, labs, loc="lower right", fontsize=10)
    
    plt.title(f"Impact of PCA Context Noise Reduction on Probing\n(Gemma-2-2b-it, Layer 16, {args.prefix.upper()})", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plot_path = out_dir / "pca_sweep_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
