#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
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
        res[:, 0] = 1.0
        return res

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

def calibrate_thresholds(probs, y_true, d_steps=20):
    best_thresh = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.02, 0.98, d_steps):
        y_pred = (probs >= th).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = th
    return best_thresh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, default="final_token_aligned_soft")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--eval_size", type=int, default=300, help="Evaluation size (balanced)")
    parser.add_argument("--C", type=float, default=1.0, help="C for raw logistic regression")
    parser.add_argument("--C_fusion", type=float, default=0.1, help="C for fusion logistic regression")
    parser.add_argument("--sub_size", type=int, default=1500, help="Subset size for training data to speed up")
    args = parser.parse_args()
    
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
    
    # Read training data
    train_f_hs_full = train_cache["f_hs"].float().numpy() # [N_train, 7, D]
    train_m_hs_full = train_cache["m_hs"].float().numpy() # [N_train, 7, D]
    train_y_labels_full = train_cache["y"].numpy()        # [N_train, 7]
    N_train_full = train_f_hs_full.shape[0]
    
    # Subsample training data subset globally to speed up all steps
    np.random.seed(42)
    sub_indices = np.random.choice(N_train_full, size=min(args.sub_size, N_train_full), replace=False)
    print(f"Subsampled training data: {len(sub_indices)} pairs (out of {N_train_full}) for baseline & fusion training.")
    
    train_f_hs = train_f_hs_full[sub_indices]
    train_m_hs = train_m_hs_full[sub_indices]
    train_y_labels = train_y_labels_full[sub_indices]
    N_train = len(sub_indices)
    
    dev_f_hs = dev_cache["f_hs"].float().numpy()     # [N_dev, 7, D]
    dev_m_hs = dev_cache["m_hs"].float().numpy()     # [N_dev, 7, D]
    dev_y = dev_cache["y"].numpy()                   # [N_dev, 7]
    
    # Reconstruct ground truths for evaluation set
    y_true_multilabel = []
    y_true_dialogue = []
    
    for idx, cond in sampled_items:
        if cond == "filled":
            y_true_multilabel.append(np.zeros(7))
            y_true_dialogue.append(0)
        else:
            y_true_multilabel.append(dev_y[idx])
            y_true_dialogue.append(1 if dev_y[idx].sum() > 0 else 0)
            
    y_true_multilabel = np.array(y_true_multilabel)
    y_true_dialogue = np.array(y_true_dialogue)
    
    # -----------------------------------------------------------------------
    # Setup A: Baseline (L2 Logistic Regression on subsampled raw space)
    # -----------------------------------------------------------------------
    print("\n[A] Training Baseline...")
    probes_base = []
    P_train_base = np.zeros((2 * N_train, 7))
    
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
            probes_base.append(clf)
            continue
            
        clf = LogisticRegression(penalty='l2', solver='lbfgs', C=args.C, tol=1e-2, max_iter=150, random_state=42)
        clf.fit(X_train_np, y_train_np)
        probes_base.append(clf)
        
        # 5-fold CV on subsampled training data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train_np):
            clf_cv = LogisticRegression(penalty='l2', solver='lbfgs', C=args.C, tol=1e-2, max_iter=150, random_state=42)
            clf_cv.fit(X_train_np[train_idx], y_train_np[train_idx])
            P_train_base[val_idx, d] = clf_cv.predict_proba(X_train_np[val_idx])[:, 1]
            
    # Calculate probabilities for Calibration Set
    P_cal_base_f = np.zeros((len(cal_indices), 7))
    P_cal_base_m = np.zeros((len(cal_indices), 7))
    for d in range(7):
        X_cal_f = dev_f_hs[cal_indices, d, :]
        X_cal_m = dev_m_hs[cal_indices, d, :]
        P_cal_base_f[:, d] = probes_base[d].predict_proba(X_cal_f)[:, 1]
        P_cal_base_m[:, d] = probes_base[d].predict_proba(X_cal_m)[:, 1]
        
    P_cal_base = np.concatenate([P_cal_base_f, P_cal_base_m], axis=0)
    
    y_cal_true_multilabel = []
    for idx in cal_indices:
        y_cal_true_multilabel.append(np.zeros(7))
    for idx in cal_indices:
        y_cal_true_multilabel.append(dev_y[idx])
    y_cal_true_multilabel = np.array(y_cal_true_multilabel)
    
    thresh_base = {}
    for d in range(7):
        slot = DIRS[d]
        best_th = calibrate_thresholds(P_cal_base[:, d], y_cal_true_multilabel[:, d])
        thresh_base[slot] = best_th
        
    # Evaluate Baseline
    y_pred_base = []
    for idx, cond in sampled_items:
        hs_7 = dev_f_hs[idx] if cond == "filled" else dev_m_hs[idx]
        pred_vector = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            prob = probes_base[d].predict_proba(feat)[0, 1]
            slot = DIRS[d]
            pred_vector.append(1 if prob >= thresh_base[slot] else 0)
        y_pred_base.append(pred_vector)
    y_pred_base = np.array(y_pred_base)
    
    y_pred_dialogue_base = (y_pred_base.sum(axis=1) > 0).astype(int)
    
    f1_macro_base = f1_score(y_true_multilabel, y_pred_base, average='macro', zero_division=0)
    f1_micro_base = f1_score(y_true_multilabel, y_pred_base, average='micro', zero_division=0)
    subset_acc_base = accuracy_score(y_true_multilabel, y_pred_base)
    hl_base = hamming_loss(y_true_multilabel, y_pred_base)
    dialogue_acc_base = accuracy_score(y_true_dialogue, y_pred_dialogue_base)
    
    # -----------------------------------------------------------------------
    # Setup B: Stacking (Cross-Slot Probability Fusion)
    # -----------------------------------------------------------------------
    print("\n[B] Training Stacking (L2)...")
    meta_classifiers = []
    thresh_stack = {}
    
    P_cal_stack = np.zeros_like(P_cal_base)
    P_eval_stack = np.zeros((len(sampled_items), 7))
    
    P_eval_base = []
    for idx, cond in sampled_items:
        hs_7 = dev_f_hs[idx] if cond == "filled" else dev_m_hs[idx]
        probs = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            probs.append(probes_base[d].predict_proba(feat)[0, 1])
        P_eval_base.append(probs)
    P_eval_base = np.array(P_eval_base)
    
    for d in range(7):
        slot = DIRS[d]
        y_sub_f = np.zeros(N_train)
        y_sub_m = train_y_labels[:, d]
        y_train_sub_d = np.concatenate([y_sub_f, y_sub_m], axis=0)
        
        if len(np.unique(y_train_sub_d)) <= 1:
            clf_meta = DummyZeroClassifier()
            meta_classifiers.append(clf_meta)
            thresh_stack[slot] = 0.5
            continue
            
        clf_meta = LogisticRegression(penalty='l2', C=1.0, random_state=42)
        clf_meta.fit(P_train_base, y_train_sub_d)
        meta_classifiers.append(clf_meta)
        
        P_cal_stack[:, d] = clf_meta.predict_proba(P_cal_base)[:, 1]
        P_eval_stack[:, d] = clf_meta.predict_proba(P_eval_base)[:, 1]
        
        best_th = calibrate_thresholds(P_cal_stack[:, d], y_cal_true_multilabel[:, d])
        thresh_stack[slot] = best_th
        
    y_pred_stack = np.zeros_like(P_eval_stack, dtype=int)
    for d in range(7):
        slot = DIRS[d]
        y_pred_stack[:, d] = (P_eval_stack[:, d] >= thresh_stack[slot]).astype(int)
        
    y_pred_dialogue_stack = (y_pred_stack.sum(axis=1) > 0).astype(int)
    
    f1_macro_stack = f1_score(y_true_multilabel, y_pred_stack, average='macro', zero_division=0)
    f1_micro_stack = f1_score(y_true_multilabel, y_pred_stack, average='micro', zero_division=0)
    subset_acc_stack = accuracy_score(y_true_multilabel, y_pred_stack)
    hl_stack = hamming_loss(y_true_multilabel, y_pred_stack)
    dialogue_acc_stack = accuracy_score(y_true_dialogue, y_pred_dialogue_stack)
    
    # -----------------------------------------------------------------------
    # Setup C: Feature Concatenation (Cross-Slot Hidden Fusion with L2 for speed)
    # -----------------------------------------------------------------------
    print("\n[C] Training Feature Concatenation (L2)...")
    
    X_fusion_f = train_f_hs.reshape(N_train, 7 * 2304)
    X_fusion_m = train_m_hs.reshape(N_train, 7 * 2304)
    X_fusion = np.concatenate([X_fusion_f, X_fusion_m], axis=0) # [2 * N_train, 16128]
    
    X_cal_fusion_f = dev_f_hs[cal_indices].reshape(len(cal_indices), 7 * 2304)
    X_cal_fusion_m = dev_m_hs[cal_indices].reshape(len(cal_indices), 7 * 2304)
    X_cal_fusion = np.concatenate([X_cal_fusion_f, X_cal_fusion_m], axis=0)
    
    X_eval_fusion = []
    for idx, cond in sampled_items:
        hs_7 = dev_f_hs[idx] if cond == "filled" else dev_m_hs[idx]
        X_eval_fusion.append(hs_7.reshape(7 * 2304))
    X_eval_fusion = np.array(X_eval_fusion)
    
    probes_fusion = []
    thresh_fusion = {}
    P_eval_fusion = np.zeros((len(sampled_items), 7))
    
    for d in range(7):
        slot = DIRS[d]
        y_sub_f = np.zeros(N_train)
        y_sub_m = train_y_labels[:, d]
        y_train_sub_d = np.concatenate([y_sub_f, y_sub_m], axis=0)
        
        if len(np.unique(y_train_sub_d)) <= 1:
            clf_fusion = DummyZeroClassifier()
            probes_fusion.append(clf_fusion)
            thresh_fusion[slot] = 0.5
            continue
            
        clf_fusion = LogisticRegression(penalty='l2', solver='lbfgs', C=args.C_fusion, tol=1e-2, max_iter=150, random_state=42)
        clf_fusion.fit(X_fusion, y_train_sub_d)
        probes_fusion.append(clf_fusion)
        
        probs_cal = clf_fusion.predict_proba(X_cal_fusion)[:, 1]
        best_th = calibrate_thresholds(probs_cal, y_cal_true_multilabel[:, d])
        thresh_fusion[slot] = best_th
        
        P_eval_fusion[:, d] = clf_fusion.predict_proba(X_eval_fusion)[:, 1]
        
    y_pred_fusion = np.zeros_like(P_eval_fusion, dtype=int)
    for d in range(7):
        slot = DIRS[d]
        y_pred_fusion[:, d] = (P_eval_fusion[:, d] >= thresh_fusion[slot]).astype(int)
        
    y_pred_dialogue_fusion = (y_pred_fusion.sum(axis=1) > 0).astype(int)
    
    f1_macro_fusion = f1_score(y_true_multilabel, y_pred_fusion, average='macro', zero_division=0)
    f1_micro_fusion = f1_score(y_true_multilabel, y_pred_fusion, average='micro', zero_division=0)
    subset_acc_fusion = accuracy_score(y_true_multilabel, y_pred_fusion)
    hl_fusion = hamming_loss(y_true_multilabel, y_pred_fusion)
    dialogue_acc_fusion = accuracy_score(y_true_dialogue, y_pred_dialogue_fusion)
    
    # -----------------------------------------------------------------------
    # Summary of Results
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print(" CROSS-SLOT FEATURE FUSION EVALUATION RESULTS (V3 - L2 - FAST) ")
    print("="*70)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Stacking (Prob)':<15} | {'Hidden Fusion':<15}")
    print("-"*70)
    print(f"{'Omission Macro F1':<25} | {f1_macro_base*100:6.2f}%     | {f1_macro_stack*100:6.2f}%        | {f1_macro_fusion*100:6.2f}%")
    print(f"{'Omission Micro F1':<25} | {f1_micro_base*100:6.2f}%     | {f1_micro_stack*100:6.2f}%        | {f1_micro_fusion*100:6.2f}%")
    print(f"{'Strict Subset Acc':<25} | {subset_acc_base*100:6.2f}%     | {subset_acc_stack*100:6.2f}%        | {subset_acc_fusion*100:6.2f}%")
    print(f"{'Hamming Loss (lower)':<25} | {hl_base:8.4f}     | {hl_stack:8.4f}        | {hl_fusion:8.4f}")
    print(f"{'Dialogue-level Acc':<25} | {dialogue_acc_base*100:6.2f}%     | {dialogue_acc_stack*100:6.2f}%        | {dialogue_acc_fusion*100:6.2f}%")
    print("="*70)
    
    print("\nDetailed Per-Slot F1 Score Comparison:")
    print(f"{'Slot':<12} | {'Baseline':<10} | {'Stacking':<10} | {'Hidden Fusion':<15}")
    print("-"*55)
    for d in range(7):
        slot = DIRS[d]
        f1_b = f1_score(y_true_multilabel[:, d], y_pred_base[:, d], zero_division=0)
        f1_s = f1_score(y_true_multilabel[:, d], y_pred_stack[:, d], zero_division=0)
        f1_f = f1_score(y_true_multilabel[:, d], y_pred_fusion[:, d], zero_division=0)
        print(f"{slot:<12} | {f1_b*100:6.2f}%   | {f1_s*100:6.2f}%   | {f1_f*100:6.2f}%")

if __name__ == "__main__":
    main()
