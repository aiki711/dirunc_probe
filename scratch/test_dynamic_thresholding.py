#!/usr/bin/env python3
import os
import sys
import gc
import json
import random
import torch
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
        res[:, 0] = 1.0
        return res
    def decision_function(self, X):
        return -999.0 * np.ones(X.shape[0])

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

def calibrate_fixed_thresholds(probes, dev_f_hs, dev_m_hs, dev_y, cal_indices, d_steps=20):
    thresholds = {}
    for d in range(7):
        slot = DIRS[d]
        X_f = dev_f_hs[cal_indices, d, :]
        X_m = dev_m_hs[cal_indices, d, :]
        
        X = np.concatenate([X_f, X_m], axis=0)
        
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
    cache_dir = Path("data/cache")
    prefix = "final_token_aligned_soft"
    layer = 26 # Use Layer 26 as identified by Layer Sweep
    eval_size = 300
    
    out_dir = Path("runs/dynamic_thresholding")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading cached tensors for Layer {layer}...")
    train_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_train.pt", map_location="cpu")
    dev_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_dev.pt", map_location="cpu")
    
    print("Loading original natural_dev.jsonl text rows...")
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    
    print("Building dev paired dataset...")
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    
    # Align dev_pairs with dev cache shape
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
        
    num_per_class = max(1, eval_size // 6)
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
    print(f"Using {len(cal_indices)} dev items for threshold calibration/fitting.")
    
    # Prepare NumPy arrays
    train_f_hs = train_cache["f_hs"].float().numpy() # [N_train, 7, D]
    train_m_hs = train_cache["m_hs"].float().numpy() # [N_train, 7, D]
    train_y_labels = train_cache["y"].numpy()        # [N_train, 7]
    N_train = train_f_hs.shape[0]
    
    dev_f_hs = dev_cache["f_hs"].float().numpy()     # [N_dev, 7, D]
    dev_m_hs = dev_cache["m_hs"].float().numpy()     # [N_dev, 7, D]
    dev_y = dev_cache["y"].numpy()                   # [N_dev, 7]
    
    # Ground truths
    y_true_multilabel = []
    y_true_dialogue = []
    for idx, cond in sampled_items:
        if cond == "filled":
            y_true_multilabel.append(np.zeros(7))
            y_true_dialogue.append(0)
        else:
            y_true_multilabel.append(dev_y[idx])
            y_true_dialogue.append(1)
    y_true_multilabel = np.array(y_true_multilabel)
    y_true_dialogue = np.array(y_true_dialogue)
    
    # ---------------------------------------------------------------
    # 1. Train Baseline Probes (L1 Logistic Regression on raw space)
    # ---------------------------------------------------------------
    print("\nTraining Baseline Probes...")
    probes = []
    for d in range(7):
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X_train = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y_train = np.concatenate([y_f, y_m], axis=0)
        
        if len(np.unique(y_train)) <= 1:
            clf = DummyZeroClassifier()
            clf.fit(X_train, y_train)
        else:
            clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
        probes.append(clf)
        
    # ---------------------------------------------------------------
    # 2. Calibrate Fixed Thresholds (Baseline)
    # ---------------------------------------------------------------
    print("Calibrating Fixed Thresholds...")
    fixed_thresholds = calibrate_fixed_thresholds(probes, dev_f_hs, dev_m_hs, dev_y, cal_indices)
    print(f"Fixed Thresholds: {fixed_thresholds}")
    
    # ---------------------------------------------------------------
    # 3. Fit Context-Adaptive Dynamic Threshold Models on Fit Split with C-Sweep
    # ---------------------------------------------------------------
    print("\nFitting Context-Adaptive Dynamic Threshold Models (with C-Sweep to avoid overfitting)...")
    # Compute context representations as the mean across the 7 slot representations
    # Shape: [N_dev, D]
    dev_f_context = np.mean(dev_f_hs, axis=1)
    dev_m_context = np.mean(dev_m_hs, axis=1)
    
    # Split calibration indices into fit (70%) and validation (30%) splits
    random.seed(42)
    shuffled_cal = list(cal_indices)
    random.shuffle(shuffled_cal)
    split_idx = int(len(shuffled_cal) * 0.7)
    fit_indices = shuffled_cal[:split_idx]
    val_indices = shuffled_cal[split_idx:]
    print(f"  Split calibration set: {len(fit_indices)} for fitting, {len(val_indices)} for threshold validation.")
    
    # Sweep candidate C values
    C_candidates = [1.0, 0.1, 0.05, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    best_c = 1.0
    best_val_macro_f1 = -1.0
    best_meta_models = []
    best_meta_thresholds = {}
    
    for C_cand in C_candidates:
        meta_models_cand = []
        for d in range(7):
            X_f_slot = dev_f_hs[fit_indices, d, :]
            X_m_slot = dev_m_hs[fit_indices, d, :]
            
            score_f = probes[d].decision_function(X_f_slot)
            score_m = probes[d].decision_function(X_m_slot)
            
            context_f = dev_f_context[fit_indices]
            context_m = dev_m_context[fit_indices]
            
            # Features for meta-model: [score_d, context_vector] (dim: 1 + D)
            features_f = np.hstack([score_f.reshape(-1, 1), context_f])
            features_m = np.hstack([score_m.reshape(-1, 1), context_m])
            
            X_meta = np.concatenate([features_f, features_m], axis=0)
            
            y_f = np.zeros(X_f_slot.shape[0])
            y_m = dev_y[fit_indices, d]
            y_meta = np.concatenate([y_f, y_m], axis=0)
            
            # Fit L2 logistic regression meta-model with C_cand and balanced class weights
            if len(np.unique(y_meta)) <= 1:
                meta_clf = DummyZeroClassifier()
                meta_clf.fit(X_meta, y_meta)
            else:
                meta_clf = LogisticRegression(penalty='l2', solver='lbfgs', C=C_cand, class_weight='balanced', tol=1e-2, max_iter=500, random_state=42)
                meta_clf.fit(X_meta, y_meta)
                
            meta_models_cand.append(meta_clf)
            
        # Calibrate thresholds on validation split
        meta_thresholds_cand = {}
        ys_val_all = []
        preds_val_all = []
        for d in range(7):
            slot = DIRS[d]
            meta_clf = meta_models_cand[d]
            
            X_f_val = dev_f_hs[val_indices, d, :]
            X_m_val = dev_m_hs[val_indices, d, :]
            score_f_val = probes[d].decision_function(X_f_val)
            score_m_val = probes[d].decision_function(X_m_val)
            
            context_f_val = dev_f_context[val_indices]
            context_m_val = dev_m_context[val_indices]
            
            features_f_val = np.hstack([score_f_val.reshape(-1, 1), context_f_val])
            features_m_val = np.hstack([score_m_val.reshape(-1, 1), context_m_val])
            X_meta_val = np.concatenate([features_f_val, features_m_val], axis=0)
            
            y_f_val = np.zeros(X_f_val.shape[0])
            y_m_val = dev_y[val_indices, d]
            y_meta_val = np.concatenate([y_f_val, y_m_val], axis=0)
            
            probs_meta = meta_clf.predict_proba(X_meta_val)[:, 1]
            
            best_th = 0.5
            best_f1 = -1.0
            for th in np.linspace(0.02, 0.98, 20):
                y_pred = (probs_meta >= th).astype(int)
                score = f1_score(y_meta_val, y_pred, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_th = th
            meta_thresholds_cand[slot] = best_th
            
            # Predict best threshold on val to select optimal C
            y_pred_val = (probs_meta >= best_th).astype(int)
            ys_val_all.append(y_meta_val)
            preds_val_all.append(y_pred_val)
            
        ys_val_all = np.array(ys_val_all).T
        preds_val_all = np.array(preds_val_all).T
        val_macro_f1 = f1_score(ys_val_all, preds_val_all, average='macro', zero_division=0)
        print(f"  Candidate C = {C_cand:<6}: Val Slot Macro F1 = {val_macro_f1*100:.2f}%")
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_c = C_cand
            best_meta_models = meta_models_cand
            best_meta_thresholds = meta_thresholds_cand

    print(f"--> Best C selected: {best_c} (Val Slot Macro F1: {best_val_macro_f1*100:.2f}%)")
    meta_models = best_meta_models
    meta_thresholds = best_meta_thresholds
    
    # Print coefficients of the selected best models
    for d in range(7):
        slot = DIRS[d]
        meta_clf = meta_models[d]
        if hasattr(meta_clf, 'coef_'):
            score_coef = meta_clf.coef_[0, 0]
            context_coef_mean = np.mean(np.abs(meta_clf.coef_[0, 1:]))
            print(f"  Slot '{slot}': score_coef = {score_coef:.4f}, mean |context_coef| = {context_coef_mean:.6f}")
            
    # ---------------------------------------------------------------
    # 4. Evaluation
    # ---------------------------------------------------------------
    # A. Baseline (Fixed Thresholds)
    y_pred_base = []
    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
        else:
            hs_7 = dev_m_hs[idx]
            
        pred_vector = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            prob = probes[d].predict_proba(feat)[0, 1]
            slot = DIRS[d]
            thresh = fixed_thresholds.get(slot, 0.5)
            pred = 1 if prob >= thresh else 0
            pred_vector.append(pred)
        y_pred_base.append(pred_vector)
    y_pred_base = np.array(y_pred_base)
    
    # B. Context-Adaptive Dynamic Thresholds (using optimal meta_models)
    y_pred_dynamic = []
    dynamic_thresholds_logged = []
    
    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
            context = dev_f_context[idx]
        else:
            hs_7 = dev_m_hs[idx]
            context = dev_m_context[idx]
            
        pred_vector = []
        sample_thresholds = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            score = probes[d].decision_function(feat)[0]
            slot = DIRS[d]
            
            # Predict using meta-model
            meta_feature = np.hstack([[score], context]).reshape(1, -1)
            meta_prob = meta_models[d].predict_proba(meta_feature)[0, 1]
            
            thresh = meta_thresholds.get(slot, 0.5)
            pred = 1 if meta_prob >= thresh else 0
            pred_vector.append(pred)
            
            if hasattr(meta_models[d], 'coef_') and meta_models[d].coef_[0, 0] > 1e-5:
                w_score = meta_models[d].coef_[0, 0]
                w_context = meta_models[d].coef_[0, 1:]
                intercept = meta_models[d].intercept_[0]
                logit_thresh = np.log(thresh / (1.0 - thresh + 1e-9) + 1e-9)
                thresh_score = (logit_thresh - intercept - np.dot(w_context, context)) / w_score
                sample_thresholds.append(thresh_score)
            else:
                sample_thresholds.append(0.0)
                
        y_pred_dynamic.append(pred_vector)
        dynamic_thresholds_logged.append(sample_thresholds)
        
    y_pred_dynamic = np.array(y_pred_dynamic)
    
    # Dialogue-level predictions
    y_pred_dialogue_base = np.array([1 if np.sum(pred) >= 1 else 0 for pred in y_pred_base])
    y_pred_dialogue_dynamic = np.array([1 if np.sum(pred) >= 1 else 0 for pred in y_pred_dynamic])
    
    # Compute metrics
    # Baseline
    diag_acc_base = accuracy_score(y_true_dialogue, y_pred_dialogue_base)
    diag_f1_base = f1_score(y_true_dialogue, y_pred_dialogue_base, pos_label=1, zero_division=0)
    slot_macro_base = f1_score(y_true_multilabel, y_pred_base, average='macro', zero_division=0)
    slot_micro_base = f1_score(y_true_multilabel, y_pred_base, average='micro', zero_division=0)
    hl_base = hamming_loss(y_true_multilabel, y_pred_base)
    
    # Dynamic Threshold
    diag_acc_dyn = accuracy_score(y_true_dialogue, y_pred_dialogue_dynamic)
    diag_f1_dyn = f1_score(y_true_dialogue, y_pred_dialogue_dynamic, pos_label=1, zero_division=0)
    slot_macro_dyn = f1_score(y_true_multilabel, y_pred_dynamic, average='macro', zero_division=0)
    slot_micro_dyn = f1_score(y_true_multilabel, y_pred_dynamic, average='micro', zero_division=0)
    hl_dyn = hamming_loss(y_true_multilabel, y_pred_dynamic)
    
    print("\n" + "="*60)
    print(" DYNAMIC THRESHOLDING VS BASELINE FIXED THRESHOLD ")
    print("="*60)
    print("Metric\t\t\tBaseline (Fixed)\tOurs (Dynamic)\tImprovement")
    print("-"*60)
    print(f"Dialogue Accuracy:\t{diag_acc_base*100:.2f}%\t\t{diag_acc_dyn*100:.2f}%\t\t{((diag_acc_dyn - diag_acc_base)*100):+.2f}%")
    print(f"Dialogue F1 (Omission):\t{diag_f1_base*100:.2f}%\t\t{diag_f1_dyn*100:.2f}%\t\t{((diag_f1_dyn - diag_f1_base)*100):+.2f}%")
    print(f"Slot Macro F1:\t\t{slot_macro_base*100:.2f}%\t\t{slot_macro_dyn*100:.2f}%\t\t{((slot_macro_dyn - slot_macro_base)*100):+.2f}%")
    print(f"Slot Micro F1:\t\t{slot_micro_base*100:.2f}%\t\t{slot_micro_dyn*100:.2f}%\t\t{((slot_micro_dyn - slot_micro_base)*100):+.2f}%")
    print(f"Hamming Loss:\t\t{hl_base:.4f}\t\t{hl_dyn:.4f}\t\t{(hl_dyn - hl_base):+.4f}")
    print("="*60)
    
    # Save comparison to JSON
    results = {
        "baseline": {
            "dialogue_accuracy": float(diag_acc_base),
            "dialogue_f1": float(diag_f1_base),
            "slot_macro_f1": float(slot_macro_base),
            "slot_micro_f1": float(slot_micro_base),
            "hamming_loss": float(hl_base)
        },
        "dynamic": {
            "dialogue_accuracy": float(diag_acc_dyn),
            "dialogue_f1": float(diag_f1_dyn),
            "slot_macro_f1": float(slot_macro_dyn),
            "slot_micro_f1": float(slot_micro_dyn),
            "hamming_loss": float(hl_dyn)
        }
    }
    with (out_dir / "dynamic_vs_fixed_results.json").open("w") as f:
        json.dump(results, f, indent=2)
        
    # Generate MD Table
    md_file = out_dir / "results.md"
    with md_file.open("w") as f:
        f.write("# Context-Adaptive Dynamic Thresholding Results\n\n")
        f.write(f"Comparison of Baseline (Fixed thresholds calibrated per slot) and Ours (Dynamic thresholds predicted from global context hidden state with best C={best_c}) on **Layer {layer}**.\n\n")
        f.write("| Metric | Baseline (Fixed Threshold) | Ours (Dynamic Threshold) | Improvement |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Dialogue Accuracy** | {diag_acc_base*100:.2f}% | **{diag_acc_dyn*100:.2f}%** | **{((diag_acc_dyn - diag_acc_base)*100):+.2f}%** |\n")
        f.write(f"| **Dialogue F1** (Omission) | {diag_f1_base*100:.2f}% | **{diag_f1_dyn*100:.2f}%** | **{((diag_f1_dyn - diag_f1_base)*100):+.2f}%** |\n")
        f.write(f"| **Slot Macro F1** | {slot_macro_base*100:.2f}% | **{slot_macro_dyn*100:.2f}%** | **{((slot_macro_dyn - slot_macro_base)*100):+.2f}%** |\n")
        f.write(f"| **Slot Micro F1** | {slot_micro_base*100:.2f}% | **{slot_micro_dyn*100:.2f}%** | **{((slot_micro_dyn - slot_micro_base)*100):+.2f}%** |\n")
        f.write(f"| **Hamming Loss** (Lower is better) | {hl_base:.4f} | **{hl_dyn:.4f}** | **{(hl_dyn - hl_base):+.4f}** |\n\n")
        f.write("### Analysis of Dynamic Threshold Behavior\n")
        f.write("Analyzing how thresholds adapt to the input context. For example, slots that are highly frequent or have high variance might see their decision boundaries lowered/raised dynamically based on the semantic complexity of the dialogue turn.\n")
        
    # Generate Plot
    metrics_plot = ['Dialogue Acc', 'Dialogue F1', 'Slot Macro F1', 'Slot Micro F1']
    base_scores = [diag_acc_base, diag_f1_base, slot_macro_base, slot_micro_base]
    dyn_scores = [diag_acc_dyn, diag_f1_dyn, slot_macro_dyn, slot_micro_dyn]
    
    x = np.arange(len(metrics_plot))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rects1 = ax.bar(x - width/2, base_scores, width, label='Baseline (Fixed Thresh)', color='#78909C')
    rects2 = ax.bar(x + width/2, dyn_scores, width, label='Ours (Dynamic Thresh)', color='#00897B')
    
    ax.set_ylabel('Scores')
    ax.set_title(f'Context-Adaptive Thresholding Comparison\n(Gemma-2-2b-it, Layer {layer})')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_plot)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = out_dir / "dynamic_threshold_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    
if __name__ == "__main__":
    main()
