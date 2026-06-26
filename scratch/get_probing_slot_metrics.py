#!/usr/bin/env python3
import os
import sys
import gc
import json
import random
import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
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

def main():
    cache_dir = Path("data/cache")
    prefix = "final_token_aligned_soft"
    layer = 26
    eval_size = 300
    
    print("Loading original natural_dev.jsonl text rows...")
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    
    print("Building dev paired dataset...")
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    
    # Load dev cache to align index
    dev_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_dev.pt", map_location="cpu")
    dev_y = dev_cache["y"].numpy()
    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
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
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled = random.sample(idxs, min(len(idxs), num_per_class))
        sampled_items.extend(sampled)
        
    print(f"Sampled {len(sampled_items)} items for evaluation.")
    
    # Build ground truth roles
    y_true_role = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        if cond == "filled":
            y_true_role.append("None")
        else:
            y_true_role.append(ROLE_TO_DIR[pair["case_role"]])
            
    # Train 7 probes
    print("Loading train cache...")
    train_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_train.pt", map_location="cpu")
    
    train_f_hs = train_cache["f_hs"].float().numpy()
    train_m_hs = train_cache["m_hs"].float().numpy()
    train_y_labels = train_cache["y"].numpy()
    N_train = train_f_hs.shape[0]
    
    print("Fitting independent probes...")
    probes = []
    for d in range(7):
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y = np.concatenate([y_f, y_m], axis=0)
        
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
            clf.fit(X, y)
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X, y)
        probes.append(clf)
        
    # Evaluate Probing slot identification (6-class)
    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()
    
    # Thresholds for Layer 26 (from Sweep logs or fixed defaults)
    # Using default threshold 0.5 or calibrated thresholds
    # Let's check what thresholds are used. In scripts/51_compare_identify_verify_probing.py,
    # thresholds are hardcoded: who=0.05, what=0.05, when=0.05, where=0.1, why=0.05, how=0.05, which=0.05
    # Let's use the same THRESHOLDS to be consistent.
    THRESHOLDS = {
        "who": 0.05,
        "what": 0.05,
        "when": 0.05,
        "where": 0.1,
        "why": 0.05,
        "how": 0.05,
        "which": 0.05
    }
    
    # Evaluate Probing slot identification (Binary per-slot classification)
    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()
    
    # Thresholds for Layer 26 (from Calibration on dev set)
    # Using fixed thresholds calibrated for Layer 26:
    # Fixed Thresholds: {'who': 0.47, 'what': 0.27, 'when': 0.37, 'where': 0.32, 'why': 0.02, 'how': 0.37, 'which': 0.47}
    THRESHOLDS = {
        "who": 0.47,
        "what": 0.27,
        "when": 0.37,
        "where": 0.32,
        "why": 0.02,
        "how": 0.37,
        "which": 0.47
    }
    
    # Build ground truth binary labels for the 300 test samples
    # FIX: Use case_role from dev_pairs (not dev_y from cache) as ground truth.
    # dev_y[idx] is the cache's soft/directional label and does NOT reliably mark
    # all case_role-based missing slots as positive.  Instead we derive y_true
    # directly from the case_role field so that every Agent-missing sample gets
    # y_who=1, every Theme-missing sample gets y_what=1, etc.
    y_true_binary = []
    for idx, cond in sampled_items:
        if cond == "filled":
            y_true_binary.append(np.zeros(7))
        else:
            label = np.zeros(7)
            role = dev_pairs[idx]["case_role"]
            if role and role in CASE_ROLES:
                mapped_dir = ROLE_TO_DIR[role]
                d = DIRS.index(mapped_dir)
                label[d] = 1
            y_true_binary.append(label)
    y_true_binary = np.array(y_true_binary) # [300, 7]
    
    # Predict binary labels for the 300 test samples
    y_pred_binary = []
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
            thresh = THRESHOLDS.get(slot, 0.5)
            pred = 1 if prob >= thresh else 0
            pred_vector.append(pred)
        y_pred_binary.append(pred_vector)
    y_pred_binary = np.array(y_pred_binary) # [300, 7]
    
    # Print binary classification report per slot
    eval_slots = ["who", "when", "how", "what", "which", "where"]
    precision_vals = []
    recall_vals = []
    f1_vals = []
    
    print("\n================ Binary Per-Slot Performance (Layer 26) ================")
    for slot in eval_slots:
        d = DIRS.index(slot)
        y_true_d = y_true_binary[:, d]
        y_pred_d = y_pred_binary[:, d]
        
        p, r, f, s = precision_recall_fscore_support(y_true_d, y_pred_d, average='binary', pos_label=1, zero_division=0)
        
        precision_vals.append(round(p * 100, 2))
        recall_vals.append(round(r * 100, 2))
        f1_vals.append(round(f * 100, 2))
        
        print(f"Slot {slot:<6} | Precision: {p*100:6.2f}% | Recall: {r*100:6.2f}% | F1-Score: {f*100:6.2f}%")
        
    print("\nCopy-paste arrays for plot_slot_performance.py:")
    print("precision =", precision_vals)
    print("recall =", recall_vals)
    print("f1_score =", f1_vals)

if __name__ == "__main__":
    main()
