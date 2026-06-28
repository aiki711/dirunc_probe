#!/usr/bin/env python3
"""
Evaluate both Binary OR and Multinomial probing classifiers at Layer 16
using the FULL training dataset, and update binary_balanced_comparison_results.json.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")

ALL_SLOTS = ["who", "what", "when", "where", "how"]
ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
}
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

class DummyZeroClassifier:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0
        return out

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    dev_rows   = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    s32 = load_s32()
    dev_pairs   = s32.PairedDirUncDataset(dev_rows).pairs
    train_pairs = s32.PairedDirUncDataset(train_rows).pairs

    print("Loading dev cache for Layer 16...")
    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    # Reconstruct the exact same binary-balanced test items (seed 42)
    test_slot_missing = {s: [] for s in ALL_SLOTS}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    random.seed(42)
    neg_sampled_idx = random.sample(test_filled_idx, 150)
    neg_items = [(idx, "filled", "None") for idx in neg_sampled_idx]

    who_pool = test_slot_missing["who"]
    who_sampled = random.sample(who_pool, min(len(who_pool), 30))
    pos_items = [(idx, "missing", "who") for idx in who_sampled]
    
    rem_to_sample = 150 - len(pos_items)
    slots_left = ["what", "when", "where", "how"]
    per_slot = rem_to_sample // len(slots_left)
    for s in slots_left:
        pool = test_slot_missing[s]
        sampled = random.sample(pool, per_slot)
        pos_items.extend([(idx, "missing", s) for idx in sampled])
    
    if len(pos_items) < 150:
        needed = 150 - len(pos_items)
        extra = random.sample(test_slot_missing["when"], needed)
        pos_items.extend([(idx, "missing", "when") for idx in extra])

    eval_items = neg_items + pos_items
    random.shuffle(eval_items)

    eval_y_verify = [1 if cond == "missing" else 0 for _, cond, _ in eval_items]
    eval_y_verify = np.array(eval_y_verify)

    print("Loading train cache for Layer 16...")
    train_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_train.pt", map_location="cpu")
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()

    # Labels for train
    N_train = train_f_hs.shape[0]
    train_y = np.zeros((N_train, 7))
    for i, pair in enumerate(train_pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            d = DIRS.index(ROLE_TO_DIR[role])
            train_y[i, d] = 1.0

    # ── 1. Evaluate Multinomial Probing Classifier ──
    print("\nTraining Multinomial Probing Classifier on Layer 16 (Full Dataset)...")
    X_multi_list = []
    y_multi_list = []
    for i, pair in enumerate(train_pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_multi_list.append(train_m_hs[i].mean(axis=0))
            y_multi_list.append(slot)
            X_multi_list.append(train_f_hs[i].mean(axis=0))
            y_multi_list.append("None")
    
    X_multi = np.array(X_multi_list)
    y_multi = np.array(y_multi_list)

    pca = PCA(n_components=256, random_state=42)
    X_pca = pca.fit_transform(X_multi)
    
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_pca, y_multi)
    
    multi_clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, class_weight='balanced')
    multi_clf.fit(X_lda, y_multi)

    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()

    pred_multi = []
    for idx, cond, _ in eval_items:
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        hs_avg = hs.mean(axis=0, keepdims=True)
        hs_pca = pca.transform(hs_avg)
        hs_lda = lda.transform(hs_pca)
        pred_class = multi_clf.predict(hs_lda)[0]
        pred_multi.append(0 if pred_class == "None" else 1)

    multi_acc = accuracy_score(eval_y_verify, pred_multi)
    multi_f1  = f1_score(eval_y_verify, pred_multi)
    print(f"  Multinomial Probing -> Acc: {multi_acc*100:.2f}% | F1: {multi_f1*100:.2f}%")

    # ── 2. Evaluate Binary OR Probing Classifier ──
    print("\nTraining Binary OR Probing Classifier on Layer 16 (Full Dataset)...")
    probes = []
    for d in range(7):
        X = np.concatenate([train_f_hs[:, d, :], train_m_hs[:, d, :]], axis=0)
        y = np.concatenate([np.zeros(N_train), train_y[:, d]], axis=0)
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
        else:
            clf = LogisticRegression(solver='liblinear', tol=1e-2, max_iter=500, C=1.0, random_state=42, class_weight='balanced')
            clf.fit(X, y)
        probes.append(clf)

    # Calibrate thresholds on CAL split
    cal_slot_missing, cal_filled_idx = {s: [] for s in DIRS}, []
    for i in cal_indices:
        if i >= N_dev: continue
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            cal_slot_missing[ROLE_TO_DIR[role]].append(i)
            cal_filled_idx.append(i)

    thresholds = []
    for d, slot in enumerate(DIRS):
        pos_idxs = cal_slot_missing.get(slot, [])
        if not pos_idxs:
            thresholds.append(0.5)
            continue
        n_cal = min(50, len(pos_idxs))
        cal_pos = random.sample(pos_idxs, n_cal)
        cal_neg = random.sample(cal_filled_idx, n_cal)
        X_cal = np.concatenate([dev_m_hs[cal_pos, d, :], dev_f_hs[cal_neg, d, :]], axis=0)
        y_cal = np.array([1]*n_cal + [0]*n_cal)
        probs_cal = probes[d].predict_proba(X_cal)[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            f = f1_score(y_cal, (probs_cal >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)

    pred_or = []
    for idx, cond, _ in eval_items:
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        any_fire = False
        for d, slot in enumerate(DIRS):
            if slot in ALL_SLOTS:
                p = probes[d].predict_proba(hs[d:d+1, :])[0, 1]
                if p >= thresholds[d]:
                    any_fire = True
                    break
        pred_or.append(1 if any_fire else 0)

    or_acc = accuracy_score(eval_y_verify, pred_or)
    or_f1  = f1_score(eval_y_verify, pred_or)
    print(f"  Binary OR Probing -> Acc: {or_acc*100:.2f}% | F1: {or_f1*100:.2f}%")

    # ── 3. Update the JSON Results file ──
    res_path = OUT_DIR / "binary_balanced_comparison_results.json"
    if res_path.exists():
        data = json.loads(res_path.read_text())
        # Update both Probing methods to reflect optimal Layer 16 scores
        data["metrics"]["Probing (Ours - Multinomial)"] = {"accuracy": float(multi_acc), "f1": float(multi_f1)}
        data["metrics"]["Probing (Ours)"] = {"accuracy": float(or_acc), "f1": float(or_f1)}
        res_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"\nSuccessfully updated {res_path} with optimal Layer 16 probing metrics!")

if __name__ == "__main__":
    main()
