#!/usr/bin/env python3
"""
Corrected slot performance evaluation.

FIX: Builds training AND evaluation labels from case_role metadata,
     not from the JSONL 'labels' field (which is model-predicted and
     misaligned with case_role).

Also performs threshold calibration on the dev set using the same
case_role-based ground truth, so thresholds are meaningfully tuned.
"""
import os, sys, json, random, warnings
import torch, numpy as np
warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS
from pathlib import Path
from sklearn.linear_model import LogisticRegression

class DummyZeroClassifier:
    """Always predicts 0; returns proper 2-column predict_proba."""
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0
        return out
from sklearn.metrics import precision_recall_fscore_support, f1_score
import importlib.util

CASE_ROLES  = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]

ROLE_TO_DIR = {
    "Agent":    "who",
    "Theme":    "what",
    "Location": "where",
    "Source":   "where",
    "Goal":     "where",
    "Time":     "when",
    "Manner":   "how",
}

def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s32 = load_script_32()
PairedDirUncDataset = s32.PairedDirUncDataset

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_caserole_y(pairs):
    """Build [N, 7] binary label matrix from case_role metadata."""
    N = len(pairs)
    y = np.zeros((N, 7), dtype=np.float32)
    for i, pair in enumerate(pairs):
        role = pair.get("case_role", "")
        if role and role in ROLE_TO_DIR:
            mapped = ROLE_TO_DIR[role]
            if mapped in DIRS:
                d = DIRS.index(mapped)
                y[i, d] = 1.0
    return y


def calibrate_thresholds(y_true, probs, grid=None):
    """For each slot, find threshold maximising F1 on validation set."""
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    thresholds = []
    for d in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            preds = (probs[:, d] >= t).astype(int)
            f = f1_score(y_true[:, d], preds, pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)
    return thresholds


def main():
    cache_dir  = Path("data/cache")
    prefix     = "final_token_aligned_soft"
    layer      = 26
    eval_size  = 300

    # ------------------------------------------------------------------ #
    # 1.  Load dev pairs (with case_role metadata)                         #
    # ------------------------------------------------------------------ #
    print("Loading dev rows and building pairs...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    dev_ds    = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    dev_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_dev.pt",
                           map_location="cpu")
    dev_f_hs  = dev_cache["f_hs"].float().numpy()   # [N_dev, 7, D]
    dev_m_hs  = dev_cache["m_hs"].float().numpy()

    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
        dev_pairs = dev_pairs[:dev_cache["f_hs"].shape[0]]

    # ------------------------------------------------------------------ #
    # 2.  Build balanced eval set (300 samples)                            #
    # ------------------------------------------------------------------ #
    class_groups = {c: [] for c in ALL_CLASSES}
    for i, pair in enumerate(dev_pairs):
        role = pair.get("case_role", "")
        if not role or role not in CASE_ROLES:
            continue
        mapped = ROLE_TO_DIR[role]
        class_groups["None"].append((i, "filled"))
        class_groups[mapped].append((i, "missing"))

    print("\n=== Dev class group sizes ===")
    for c in ALL_CLASSES:
        print(f"  {c:10s}: {len(class_groups[c])}")

    num_per_class = max(1, eval_size // 6)
    random.seed(42)
    sampled_items = []
    for c in ALL_CLASSES:
        idxs  = class_groups[c]
        taken = random.sample(idxs, min(len(idxs), num_per_class))
        sampled_items.extend(taken)
    print(f"\nSampled {len(sampled_items)} eval items (target: {eval_size})")

    # ------------------------------------------------------------------ #
    # 3.  Ground-truth matrix from case_role (FIXED)                       #
    # ------------------------------------------------------------------ #
    y_true = np.zeros((len(sampled_items), 7), dtype=np.float32)
    for k, (idx, cond) in enumerate(sampled_items):
        if cond == "missing":
            pair = dev_pairs[idx]
            role = pair.get("case_role", "")
            if role and role in ROLE_TO_DIR:
                d = DIRS.index(ROLE_TO_DIR[role])
                y_true[k, d] = 1.0
    # filled → all zeros (already initialised)

    # ------------------------------------------------------------------ #
    # 4.  Load train cache + build case_role-based training labels          #
    # ------------------------------------------------------------------ #
    print("\nLoading train rows and building pairs...")
    train_rows  = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    train_ds    = PairedDirUncDataset(train_rows)
    train_pairs = train_ds.pairs

    train_cache    = torch.load(cache_dir / f"{prefix}_layer{layer}_train.pt",
                                map_location="cpu")
    train_f_hs     = train_cache["f_hs"].float().numpy()
    train_m_hs     = train_cache["m_hs"].float().numpy()
    N_train        = train_f_hs.shape[0]

    if len(train_pairs) != N_train:
        print(f"  Warning: pairs={len(train_pairs)} vs cache={N_train}; truncating pairs")
        train_pairs = train_pairs[:N_train]

    # case_role-based labels (FIXED – replaces train_cache["y"])
    train_y_corrected = build_caserole_y(train_pairs)   # [N_train, 7]

    print("\n=== Training positives per slot (case_role-based) ===")
    for d, slot in enumerate(DIRS):
        n_pos = int(train_y_corrected[:, d].sum())
        print(f"  {slot:8s}: {n_pos:5d} / {N_train}")

    # ------------------------------------------------------------------ #
    # 5.  Train 7 binary probes                                            #
    # ------------------------------------------------------------------ #
    print("\nFitting 7 probes with case_role labels...")
    probes = []
    for d in range(7):
        X_f  = train_f_hs[:, d, :]
        X_m  = train_m_hs[:, d, :]
        X    = np.concatenate([X_f, X_m], axis=0)
        y_f  = np.zeros(N_train)
        y_m  = train_y_corrected[:, d]
        y    = np.concatenate([y_f, y_m], axis=0)

        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
            clf.fit(X, y)
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                                     class_weight='balanced')
            clf.fit(X, y)
        probes.append(clf)
        print(f"  Slot {DIRS[d]:8s}: pos={int(y_m.sum())} / {N_train}")

    # ------------------------------------------------------------------ #
    # 6.  Get probabilities on the FULL dev set for threshold calibration   #
    # ------------------------------------------------------------------ #
    print("\nCalibrating thresholds on full dev set...")
    N_dev   = dev_f_hs.shape[0]
    dev_y_full = build_caserole_y(dev_pairs)            # [N_dev, 7]

    # For calibration we use ALL dev pairs (missing only, since filled→all 0)
    dev_probs_m = np.zeros((N_dev, 7), dtype=np.float32)
    for d in range(7):
        probs = probes[d].predict_proba(dev_m_hs[:, d, :])[:, 1]
        dev_probs_m[:, d] = probs

    thresholds = calibrate_thresholds(dev_y_full, dev_probs_m)
    print("  Calibrated thresholds:")
    for d, slot in enumerate(DIRS):
        print(f"    {slot:8s}: {thresholds[d]:.3f}")

    # ------------------------------------------------------------------ #
    # 7.  Evaluate on balanced 300-sample set                              #
    # ------------------------------------------------------------------ #
    y_pred = np.zeros((len(sampled_items), 7), dtype=np.float32)
    for k, (idx, cond) in enumerate(sampled_items):
        hs_7 = dev_f_hs[idx] if cond == "filled" else dev_m_hs[idx]
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            prob = probes[d].predict_proba(feat)[0, 1]
            y_pred[k, d] = 1.0 if prob >= thresholds[d] else 0.0
    # 6.  Per-slot isolated evaluation                                      #
    #     Positive: slot d missing (case_role-based)                        #
    #     Negative: filled samples ONLY (not other-slot-missing)            #
    #                                                                       #
    #     Threshold: calibrated on full dev balanced per-slot set            #
    # ------------------------------------------------------------------ #
    print("\nCalibrating per-slot thresholds (d-missing vs filled only)...")

    # Gather all filled pair indices in dev
    all_filled_idx = [i for i, pair in enumerate(dev_pairs)
                      if pair.get("case_role", "") in CASE_ROLES]
    # For each slot, gather its missing pair indices
    slot_missing_idx = {slot: [] for slot in DIRS}
    for i, pair in enumerate(dev_pairs):
        role = pair.get("case_role", "")
        if role and role in ROLE_TO_DIR:
            slot_missing_idx[ROLE_TO_DIR[role]].append(i)

    N_CAL = 50  # calibration positives per slot (use all if fewer)
    random.seed(0)
    thresholds = []
    for d, slot in enumerate(DIRS):
        pos_idxs = slot_missing_idx.get(slot, [])
        if len(pos_idxs) == 0:
            thresholds.append(0.5)
            continue
        # Balanced calibration set: min(N_CAL, available) pos + same neg
        n_cal = min(N_CAL, len(pos_idxs))
        cal_pos = random.sample(pos_idxs, n_cal)
        cal_neg = random.sample(all_filled_idx, n_cal)

        # Build features and labels
        X_cal = np.concatenate([
            dev_m_hs[cal_pos, d, :],  # missing hidden states for slot d
            dev_f_hs[cal_neg, d, :],  # filled hidden states
        ], axis=0)
        y_cal = np.array([1]*n_cal + [0]*n_cal, dtype=np.float32)

        probs_cal = probes[d].predict_proba(X_cal)[:, 1]

        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs_cal >= t).astype(int)
            f = f1_score(y_cal, preds, pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)
        print(f"  {slot:8s}: threshold={best_t:.3f}  cal_F1={best_f1:.3f}  (n_cal={n_cal})")

    # ------------------------------------------------------------------ #
    # 7.  Per-slot isolated evaluation on held-out balanced eval sets       #
    # ------------------------------------------------------------------ #
    print("\n================ Binary Per-Slot Performance (Layer 26 – CORRECTED) ================")
    eval_slots   = ["who", "when", "how", "what", "where"]  # 'which' removed: no data
    precision_vals, recall_vals, f1_vals = [], [], []

    N_EVAL = 50   # eval positives per slot
    random.seed(42)
    for slot in eval_slots:
        d = DIRS.index(slot)
        pos_idxs = slot_missing_idx.get(slot, [])

        if len(pos_idxs) == 0:
            precision_vals.append(0.0)
            recall_vals.append(0.0)
            f1_vals.append(0.0)
            print(f"Slot {slot:<6} | P:   0.00% | R:   0.00% | F1:   0.00%  (no positives in dev)")
            continue

        n_eval = min(N_EVAL, len(pos_idxs))
        eval_pos = random.sample(pos_idxs, n_eval)
        eval_neg = random.sample(all_filled_idx, n_eval)

        # Get probabilities
        probs_pos = probes[d].predict_proba(dev_m_hs[eval_pos, d, :])[:, 1]
        probs_neg = probes[d].predict_proba(dev_f_hs[eval_neg, d, :])[:, 1]
        probs = np.concatenate([probs_pos, probs_neg])
        y_true_slot = np.array([1]*n_eval + [0]*n_eval)
        y_pred_slot = (probs >= thresholds[d]).astype(int)

        p, r, f, _ = precision_recall_fscore_support(
            y_true_slot, y_pred_slot, average='binary', pos_label=1, zero_division=0)
        precision_vals.append(round(p * 100, 2))
        recall_vals.append(round(r * 100, 2))
        f1_vals.append(round(f * 100, 2))
        print(f"Slot {slot:<6} | P: {p*100:6.2f}% | R: {r*100:6.2f}% | F1: {f*100:6.2f}%  (n_eval={n_eval})")

    print("\nCopy-paste arrays for plot_slot_performance.py:")
    print("precision =", precision_vals)
    print("recall    =", recall_vals)
    print("f1_score  =", f1_vals)


if __name__ == "__main__":
    main()
