#!/usr/bin/env python3
"""
Slot-level binary probing evaluation with proper cal/test split.

Data protocol:
  train  → probe training
  cal    → threshold calibration  (50% of dev, stratified by case_role, seed=42)
  test   → final evaluation       (remaining 50%, held-out)

Layer 26 was selected via layer sweep on full dev (acknowledged as limitation;
results on the held-out test split are still unbiased for threshold decisions).
"""
import os, sys, json, random, warnings
import torch, numpy as np
warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from scripts.common import DIRS
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, f1_score
import importlib.util

# ── Constants ────────────────────────────────────────────────────────────────
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

class DummyZeroClassifier:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0
        return out

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
                                                   "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def build_caserole_y(pairs):
    """[N, 7] binary label matrix from case_role metadata."""
    y = np.zeros((len(pairs), 7), dtype=np.float32)
    for i, pair in enumerate(pairs):
        role = pair.get("case_role", "")
        if role and role in ROLE_TO_DIR:
            d = DIRS.index(ROLE_TO_DIR[role])
            y[i, d] = 1.0
    return y

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR = Path("data/cache")
    PREFIX    = "final_token_aligned_soft"
    LAYER     = 26
    N_CAL     = 50    # calibration samples per slot (positive & negative)
    N_EVAL    = 50    # evaluation samples per slot

    # ── 1. Load dev cache + cal/test split ───────────────────────────────
    print(f"Loading layer {LAYER} dev cache...")
    dev_cache = torch.load(CACHE_DIR / f"{PREFIX}_layer{LAYER}_dev.pt", map_location="cpu")
    dev_f_hs  = dev_cache["f_hs"].float().numpy()   # [N_dev, 7, D]
    dev_m_hs  = dev_cache["m_hs"].float().numpy()
    dev_meta  = dev_cache["metadata"]               # list of dicts

    print("Loading dev JSONL pairs...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32 = load_s32()
    PairedDirUncDataset = s32.PairedDirUncDataset
    dev_ds    = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
        dev_pairs = dev_pairs[:dev_cache["f_hs"].shape[0]]
    N_dev = len(dev_pairs)
    print(f"  Total dev pairs: {N_dev}")

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")
    print(f"  Cal split:  {len(cal_indices)} pairs")
    print(f"  Test split: {len(test_indices)} pairs")

    # ── 2. Build slot→indices maps for CAL and TEST ───────────────────────
    def slot_maps(indices):
        """Returns: slot_missing[slot] = list of indices, all_filled = list of indices"""
        slot_missing = {s: [] for s in DIRS}
        all_filled   = []
        for i in indices:
            pair = dev_pairs[i]
            role = pair.get("case_role", "")
            if role and role in ROLE_TO_DIR:
                slot_missing[ROLE_TO_DIR[role]].append(i)
                all_filled.append(i)      # filled version at same index
        return slot_missing, all_filled

    cal_slot_missing, cal_filled_idx  = slot_maps(cal_indices)
    test_slot_missing, test_filled_idx = slot_maps(test_indices)

    print("\n=== Dev class group sizes ===")
    for s in DIRS:
        print(f"  {s:8s}: cal={len(cal_slot_missing[s]):3d}  test={len(test_slot_missing[s]):3d}")

    # ── 3. Load train cache + train 7 binary probes ───────────────────────
    print("\nLoading train rows and building pairs...")
    train_rows  = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    train_ds    = PairedDirUncDataset(train_rows)
    train_pairs = train_ds.pairs

    train_cache = torch.load(CACHE_DIR / f"{PREFIX}_layer{LAYER}_train.pt", map_location="cpu")
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()
    N_train     = train_f_hs.shape[0]
    if len(train_pairs) != N_train:
        train_pairs = train_pairs[:N_train]

    train_y = build_caserole_y(train_pairs)

    print("\n=== Training positives per slot (case_role-based) ===")
    for d, slot in enumerate(DIRS):
        print(f"  {slot:8s}: {int(train_y[:, d].sum()):5d} / {N_train}")

    print("\nFitting 7 probes with case_role labels...")
    probes = []
    for d in range(7):
        X = np.concatenate([train_f_hs[:, d, :], train_m_hs[:, d, :]], axis=0)
        y = np.concatenate([np.zeros(N_train), train_y[:, d]], axis=0)
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                                     class_weight='balanced')
        clf.fit(X, y)
        probes.append(clf)
        print(f"  Slot {DIRS[d]:8s}: pos={int(train_y[:, d].sum())} / {N_train}")

    # ── 4. Calibrate per-slot thresholds on CAL split ONLY ───────────────
    print("\n=== Calibrating per-slot thresholds on CAL split ===")
    random.seed(0)
    thresholds = []
    for d, slot in enumerate(DIRS):
        pos_idxs = cal_slot_missing.get(slot, [])
        if not pos_idxs:
            thresholds.append(0.5)
            continue
        n_cal = min(N_CAL, len(pos_idxs))
        cal_pos = random.sample(pos_idxs, n_cal)
        cal_neg = random.sample(cal_filled_idx, n_cal)
        X_cal = np.concatenate([
            dev_m_hs[cal_pos, d, :],
            dev_f_hs[cal_neg, d, :],
        ], axis=0)
        y_cal = np.array([1]*n_cal + [0]*n_cal, dtype=np.float32)
        probs_cal = probes[d].predict_proba(X_cal)[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            f = f1_score(y_cal, (probs_cal >= t).astype(int),
                         pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)
        print(f"  {slot:8s}: threshold={best_t:.3f}  cal_F1={best_f1:.3f}  "
              f"(n_cal={n_cal})")

    # ── 5. Per-slot isolated evaluation on TEST split ─────────────────────
    print("\n====== Binary Per-Slot Performance on HELD-OUT TEST SPLIT "
          f"(Layer {LAYER}) ======")
    eval_slots = ["who", "when", "how", "what", "where"]   # 'which' removed: no data
    precision_vals, recall_vals, f1_vals = [], [], []

    random.seed(42)
    for slot in eval_slots:
        d = DIRS.index(slot)
        pos_idxs = test_slot_missing.get(slot, [])
        if not pos_idxs:
            precision_vals.append(0.0)
            recall_vals.append(0.0)
            f1_vals.append(0.0)
            print(f"Slot {slot:<6} | no positives in test split")
            continue

        n_eval   = min(N_EVAL, len(pos_idxs))
        eval_pos = random.sample(pos_idxs, n_eval)
        eval_neg = random.sample(test_filled_idx, n_eval)

        probs_pos = probes[d].predict_proba(dev_m_hs[eval_pos, d, :])[:, 1]
        probs_neg = probes[d].predict_proba(dev_f_hs[eval_neg, d, :])[:, 1]
        probs     = np.concatenate([probs_pos, probs_neg])
        y_true_s  = np.array([1]*n_eval + [0]*n_eval)
        y_pred_s  = (probs >= thresholds[d]).astype(int)

        p, r, f, _ = precision_recall_fscore_support(
            y_true_s, y_pred_s, average='binary', pos_label=1, zero_division=0)
        precision_vals.append(round(p * 100, 2))
        recall_vals.append(round(r * 100, 2))
        f1_vals.append(round(f * 100, 2))
        print(f"Slot {slot:<6} | P: {p*100:6.2f}% | R: {r*100:6.2f}% | "
              f"F1: {f*100:6.2f}%  (n_eval={n_eval})")

    print("\nCopy-paste arrays for plot_slot_performance.py:")
    print("precision =", precision_vals)
    print("recall    =", recall_vals)
    print("f1_score  =", f1_vals)

    # Save for comprehensive comparison
    results = {
        "eval_split": "test (held-out 50% of dev, stratified by case_role)",
        "layer": LAYER,
        "label_source": "case_role (corrected)",
        "cal_split_size": len(cal_indices),
        "test_split_size": len(test_indices),
        "slots": eval_slots,
        "precision": precision_vals,
        "recall": recall_vals,
        "f1": f1_vals,
    }
    out = Path("runs/identify_verify_comparison/slot_metrics_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
