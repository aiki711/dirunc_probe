#!/usr/bin/env python3
"""
Baseline C: TF-IDF + Logistic Regression for sufficiency detection.

Inputs:
  - natural_train.jsonl (text, condition)
  - natural_dev.jsonl   (test split only, via dev_test_indices.npy)

Label:
  condition == "missing" → 1 (Insufficient)
  condition == "filled"  → 0 (Sufficient)
"""
import json, random, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import torch, importlib.util, sys, os, warnings
warnings.simplefilter('ignore')

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    # ── 1. Load data ─────────────────────────────────────────────────────
    print("Loading train rows...")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    # Build (text, label) from each row
    train_X = [r["text"] for r in train_rows if "text" in r]
    train_y = [1 if r["condition"] == "missing" else 0
               for r in train_rows if "text" in r]
    print(f"  Train samples: {len(train_X)}")
    print(f"  Positive (missing): {sum(train_y)} / {len(train_y)}")

    # ── 2. Load dev pairs for test split ─────────────────────────────────
    print("\nLoading dev pairs for test split evaluation...")
    dev_rows = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32 = load_s32()
    dev_ds    = s32.PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt",
                           map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")
    print(f"  Test split: {len(test_indices)} pairs")

    # Build balanced eval set from test split (same as rerun_probing_for_comparison.py)
    CASE_ROLES = ["Agent","Theme","Location","Source","Goal","Time","Manner"]
    ROLE_TO_DIR = {
        "Agent":"who","Theme":"what","Location":"where",
        "Source":"where","Goal":"where","Time":"when","Manner":"how",
    }
    ALL_CLASSES = ["who","what","when","where","how","None"]
    test_slot_missing = {s: [] for s in ["who","what","when","where","how"]}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role","")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    class_groups = {c: [] for c in ALL_CLASSES}
    for slot, idxs in test_slot_missing.items():
        if slot in ALL_CLASSES:
            for i in idxs:
                class_groups[slot].append((i, "missing"))
    for i in test_filled_idx:
        class_groups["None"].append((i, "filled"))

    EVAL_SIZE     = 300
    num_per_class = max(1, EVAL_SIZE // 6)
    random.seed(42)
    sampled_items = []
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled_items.extend(random.sample(idxs, min(len(idxs), num_per_class)))
    print(f"  Sampled {len(sampled_items)} eval items")

    # Get texts and labels for eval
    eval_X = []
    eval_y_str = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        eval_X.append(text)
        eval_y_str.append("Insufficient" if cond == "missing" else "Sufficient")

    # ── 3. TF-IDF + LR ───────────────────────────────────────────────────
    print("\nFitting TF-IDF vectorizer (max_features=10000)...")
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                          sublinear_tf=True)
    X_train_tfidf = vec.fit_transform(train_X)
    X_eval_tfidf  = vec.transform(eval_X)

    print("Training LR classifier...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42,
                             class_weight='balanced')
    y_train_bin = np.array(train_y)
    clf.fit(X_train_tfidf, y_train_bin)

    # ── 4. Predict & evaluate ─────────────────────────────────────────────
    y_pred_bin = clf.predict(X_eval_tfidf)
    y_pred_str = ["Insufficient" if p == 1 else "Sufficient" for p in y_pred_bin]

    acc = accuracy_score(eval_y_str, y_pred_str)
    f1  = f1_score(eval_y_str, y_pred_str, pos_label="Insufficient", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        eval_y_str, y_pred_str, pos_label="Insufficient", average="binary",
        zero_division=0)

    print(f"\n====== TF-IDF + LR Results (Test Split) ======")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  Precision (Omission): {p*100:.2f}%")
    print(f"  Recall    (Omission): {r*100:.2f}%")
    print(f"  F1        (Omission): {f*100:.2f}%")

    # ── 5. Save ───────────────────────────────────────────────────────────
    results = {
        "method": "TF-IDF (unigram+bigram, 10k features) + LogisticRegression",
        "eval_split": "test (held-out 50% of dev)",
        "eval_size": len(sampled_items),
        "verify_accuracy":    float(acc),
        "verify_f1_omission": float(f),
        "verify_precision":   float(p),
        "verify_recall":      float(r),
    }
    out = OUT_DIR / "tfidf_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
