#!/usr/bin/env python3
"""
Evaluate the Multinomial PCA+LDA Probing Classifier on the binary-balanced test set
specifically using Layer 16 (where performance peaks) trained on the FULL training dataset.
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

ALL_SLOTS = ["who", "what", "when", "where", "how"]
ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
}
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

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

    # Load layer 16 dev cache
    print("Loading dev cache for Layer 16...")
    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

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

    # Load layer 16 train cache
    print("Loading train cache for Layer 16...")
    train_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_train.pt", map_location="cpu")
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()

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

    print("Fitting PCA + LDA + LR on Layer 16...")
    pca = PCA(n_components=256, random_state=42)
    X_pca = pca.fit_transform(X_multi)
    
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_pca, y_multi)
    
    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, class_weight='balanced')
    clf.fit(X_lda, y_multi)

    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()

    pred_suff = []
    for idx, cond, _ in eval_items:
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        hs_avg = hs.mean(axis=0, keepdims=True)
        hs_pca = pca.transform(hs_avg)
        hs_lda = lda.transform(hs_pca)
        pred_class = clf.predict(hs_lda)[0]
        pred_suff.append(0 if pred_class == "None" else 1)

    acc = accuracy_score(eval_y_verify, pred_suff)
    f1  = f1_score(eval_y_verify, pred_suff)
    print(f"\n====== Layer 16 Probing Results (Balanced Test Split) ======")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")

if __name__ == "__main__":
    main()
