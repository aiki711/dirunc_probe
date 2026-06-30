#!/usr/bin/env python3
"""
Generate a beautiful, high-DPI Confusion Matrix plot for the Multinomial Probing Classifier
evaluated on the 300-sample balanced test split.
Saves to runs/identify_verify_comparison/probing_multinomial_confusion_matrix.png.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SLOTS  = ["who", "what", "when", "where", "how"]
DIRS       = ["who", "what", "when", "where", "why", "how", "which"]
ROLE_TO_DIR = {
    "Agent": "who", "Theme": "what", "Location": "where", "Source": "where",
    "Goal": "where", "Time": "when", "Manner": "how"
}

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data splits...")
    dev_rows   = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    s32 = load_s32()
    dev_pairs   = s32.PairedDirUncDataset(dev_rows).pairs
    train_pairs = s32.PairedDirUncDataset(train_rows).pairs

    # Load layer 16 dev cache to get N_dev
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
    # DO NOT shuffle for matching predictions to true labels easily
    
    # Load layer 16 train cache for Probing
    print("Loading Layer 16 train cache...")
    train_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_train.pt", map_location="cpu")
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()
    N_train = train_f_hs.shape[0]

    # Train Multinomial Probing Classifier
    print("Training Multinomial Probing Classifier...")
    X_multi_list, y_multi_list = [], []
    for i, pair in enumerate(train_pairs[:N_train]):
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
    lda = LinearDiscriminantAnalysis() # default n_components = 5
    X_lda = lda.fit_transform(X_pca, y_multi)
    
    multi_clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, class_weight='balanced')
    multi_clf.fit(X_lda, y_multi)

    # Evaluate on balanced test set to collect y_true and y_pred
    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()

    y_true_labels = []
    y_pred_labels = []
    
    for idx, cond, true_slot in eval_items:
        y_true_labels.append(true_slot)
        
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        hs_avg = hs.mean(axis=0, keepdims=True)
        hs_pca = pca.transform(hs_avg)
        hs_lda = lda.transform(hs_pca)
        pred_class = multi_clf.predict(hs_lda)[0]
        y_pred_labels.append(pred_class)

    # Calculate Confusion Matrix
    class_labels = ["None", "who", "what", "when", "where", "how"]
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=class_labels)
    
    # Normalize by row (True Class) to get recall rates per class
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0

    # Plot Confusion Matrix
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.figure(figsize=(7.5, 6), dpi=150)
    
    # Custom color palette matching the paper theme
    ax = sns.heatmap(
        cm_normalized, annot=True, fmt=".1f", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels,
        cbar_kws={'label': 'Percentage (%)'},
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )
    
    # Add actual counts in parentheses under the percentage labels
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            text = ax.texts[i * len(class_labels) + j]
            pct = float(text.get_text())
            count = cm[i, j]
            text.set_text(f"{pct:.1f}%\n({count})")
            
    plt.xlabel("Predicted Case Role Omission", fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel("True Case Role Omission", fontsize=11, fontweight='bold', labelpad=10)
    plt.title("Confusion Matrix of Multinomial Probing (Layer 16)\nBalanced Test Split Evaluation", 
              fontsize=12, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plot_path = OUT_DIR / "probing_multinomial_confusion_matrix.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved confusion matrix plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
