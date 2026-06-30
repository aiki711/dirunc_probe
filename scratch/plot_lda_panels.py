#!/usr/bin/env python3
"""
Generate two separate LDA projection plots to address user feedback:
1. lda_sufficient_vs_insufficient.png: Visualizes Task 1 (Sufficient vs Insufficient)
   using binary coloring to show the clean left-right separation boundary.
2. lda_case_role_separation_only.png: Visualizes Task 2 by removing the "None" class
   entirely, showing how Gemma structures the 5 semantic omission slots relative to each other.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
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
    layer = 16
    print(f"Loading cached hidden states for Layer {layer}...")
    train_cache = torch.load(CACHE_DIR / f"final_token_aligned_soft_layer{layer}_train.pt", map_location="cpu")
    dev_cache   = torch.load(CACHE_DIR / f"final_token_aligned_soft_layer{layer}_dev.pt", map_location="cpu")
    
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()
    dev_f_hs    = dev_cache["f_hs"].float().numpy()
    dev_m_hs    = dev_cache["m_hs"].float().numpy()

    print("Loading data splits...")
    dev_rows   = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    s32 = load_s32()
    dev_pairs   = s32.PairedDirUncDataset(dev_rows).pairs
    train_pairs = s32.PairedDirUncDataset(train_rows).pairs

    N_train = train_f_hs.shape[0]
    N_dev   = dev_f_hs.shape[0]

    # =========================================================================
    # PLOT 1: Sufficient vs. Insufficient (Task 1) - Binary Coloring
    # =========================================================================
    print("\n--- Preparing Plot 1: Sufficient vs. Insufficient ---")
    X_train_list, y_train_list = [], []
    for i, pair in enumerate(train_pairs[:N_train]):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_train_list.append(train_m_hs[i].mean(axis=0))
            y_train_list.append(slot)
            X_train_list.append(train_f_hs[i].mean(axis=0))
            y_train_list.append("None")

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    X_dev_list, y_dev_list = [], []
    for i, pair in enumerate(dev_pairs[:N_dev]):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_dev_list.append(dev_m_hs[i].mean(axis=0))
            y_dev_list.append(slot)
            X_dev_list.append(dev_f_hs[i].mean(axis=0))
            y_dev_list.append("None")

    X_dev = np.array(X_dev_list)
    y_dev = np.array(y_dev_list)

    # PCA + LDA for Plot 1
    pca_1 = PCA(n_components=256, random_state=42)
    X_train_pca = pca_1.fit_transform(X_train)
    X_dev_pca = pca_1.transform(X_dev)

    lda_1 = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda_1.fit_transform(X_train_pca, y_train)
    X_dev_lda = lda_1.transform(X_dev_pca)

    # Plot 1
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.figure(figsize=(9, 7), dpi=150)

    # Color scheme for binary plot
    c_sufficient   = "#78909C"  # Slate Blue-Gray
    c_insufficient = "#FF7043"  # Vibrant Coral/Orange

    # Plot train points in background
    for is_none in [True, False]:
        col = c_sufficient if is_none else c_insufficient
        mask_train = (y_train == "None") if is_none else (y_train != "None")
        plt.scatter(
            X_train_lda[mask_train, 0], X_train_lda[mask_train, 1],
            color=col, alpha=0.08, s=8, label=None
        )

    # Plot dev points in foreground
    for is_none in [True, False]:
        col = c_sufficient if is_none else c_insufficient
        lbl = "Sufficient (Complete Context)" if is_none else "Insufficient (Omission Present)"
        mask_dev = (y_dev == "None") if is_none else (y_dev != "None")
        plt.scatter(
            X_dev_lda[mask_dev, 0], X_dev_lda[mask_dev, 1],
            color=col, alpha=0.85, s=25, edgecolor='none', label=lbl
        )

    plt.xlabel("LDA Dimension 1 (Information Sufficiency Axis)", fontsize=11, fontweight='bold')
    plt.ylabel("LDA Dimension 2 (Case Role Discriminator Axis)", fontsize=11, fontweight='bold')
    plt.title("LDA Separation: Sufficient vs. Insufficient Context\nGemma-2-2b-it Hidden States (Layer 16)", 
              fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", fontsize=10, framealpha=0.95)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path_1 = OUT_DIR / "lda_sufficient_vs_insufficient.png"
    plt.savefig(plot_path_1, bbox_inches='tight')
    print(f"Saved Plot 1 to {plot_path_1}")
    plt.close()


    # =========================================================================
    # PLOT 2: Case Role Separation Only (Excluding "None")
    # =========================================================================
    print("\n--- Preparing Plot 2: Case Role Separation Only (Excluding None) ---")
    
    # Filter training data: keep only missing slots (exclude "None")
    X_train_slot_list, y_train_slot_list = [], []
    for i, pair in enumerate(train_pairs[:N_train]):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_train_slot_list.append(train_m_hs[i].mean(axis=0))
            y_train_slot_list.append(slot)

    X_train_slot = np.array(X_train_slot_list)
    y_train_slot = np.array(y_train_slot_list)

    # Filter dev data: keep only missing slots (exclude "None")
    X_dev_slot_list, y_dev_slot_list = [], []
    for i, pair in enumerate(dev_pairs[:N_dev]):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_dev_slot_list.append(dev_m_hs[i].mean(axis=0))
            y_dev_slot_list.append(slot)

    X_dev_slot = np.array(X_dev_slot_list)
    y_dev_slot = np.array(y_dev_slot_list)

    # PCA + LDA for Plot 2 (5 classes: who, what, when, where, how)
    pca_2 = PCA(n_components=256, random_state=42)
    X_train_slot_pca = pca_2.fit_transform(X_train_slot)
    X_dev_slot_pca = pca_2.transform(X_dev_slot)

    # For 5 classes, max components is 4
    lda_2 = LinearDiscriminantAnalysis(n_components=2)
    X_train_slot_lda = lda_2.fit_transform(X_train_slot_pca, y_train_slot)
    X_dev_slot_lda = lda_2.transform(X_dev_slot_pca)

    # Calculate Silhouette score for just the 5 slots
    sil_5slots_2d = silhouette_score(X_dev_slot_lda, y_dev_slot)
    sil_5slots_raw = silhouette_score(X_dev_slot_pca, y_dev_slot)
    print(f"Silhouette Score (5 slots - 2D LDA) : {sil_5slots_2d:.4f}")
    print(f"Silhouette Score (5 slots - 256D PCA): {sil_5slots_raw:.4f}")

    # Plot 2
    plt.figure(figsize=(9, 7), dpi=150)

    colors_slot = {
        "who": "#1E88E5",    # Soft blue
        "what": "#FFB300",   # Soft amber
        "when": "#43A047",   # Soft green
        "where": "#D81B60",  # Soft pink/magenta
        "how": "#8E24AA",    # Soft purple
    }

    # Plot train points in background
    for class_name, col in colors_slot.items():
        mask_train = (y_train_slot == class_name)
        plt.scatter(
            X_train_slot_lda[mask_train, 0], X_train_slot_lda[mask_train, 1],
            color=col, alpha=0.10, s=8, label=None
        )

    # Plot dev points in foreground
    for class_name, col in colors_slot.items():
        mask_dev = (y_dev_slot == class_name)
        plt.scatter(
            X_dev_slot_lda[mask_dev, 0], X_dev_slot_lda[mask_dev, 1],
            color=col, alpha=0.85, s=30, edgecolor='none', label=class_name
        )

    plt.xlabel("LDA Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("LDA Dimension 2", fontsize=11, fontweight='bold')
    plt.title(f"Case Role Omission Space (Excluding 'None')\nGemma-2-2b-it Hidden States (Layer 16) | Slot Silhouette Score: {sil_5slots_2d:.4f}", 
              fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle="--", alpha=0.3)
    
    legend_patches = [
        mpatches.Patch(color=colors_slot["who"], label="who (Agent) missing"),
        mpatches.Patch(color=colors_slot["what"], label="what (Theme) missing"),
        mpatches.Patch(color=colors_slot["when"], label="when (Time) missing"),
        mpatches.Patch(color=colors_slot["where"], label="where (Location/Source/Goal) missing"),
        mpatches.Patch(color=colors_slot["how"], label="how (Manner) missing"),
    ]
    plt.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.95)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plot_path_2 = OUT_DIR / "lda_case_role_separation_only.png"
    plt.savefig(plot_path_2, bbox_inches='tight')
    print(f"Saved Plot 2 to {plot_path_2}")
    plt.close()

if __name__ == "__main__":
    main()
