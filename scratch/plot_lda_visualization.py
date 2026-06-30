#!/usr/bin/env python3
"""
Generate a beautiful 2D scatter plot visualizing the LDA projection space.
Shows how the representation space of Gemma-2-2b-it (Layer 26) separates
the 6 classes: "who", "what", "when", "where", "how", and "None" (Sufficient).
"""
import json, random, os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
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
    # Load caches for Layer 16 (our optimal evaluation layer)
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

    # Reconstruct training samples for PCA+LDA
    print("Preparing training matrices...")
    X_multi_list = []
    y_multi_list = []
    for i, pair in enumerate(train_pairs[:N_train]):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_multi_list.append(train_m_hs[i].mean(axis=0)) # Missing slot text
            y_multi_list.append(slot)
            X_multi_list.append(train_f_hs[i].mean(axis=0)) # Filled text (None)
            y_multi_list.append("None")

    X_train = np.array(X_multi_list)
    y_train = np.array(y_multi_list)

    # Reconstruct dev samples for visualization
    X_dev_list = []
    y_dev_list = []
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

    # ── 1. Fit PCA and LDA on Train Split ──────────────────────────────────
    print("Fitting PCA (256 components)...")
    pca = PCA(n_components=256, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    
    print("Fitting LDA (2 components for 2D visualization)...")
    # LDA with n_components=2 forces projection into 2D space
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train_lda = lda.fit_transform(X_train_pca, y_train)

    # ── 2. Project Dev Split to the LDA Space ──────────────────────────────
    print("Projecting Dev split into 2D LDA space...")
    X_dev_pca = pca.transform(X_dev)
    X_dev_lda = lda.transform(X_dev_pca)

    # Calculate class separation metrics
    from sklearn.metrics import silhouette_score
    y_dev_binary = np.array(["Omission" if y != "None" else "None" for y in y_dev])
    sil_6class_lda = silhouette_score(X_dev_lda, y_dev)
    sil_binary_lda = silhouette_score(X_dev_lda, y_dev_binary)
    sil_6class_pca = silhouette_score(X_dev_pca, y_dev)
    sil_binary_pca = silhouette_score(X_dev_pca, y_dev_binary)
    
    print("\n====== Class Separation Metrics (Silhouette Score) ======")
    print(f"  Visual 2D LDA Space:")
    print(f"    Sufficient (None) vs. Insufficient (Omission) : {sil_binary_lda:.4f}")
    print(f"    6-Class separation (None + 5 slots)           : {sil_6class_lda:.4f}")
    print(f"  Intrinsic 256D PCA Space:")
    print(f"    Sufficient (None) vs. Insufficient (Omission) : {sil_binary_pca:.4f}")
    print(f"    6-Class separation (None + 5 slots)           : {sil_6class_pca:.4f}")
    print("=========================================================\n")

    # ── 3. Plotting ────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.figure(figsize=(10, 8), dpi=150)

    # Define colors for the 6 classes
    colors = {
        "who": "#1E88E5",    # Soft blue
        "what": "#FFB300",   # Soft amber
        "when": "#43A047",   # Soft green
        "where": "#D81B60",  # Soft pink/magenta
        "how": "#8E24AA",    # Soft purple
        "None": "#78909C"    # Slate gray (Sufficient / No Omission)
    }

    # Plot train points in background with alpha
    for class_name, col in colors.items():
        mask_train = (y_train == class_name)
        plt.scatter(
            X_train_lda[mask_train, 0], X_train_lda[mask_train, 1],
            color=col, alpha=0.12, s=8, label=None
        )

    # Plot dev points in foreground
    for class_name, col in colors.items():
        mask_dev = (y_dev == class_name)
        plt.scatter(
            X_dev_lda[mask_dev, 0], X_dev_lda[mask_dev, 1],
            color=col, alpha=0.85, s=25, edgecolor='none', label=class_name
        )

    plt.xlabel("LDA Dimension 1", fontsize=11, fontweight='bold')
    plt.ylabel("LDA Dimension 2", fontsize=11, fontweight='bold')
    plt.title("2D LDA Projection of Gemma-2-2b-it Hidden States (Layer 16)\nSemantic Case Role Omission Space", 
              fontsize=13, fontweight='bold', pad=15)
    
    plt.grid(True, linestyle="--", alpha=0.3)
    
    # Custom legend
    legend_patches = [
        mpatches.Patch(color=colors["who"], label="who (Agent) missing"),
        mpatches.Patch(color=colors["what"], label="what (Theme) missing"),
        mpatches.Patch(color=colors["when"], label="when (Time) missing"),
        mpatches.Patch(color=colors["where"], label="where (Location/Source/Goal) missing"),
        mpatches.Patch(color=colors["how"], label="how (Manner) missing"),
        mpatches.Patch(color=colors["None"], label="None (Sufficient / Complete context)")
    ]
    plt.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.95)

    # Remove top/right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = OUT_DIR / "lda_projection_2d.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved visualization to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
