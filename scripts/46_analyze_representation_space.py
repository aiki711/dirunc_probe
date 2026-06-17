#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sys.path.append(os.getcwd())
try:
    from scripts.common import DIRS
except ImportError:
    print("Cannot import DIRS from scripts.common. Running from project root?")
    sys.exit(1)

ROLE_TO_DIR = {
    "Agent": "who",
    "Theme": "what",
    "Location": "where",
    "Source": "where",
    "Goal": "where",
    "Time": "when",
    "Manner": "how",
}
CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, required=True, help="e.g. nq_aligned_soft")
    parser.add_argument("--layer_idx", type=int, default=16, help="Layer index to analyze")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"])
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    cache_path = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = cache_path / f"{args.prefix}_layer{args.layer_idx}_{args.split}.pt"
    if not file_path.exists():
        print(f"Error: Cache file not found: {file_path}")
        sys.exit(1)
        
    print(f"Loading cached states from {file_path}...")
    data = torch.load(file_path, map_location="cpu")
    
    f_hs = data["f_hs"]          # [B, 7, D] or [B, D]
    m_hs = data["m_hs"]          # [B, 7, D] or [B, D]
    metadata = data["metadata"]
    
    # Check if cached hidden states have token dimensions
    has_tokens = len(f_hs.shape) == 3
    
    filled_vecs = []
    missing_vecs = []
    sample_roles = []
    
    for i, meta in enumerate(metadata):
        role = meta.get("case_role", "")
        if not role or role not in CASE_ROLES:
            continue
            
        dir_str = ROLE_TO_DIR[role]
        if dir_str not in DIRS:
            continue
            
        dir_idx = DIRS.index(dir_str)
        
        if has_tokens:
            f_v = f_hs[i, dir_idx, :]
            m_v = m_hs[i, dir_idx, :]
        else:
            f_v = f_hs[i, :]
            m_v = m_hs[i, :]
            
        filled_vecs.append(f_v.float().numpy())
        missing_vecs.append(m_v.float().numpy())
        sample_roles.append(role)
        
    if not filled_vecs:
        print("No valid samples found for the specified roles.")
        sys.exit(1)
        
    F = np.array(filled_vecs)  # [N, D]
    M = np.array(missing_vecs) # [N, D]
    N, D = F.shape
    print(f"Loaded {N} samples with dimension D={D}.")
    
    # -------------------------------------------------------------------------
    # 1. PCA Visualization
    # -------------------------------------------------------------------------
    print("Performing PCA on joint representation space...")
    X_joint = np.vstack([F, M])  # [2N, D]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_joint)
    
    F_pca = X_pca[:N]
    M_pca = X_pca[N:]
    
    plt.figure(figsize=(10, 8), dpi=150)
    
    # Use distinct markers for each role
    markers = {
        "Agent": "o",
        "Theme": "s",
        "Location": "^",
        "Source": "v",
        "Goal": "<",
        "Time": ">",
        "Manner": "p"
    }
    
    # Track which labels are plotted to avoid duplicate legends
    plotted_labels = set()
    
    for role in CASE_ROLES:
        role_indices = [idx for idx, r in enumerate(sample_roles) if r == role]
        if not role_indices:
            continue
            
        # Plot Filled points (Blue-ish spectrum)
        plt.scatter(
            F_pca[role_indices, 0], F_pca[role_indices, 1],
            marker=markers[role], color="#1565C0", alpha=0.6, s=50,
            label="Filled (Information Present)" if "Filled" not in plotted_labels else ""
        )
        plotted_labels.add("Filled")
        
        # Plot Missing points (Red-ish spectrum)
        plt.scatter(
            M_pca[role_indices, 0], M_pca[role_indices, 1],
            marker=markers[role], color="#D84315", alpha=0.6, s=50,
            label="Missing (Information Absent)" if "Missing" not in plotted_labels else ""
        )
        plotted_labels.add("Missing")
        
        # Draw small shift arrows for a subset of points (e.g. up to 3 points per role)
        # to visualize the shift direction
        draw_count = 0
        for idx in role_indices:
            if draw_count >= 2:
                break
            plt.arrow(
                F_pca[idx, 0], F_pca[idx, 1],
                M_pca[idx, 0] - F_pca[idx, 0], M_pca[idx, 1] - F_pca[idx, 1],
                color="gray", alpha=0.3, width=0.01, head_width=0.08, length_includes_head=True
            )
            draw_count += 1
            
    # Add dummy scatter points just for role markers legend
    for role in CASE_ROLES:
        role_indices = [idx for idx, r in enumerate(sample_roles) if r == role]
        if role_indices:
            plt.scatter(
                [], [], marker=markers[role], color="black", alpha=0.7, s=50,
                label=f"Role: {role} (n={len(role_indices)})"
            )
            
    plt.xlabel(f"PC1 (Explained Var: {pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
    plt.ylabel(f"PC2 (Explained Var: {pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
    plt.title(f"PCA of Filled vs Missing Representations (Layer {args.layer_idx}, {args.prefix.upper()})", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    pca_plot_path = out_dir / f"pca_filled_vs_missing_layer{args.layer_idx}.png"
    plt.tight_layout()
    plt.savefig(pca_plot_path, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved to {pca_plot_path}")
    
    # -------------------------------------------------------------------------
    # 2. Cosine Similarity of Missingness Shift Vectors
    # -------------------------------------------------------------------------
    print("Computing shift vectors and cosine similarity matrix...")
    # Shift vector per sample: Missing - Filled
    shifts = M - F # [N, D]
    
    # Compute average shift vector per role
    role_shifts = {}
    for role in CASE_ROLES:
        role_indices = [idx for idx, r in enumerate(sample_roles) if r == role]
        if len(role_indices) >= 5: # Require at least 5 samples to compute a robust average
            role_shifts[role] = np.mean(shifts[role_indices], axis=0)
            print(f"  Role '{role}': averaged over {len(role_indices)} samples.")
            
    active_roles = list(role_shifts.keys())
    num_active = len(active_roles)
    
    if num_active < 2:
        print("Error: Not enough active roles with >=5 samples to compute similarity matrix.")
        return
        
    cos_sim_matrix = np.zeros((num_active, num_active))
    for i in range(num_active):
        v_i = role_shifts[active_roles[i]]
        norm_i = np.linalg.norm(v_i)
        for j in range(num_active):
            v_j = role_shifts[active_roles[j]]
            norm_j = np.linalg.norm(v_j)
            
            if norm_i > 0 and norm_j > 0:
                cos_sim_matrix[i, j] = np.dot(v_i, v_j) / (norm_i * norm_j)
            else:
                cos_sim_matrix[i, j] = 0.0
                
    # Plot cosine similarity matrix as heatmap
    plt.figure(figsize=(8, 7), dpi=150)
    sns.heatmap(
        cos_sim_matrix, annot=True, fmt=".3f", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
        xticklabels=active_roles, yticklabels=active_roles, square=True,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title(f"Cosine Similarity of Missingness Shift Vectors (Layer {args.layer_idx})\n({args.prefix.upper()})", fontsize=12, fontweight="bold")
    
    heatmap_path = out_dir / f"missingness_shift_cosine_similarity_layer{args.layer_idx}.png"
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Heatmap saved to {heatmap_path}")
    
    # Print average off-diagonal cosine similarity
    off_diag_vals = cos_sim_matrix[~np.eye(cos_sim_matrix.shape[0], dtype=bool)]
    avg_sim = np.mean(off_diag_vals)
    print(f"Average off-diagonal cosine similarity: {avg_sim:.4f}")
    
    # Save statistics to JSON
    stats = {
        "prefix": args.prefix,
        "layer_idx": args.layer_idx,
        "split": args.split,
        "active_roles": active_roles,
        "cosine_similarity_matrix": cos_sim_matrix.tolist(),
        "average_off_diagonal_similarity": float(avg_sim),
        "role_sample_sizes": {role: len([idx for idx, r in enumerate(sample_roles) if r == role]) for role in CASE_ROLES}
    }
    stats_path = out_dir / f"stats_layer{args.layer_idx}.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    main()
