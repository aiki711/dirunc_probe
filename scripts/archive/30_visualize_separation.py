import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

# Ensure common.py is accessible
sys.path.append(os.getcwd())
from scripts.common import DIRS

def plot_projection_histograms(data, layer, output_dir):
    """Plots histograms of states projected onto the slot fulfillment vector."""
    A = data[layer]["A"]
    B = data[layer]["B"]
    labels = np.array(data[layer]["label"])
    
    unique_labels = sorted(list(set(labels)))
    fig, axes = plt.subplots(len(unique_labels), 1, figsize=(8, 3 * len(unique_labels)), sharex=False)
    if len(unique_labels) == 1: axes = [axes]
    
    for i, slot in enumerate(unique_labels):
        mask = (labels == slot)
        slot_A = A[mask]
        slot_B = B[mask]
        
        # Fulfillment vector (direction of B - A)
        v = (slot_B - slot_A).mean(dim=0)
        v = v / torch.norm(v)
        
        proj_a = (slot_A @ v).numpy()
        proj_b = (slot_B @ v).numpy()
        
        sns.histplot(proj_a, color="red", label="Missing", kde=True, ax=axes[i], stat="density", alpha=0.5)
        sns.histplot(proj_b, color="blue", label="Filled", kde=True, ax=axes[i], stat="density", alpha=0.5)
        axes[i].set_title(f"Slot: {slot.upper()} (Layer {layer})")
        axes[i].legend()
        
    plt.tight_layout()
    plt.savefig(output_dir / f"projections_L{layer}.png")
    plt.close()

def plot_pca_clusters(data, layer, output_dir):
    """Plots 2D PCA clusters of all states in the layer."""
    A = data[layer]["A"].numpy()
    B = data[layer]["B"].numpy()
    labels = np.array(data[layer]["label"])
    
    X = np.concatenate([A, B], axis=0)
    # 0 for Missing, 1 for Filled
    states = np.concatenate([np.zeros(len(A)), np.ones(len(B))], axis=0)
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=states, cmap="coolwarm", alpha=0.6)
    plt.colorbar(scatter, label="State (0:Missing, 1:Filled)")
    plt.title(f"Global PCA of Latent States (Layer {layer})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    
    # Annotate some points with slot names
    for i in range(0, len(X), len(X)//20):
        plt.annotate(labels[i % len(labels)], (X_2d[i, 0], X_2d[i, 1]), fontsize=8)
        
    plt.savefig(output_dir / f"pca_L{layer}.png")
    plt.close()

def main():
    input_file = Path("runs/latent_analysis/latent_states.pt")
    output_dir = Path("results/latent_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return
        
    print(f"Loading latent states for visualization...")
    data = torch.load(input_file)
    
    for layer in data.keys():
        print(f"Generating plots for Layer {layer}...")
        plot_projection_histograms(data, layer, output_dir)
        plot_pca_clusters(data, layer, output_dir)
        
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
