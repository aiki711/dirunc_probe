import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
import os

# Ensure common.py is accessible
sys.path.append(os.getcwd())
from scripts.common import DIRS

def compute_cosine_similarity_matrix(vectors):
    """Computes pairwise cosine similarity between vectors."""
    # vectors: (N, D)
    norm = torch.nn.functional.normalize(vectors, p=2, dim=1)
    sim_matrix = torch.mm(norm, norm.t())
    return sim_matrix

def analyze_layer(layer_data):
    A = layer_data["A"] # (N, D)
    B = layer_data["B"] # (N, D)
    labels = np.array(layer_data["label"])
    
    deltas = B - A
    
    results = {}
    
    for slot in DIRS:
        mask = (labels == slot)
        if not np.any(mask):
            continue
            
        slot_deltas = deltas[mask]
        slot_A = A[mask]
        slot_B = B[mask]
        
        # 1. Vector Consistency (Cosine Similarity of Deltas)
        if len(slot_deltas) > 1:
            sim_matrix = compute_cosine_similarity_matrix(slot_deltas)
            # Take upper triangle excluding diagonal
            triu_indices = torch.triu_indices(len(slot_deltas), len(slot_deltas), offset=1)
            avg_cos_sim = sim_matrix[triu_indices[0], triu_indices[1]].mean().item()
        else:
            avg_cos_sim = 1.0
            
        # 2. Separation Metric (LDA Accuracy & Silhouette)
        X = torch.cat([slot_A, slot_B], dim=0).numpy()
        y = np.array([0] * len(slot_A) + [1] * len(slot_B))
        
        # Silhouette Score
        sil = silhouette_score(X, y)
        
        # LDA separation power
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        lda_acc = lda.score(X, y)
        
        # 3. Distance Metrics
        centroid_a = slot_A.mean(dim=0)
        centroid_b = slot_B.mean(dim=0)
        dist = torch.norm(centroid_b - centroid_a).item()
        
        results[slot] = {
            "avg_cosine_sim": avg_cos_sim,
            "silhouette": float(sil),
            "lda_accuracy": float(lda_acc),
            "centroid_dist": float(dist),
            "count": int(mask.sum())
        }
        
    return results

def main():
    input_file = Path("runs/latent_analysis/latent_states.pt")
    output_file = Path("runs/latent_analysis/metrics.json")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Run extraction script first.")
        return
        
    print(f"Loading latent states from {input_file}...")
    data = torch.load(input_file)
    
    layers = list(data.keys())
    final_results = {}
    
    for layer in layers:
        print(f"Analyzing Layer {layer}...")
        final_results[layer] = analyze_layer(data[layer])
        
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    main()
