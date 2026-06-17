import os
import sys
import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main():
    cache_path_features = "data/cache/hidden_states_2d_soft.npy"
    cache_path_meta = "data/cache/hidden_states_2d_soft_meta.json"
    
    if not os.path.exists(cache_path_features):
        print(f"Error: Cache file {cache_path_features} not found.")
        return
        
    print("Loading features...")
    features = np.load(cache_path_features) # [N, 2, num_layers, num_positions, hidden_size]
    with open(cache_path_meta, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    print(f"Features shape: {features.shape}")
    
    # Layer 12 (index 3), Position 0 (index 3 in positions [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    l_idx = 3
    r_idx = 3
    
    # Extract data
    N = len(features)
    X_filled = features[:, 0, l_idx, r_idx]
    X_missing = features[:, 1, l_idx, r_idx]
    
    print(f"Sample Filled feature range: min={X_filled.min()}, max={X_filled.max()}, mean={X_filled.mean()}")
    print(f"Sample Missing feature range: min={X_missing.min()}, max={X_missing.max()}, mean={X_missing.mean()}")
    
    # Check if features are zero vectors
    zero_filled_count = np.sum(np.all(X_filled == 0, axis=-1))
    zero_missing_count = np.sum(np.all(X_missing == 0, axis=-1))
    print(f"Number of zero vectors - Filled: {zero_filled_count}/{N}, Missing: {zero_missing_count}/{N}")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for use_scaling in [False, True]:
        print(f"\n--- Evaluation (StandardScaler: {use_scaling}) ---")
        fold_accs = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):
            X_train = []
            y_train = []
            for idx in train_idx:
                X_train.append(features[idx, 0, l_idx, r_idx]) # Filled (0)
                y_train.append(0)
                X_train.append(features[idx, 1, l_idx, r_idx]) # Missing (1)
                y_train.append(1)
                
            X_train = np.array(X_train, dtype=np.float32)
            y_train = np.array(y_train, dtype=np.int32)
            
            X_test_f = np.array([features[idx, 0, l_idx, r_idx] for idx in test_idx], dtype=np.float32)
            X_test_m = np.array([features[idx, 1, l_idx, r_idx] for idx in test_idx], dtype=np.float32)
            
            if use_scaling:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test_f = scaler.transform(X_test_f)
                X_test_m = scaler.transform(X_test_m)
                
            clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
            clf.fit(X_train, y_train)
            
            probs_f = clf.predict_proba(X_test_f)[:, 1]
            probs_m = clf.predict_proba(X_test_m)[:, 1]
            
            correct = (probs_m > probs_f).sum()
            acc = correct / len(test_idx)
            fold_accs.append(acc)
            
            if fold == 0:
                print(f"First 10 test sample probabilities (StandardScaler: {use_scaling}):")
                for i in range(min(10, len(test_idx))):
                    print(f"  idx {test_idx[i]}: Filled_prob={probs_f[i]:.4f}, Missing_prob={probs_m[i]:.4f}, (m > f)={probs_m[i] > probs_f[i]}")
                    
        print(f"Mean Pairwise Accuracy: {np.mean(fold_accs):.4f}")

if __name__ == "__main__":
    main()
