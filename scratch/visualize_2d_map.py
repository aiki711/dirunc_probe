#!/usr/bin/env python3
"""
scratch/visualize_2d_map.py

Visualizes argument omission representation as a 2D Heatmap (Layer vs Token Position).
Extracts features, trains position-specific probes via 5-Fold Cross-Validation,
and plots the Standard Pair Accuracy.
"""
import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

# Import configurations and helpers dynamically from 32_train_contrastive_nq_probe
import importlib.util
spec = importlib.util.spec_from_file_location("nq_probe", "scripts/32_train_contrastive_nq_probe.py")
nq_probe = importlib.util.module_from_spec(spec)
sys.modules["nq_probe"] = nq_probe
spec.loader.exec_module(nq_probe)

ProbeModelBase = nq_probe.ProbeModelBase
PairedDirUncDataset = nq_probe.PairedDirUncDataset

# Patch dataset __getitem__ to pass through the predicate metadata
original_getitem = PairedDirUncDataset.__getitem__
def patched_getitem(self, idx):
    res = original_getitem(self, idx)
    res["predicate"] = self.aligned_pairs[idx].get("predicate", "")
    return res
PairedDirUncDataset.__getitem__ = patched_getitem


def find_predicate_index(input_ids, predicate, tokenizer):
    """Finds the token index where the verb predicate starts."""
    ids_variants = [
        tokenizer.encode(" " + predicate, add_special_tokens=False),
        tokenizer.encode(predicate, add_special_tokens=False),
        tokenizer.encode(" " + predicate.lower(), add_special_tokens=False),
        tokenizer.encode(predicate.lower(), add_special_tokens=False)
    ]
    for target_ids in ids_variants:
        n_t = len(target_ids)
        if n_t == 0:
            continue
        for i in range(len(input_ids) - n_t + 1):
            if input_ids[i:i+n_t] == target_ids:
                return i
    return None


def extract_and_cache(omission_type, layers, positions, cache_path_features, cache_path_meta, model_name="google/gemma-2-2b-it"):
    from transformers import AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()} (Device: {device})")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading base model Gemma...")
    base_model = ProbeModelBase(model_name).to(device)
    base_model.eval()
    
    dev_path = f"data/processed/case_grammar/paired_dev_gemini_{omission_type}.jsonl"
    print(f"Loading dataset from: {dev_path}")
    rows = nq_probe.read_jsonl(Path(dev_path))
    ds = PairedDirUncDataset(rows, tokenizer)
    
    # Restore predicate metadata from pairs to aligned_pairs
    for i, item in enumerate(ds.pairs):
        ds.aligned_pairs[i]["predicate"] = item["predicate"]
        
    extracted_features = []
    metadata = []
    matched_count = 0
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(ds.pairs, desc="Extracting Activations")):
            aligned = ds.aligned_pairs[i]
            filled_text = aligned["filled_text_aligned"]
            missing_text = aligned["missing_text_aligned"]
            pred = aligned["predicate"]
            
            enc_f = tokenizer([filled_text], return_tensors="pt").to(device)
            enc_m = tokenizer([missing_text], return_tensors="pt").to(device)
            
            ids_f = enc_f["input_ids"][0].tolist()
            ids_m = enc_m["input_ids"][0].tolist()
            
            t_verb_f = find_predicate_index(ids_f, pred, tokenizer)
            t_verb_m = find_predicate_index(ids_m, pred, tokenizer)
            
            if t_verb_f is None or t_verb_m is None:
                continue
                
            matched_count += 1
            
            # Run model forward passes
            out_f = base_model.lm(input_ids=enc_f["input_ids"], attention_mask=enc_f["attention_mask"])
            out_m = base_model.lm(input_ids=enc_m["input_ids"], attention_mask=enc_m["attention_mask"])
            
            hs_pair_f = np.zeros((len(layers), len(positions), base_model.hidden_size), dtype=np.float16)
            hs_pair_m = np.zeros((len(layers), len(positions), base_model.hidden_size), dtype=np.float16)
            
            for l_idx, L in enumerate(layers):
                hs_f = base_model.get_layer_hidden(out_f.hidden_states, L)[0].float().cpu().numpy().astype(np.float16)
                hs_m = base_model.get_layer_hidden(out_m.hidden_states, L)[0].float().cpu().numpy().astype(np.float16)
                
                for r_idx, r in enumerate(positions):
                    p_f = t_verb_f + r
                    p_m = t_verb_m + r
                    
                    if 0 <= p_f < len(ids_f):
                        hs_pair_f[l_idx, r_idx] = hs_f[p_f]
                    if 0 <= p_m < len(ids_m):
                        hs_pair_m[l_idx, r_idx] = hs_m[p_m]
                        
            # Shape of hs_pair_f: [num_layers, num_positions, hidden_size]
            extracted_features.append(np.stack([hs_pair_f, hs_pair_m], axis=0)) # shape [2, num_layers, num_positions, hidden_size]
            metadata.append({
                "case_role": aligned["case_role"],
                "predicate": pred,
                "dataset_name": aligned["dataset_name"]
            })
            
    print(f"Predicate matched in {matched_count}/{len(ds.pairs)} pairs ({matched_count/len(ds.pairs)*100:.1f}%)")
    
    # Save cache
    extracted_features = np.array(extracted_features, dtype=np.float16) # shape [N, 2, num_layers, num_positions, hidden_size]
    np.save(cache_path_features, extracted_features)
    with open(cache_path_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        
    print(f"Features cached at: {cache_path_features}")
    print(f"Metadata cached at: {cache_path_meta}")
    
    # Free up GPU memory
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--omission_type", type=str, default="soft", choices=["soft", "strong"])
    parser.add_argument("--role", type=str, default="all", help="Role filter (e.g. Agent, Goal, Time, or all)")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of hidden states")
    args = parser.parse_args()
    
    layers = [0, 4, 8, 12, 16, 20, 24, 26]
    positions = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    cache_dir = Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path_features = cache_dir / f"hidden_states_2d_{args.omission_type}.npy"
    cache_path_meta = cache_dir / f"hidden_states_2d_{args.omission_type}_meta.json"
    
    # 1. Extraction Phase
    if args.force_extract or not cache_path_features.exists() or not cache_path_meta.exists():
        print("Running activation extraction...")
        extract_and_cache(args.omission_type, layers, positions, cache_path_features, cache_path_meta)
    else:
        print("Loading cached activation representations...")
        
    # 2. Loading features
    features = np.load(cache_path_features) # [N, 2, num_layers, num_positions, hidden_size]
    with open(cache_path_meta, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    # 3. Filtering by role if specified
    role_to_filter = args.role.strip().lower()
    if role_to_filter != "all":
        print(f"Filtering dataset by role: {role_to_filter}")
        indices = [idx for idx, item in enumerate(metadata) if item["case_role"].lower() == role_to_filter]
        if not indices:
            print(f"No samples found for role {role_to_filter}!")
            sys.exit(1)
        filtered_features = features[indices]
        print(f"Filtered samples: {len(filtered_features)} (out of {len(features)})")
    else:
        filtered_features = features
        print(f"Using all samples: {len(filtered_features)}")
        
    # 4. Probing classifier sweeps
    acc_matrix = np.zeros((len(layers), len(positions)))
    
    print("\nTraining diagnostic probes for each coordinate (Layer x Position)...")
    for l_idx, L in enumerate(layers):
        for r_idx, r in enumerate(positions):
            # 5-Fold Cross-Validation over pairs
            fold_accs = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, test_idx in kf.split(np.arange(len(filtered_features))):
                # Construct train set (combining filled and missing)
                X_train = []
                y_train = []
                for idx in train_idx:
                    X_train.append(filtered_features[idx, 0, l_idx, r_idx]) # Filled (0)
                    y_train.append(0)
                    X_train.append(filtered_features[idx, 1, l_idx, r_idx]) # Missing (1)
                    y_train.append(1)
                    
                X_train = np.array(X_train, dtype=np.float32)
                y_train = np.array(y_train, dtype=np.int32)
                
                clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
                clf.fit(X_train, y_train)
                
                # Evaluate on test set (pairwise contrastive evaluation)
                X_test_f = np.array([filtered_features[idx, 0, l_idx, r_idx] for idx in test_idx], dtype=np.float32)
                X_test_m = np.array([filtered_features[idx, 1, l_idx, r_idx] for idx in test_idx], dtype=np.float32)
                
                probs_f = clf.predict_proba(X_test_f)[:, 1]
                probs_m = clf.predict_proba(X_test_m)[:, 1]
                
                correct = (probs_m > probs_f).sum()
                fold_accs.append(correct / len(test_idx))
                
            acc_matrix[l_idx, r_idx] = np.mean(fold_accs)
            
    # 5. Plotting Heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(11, 7))
    
    # Reverse rows so Layer 26 is at the top, Layer 0 at the bottom
    plot_matrix = acc_matrix[::-1, :]
    plot_layers = layers[::-1]
    
    sns.heatmap(
        plot_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=positions,
        yticklabels=plot_layers,
        vmin=0.5,
        vmax=1.0,
        cbar_kws={'label': 'Standard Pair Accuracy'}
    )
    
    # Highlight the Verb position at 0
    verb_col_idx = positions.index(0)
    plt.axvline(x=verb_col_idx + 0.5, color='black', linestyle='--', linewidth=2.5, label="Verb Position")
    
    plt.xlabel("Relative Token Position to Verb", fontsize=12)
    plt.ylabel("Layer Index", fontsize=12)
    plt.title(f"2D Probing Accuracy Map (Layer vs Token Position)\n({args.omission_type.capitalize()} Omission, Role: {args.role.upper()})", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"omission_2d_heatmap_{args.omission_type}_{args.role}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"\n==========================================")
    print(f"SUCCESS: 2D Heatmap saved at {out_path}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
