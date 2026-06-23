#!/usr/bin/env python3
import os
import sys
import gc
import json
import random
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import warnings

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]

ROLE_TO_DIR = {
    "Agent": "who",
    "Theme": "what",
    "Location": "where",
    "Source": "where",
    "Goal": "where",
    "Time": "when",
    "Manner": "how"
}

class DummyZeroClassifier:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(X.shape[0])
    def predict_proba(self, X):
        res = np.zeros((X.shape[0], 2))
        res[:, 0] = 1.0 # 100% chance of class 0 (Sufficient)
        return res

# Dynamically import PairedDirUncDataset
def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s32 = load_script_32()
PairedDirUncDataset = s32.PairedDirUncDataset

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def calibrate_thresholds(probes, dev_f_hs, dev_m_hs, dev_y, cal_indices, d_steps=20):
    thresholds = {}
    for d in range(7):
        slot = DIRS[d]
        X_f = dev_f_hs[cal_indices, d, :]
        X_m = dev_m_hs[cal_indices, d, :]
        
        X = np.concatenate([X_f, X_m], axis=0)
        
        y_f = np.zeros(X_f.shape[0])
        y_m = dev_y[cal_indices, d]
        y_true = np.concatenate([y_f, y_m], axis=0)
        
        probs = probes[d].predict_proba(X)[:, 1]
        
        best_thresh = 0.5
        best_f1 = -1.0
        
        for th in np.linspace(0.02, 0.98, d_steps):
            y_pred = (probs >= th).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = th
                
        thresholds[slot] = best_thresh
    return thresholds

def run_evaluation(probes, dev_f_hs, dev_m_hs, sampled_items, thresholds):
    y_preds = []
    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
        else:
            hs_7 = dev_m_hs[idx]
            
        pred_vector = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            prob = probes[d].predict_proba(feat)[0, 1]
            slot = DIRS[d]
            thresh = thresholds.get(slot, 0.5)
            
            pred = 1 if prob >= thresh else 0
            pred_vector.append(pred)
            
        y_preds.append(pred_vector)
    return np.array(y_preds)

def main():
    cache_dir = Path("data/cache")
    prefix = "final_token_aligned_soft"
    layers = [0, 4, 8, 12, 16, 20, 24, 26]
    eval_size = 300
    
    out_dir = Path("runs/layer_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading original natural_dev.jsonl text rows...")
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    
    print("Building dev paired dataset...")
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    
    # Load layer 16 dev cache to align dev_pairs (all layers have the same cache shape)
    ref_dev_cache = torch.load(cache_dir / f"{prefix}_layer16_dev.pt", map_location="cpu")
    if len(dev_pairs) != ref_dev_cache["f_hs"].shape[0]:
        print(f"Warning: Paired dataset size ({len(dev_pairs)}) does not match cache size ({ref_dev_cache['f_hs'].shape[0]}). Aligning dev_pairs...")
        dev_pairs = dev_pairs[:ref_dev_cache["f_hs"].shape[0]]
    del ref_dev_cache
    gc.collect()
    
    # Align sampled items using seed 42 to make sure it's the exact same eval set.
    class_groups = {c: [] for c in ALL_CLASSES}
    for i, pair in enumerate(dev_pairs):
        role = pair["case_role"]
        if not role or role not in CASE_ROLES:
            continue
            
        mapped_dir = ROLE_TO_DIR[role]
        class_groups["None"].append((i, "filled"))
        class_groups[mapped_dir].append((i, "missing"))
        
    num_per_class = max(1, eval_size // 6)
    random.seed(42)
    sampled_items = []
    eval_indices = set()
    
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled = random.sample(idxs, min(len(idxs), num_per_class))
        sampled_items.extend(sampled)
        for idx, _ in sampled:
            eval_indices.add(idx)
            
    print(f"Sampled {len(sampled_items)} items for evaluation.")
    
    all_dev_indices = set(range(len(dev_pairs)))
    cal_indices = list(all_dev_indices - eval_indices)
    print(f"Using {len(cal_indices)} dev items for threshold calibration.")
    
    sweep_results = []
    
    for layer in layers:
        print(f"\n==================================================")
        print(f" Processing Layer {layer} ...")
        print(f"==================================================")
        
        # Load cache for this layer
        train_path = cache_dir / f"{prefix}_layer{layer}_train.pt"
        dev_path = cache_dir / f"{prefix}_layer{layer}_dev.pt"
        
        if not train_path.exists() or not dev_path.exists():
            print(f"Warning: Cache for layer {layer} not found. Skipping.")
            continue
            
        train_cache = torch.load(train_path, map_location="cpu")
        dev_cache = torch.load(dev_path, map_location="cpu")
        
        train_f_hs = train_cache["f_hs"].float().numpy() # [N_train, 7, D]
        train_m_hs = train_cache["m_hs"].float().numpy() # [N_train, 7, D]
        train_y_labels = train_cache["y"].numpy()        # [N_train, 7]
        N_train = train_f_hs.shape[0]
        
        dev_f_hs = dev_cache["f_hs"].float().numpy()     # [N_dev, 7, D]
        dev_m_hs = dev_cache["m_hs"].float().numpy()     # [N_dev, 7, D]
        dev_y = dev_cache["y"].numpy()                   # [N_dev, 7]
        
        # Ground truths
        y_true_multilabel = []
        y_true_dialogue = []
        for idx, cond in sampled_items:
            if cond == "filled":
                y_true_multilabel.append(np.zeros(7))
                y_true_dialogue.append(0) # Sufficient
            else:
                y_true_multilabel.append(dev_y[idx])
                y_true_dialogue.append(1) # Insufficient
        y_true_multilabel = np.array(y_true_multilabel)
        y_true_dialogue = np.array(y_true_dialogue)
        
        # Train linear probes (L1 logistic regression)
        probes = []
        for d in range(7):
            X_f = train_f_hs[:, d, :]
            X_m = train_m_hs[:, d, :]
            X_train = np.concatenate([X_f, X_m], axis=0)
            
            y_f = np.zeros(N_train)
            y_m = train_y_labels[:, d]
            y_train = np.concatenate([y_f, y_m], axis=0)
            
            if len(np.unique(y_train)) <= 1:
                clf = DummyZeroClassifier()
                clf.fit(X_train, y_train)
            else:
                clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=500, random_state=42)
                clf.fit(X_train, y_train)
            probes.append(clf)
            
        # Calibrate thresholds
        thresholds = calibrate_thresholds(probes, dev_f_hs, dev_m_hs, dev_y, cal_indices)
        
        # Evaluate
        y_pred_multilabel = run_evaluation(probes, dev_f_hs, dev_m_hs, sampled_items, thresholds)
        
        # Dialogue-level predictions
        y_pred_dialogue = np.array([1 if np.sum(pred) >= 1 else 0 for pred in y_pred_multilabel])
        
        # Compute metrics
        dialogue_acc = accuracy_score(y_true_dialogue, y_pred_dialogue)
        dialogue_f1 = f1_score(y_true_dialogue, y_pred_dialogue, pos_label=1, zero_division=0)
        slot_macro_f1 = f1_score(y_true_multilabel, y_pred_multilabel, average='macro', zero_division=0)
        slot_micro_f1 = f1_score(y_true_multilabel, y_pred_multilabel, average='micro', zero_division=0)
        hamming_loss_val = hamming_loss(y_true_multilabel, y_pred_multilabel)
        
        print(f"Layer {layer} Results:")
        print(f"  Dialogue-level Accuracy: {dialogue_acc*100:.2f}%")
        print(f"  Dialogue-level F1 (Omission): {dialogue_f1*100:.2f}%")
        print(f"  Slot-level Macro F1: {slot_macro_f1*100:.2f}%")
        print(f"  Slot-level Micro F1: {slot_micro_f1*100:.2f}%")
        print(f"  Hamming Loss: {hamming_loss_val:.4f}")
        
        sweep_results.append({
            "layer": layer,
            "dialogue_accuracy": float(dialogue_acc),
            "dialogue_f1": float(dialogue_f1),
            "slot_macro_f1": float(slot_macro_f1),
            "slot_micro_f1": float(slot_micro_f1),
            "hamming_loss": float(hamming_loss_val)
        })
        
        # Explicitly clean up memory
        del train_cache, dev_cache, train_f_hs, train_m_hs, train_y_labels, dev_f_hs, dev_m_hs, dev_y, probes
        gc.collect()
        
    # Save sweep results to JSON
    with (out_dir / "sweep_results.json").open("w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nAll results saved to runs/layer_sweep/sweep_results.json")
    
    # Generate MD Table summary
    md_file = out_dir / "sweep_results.md"
    with md_file.open("w") as f:
        f.write("# Layer Sweep Probing Results\n\n")
        f.write("Evaluation of linear L1 probes across various transformer layers of Gemma-2-2b-it.\n\n")
        f.write("| Layer | Dialogue Accuracy | Dialogue F1 (Omission) | Slot Macro F1 | Slot Micro F1 | Hamming Loss |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for res in sweep_results:
            f.write(f"| {res['layer']} | {res['dialogue_accuracy']*100:.2f}% | {res['dialogue_f1']*100:.2f}% | {res['slot_macro_f1']*100:.2f}% | {res['slot_micro_f1']*100:.2f}% | {res['hamming_loss']:.4f} |\n")
            
    # Generate plot
    layers_plot = [res["layer"] for res in sweep_results]
    diag_accs = [res["dialogue_accuracy"] for res in sweep_results]
    diag_f1s = [res["dialogue_f1"] for res in sweep_results]
    slot_f1s = [res["slot_macro_f1"] for res in sweep_results]
    
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(layers_plot, diag_accs, marker='o', color='#1565C0', linewidth=2.5, label="Dialogue Accuracy")
    plt.plot(layers_plot, diag_f1s, marker='s', color='#00897B', linewidth=2.0, linestyle="--", label="Dialogue F1 (Omission)")
    plt.plot(layers_plot, slot_f1s, marker='^', color='#D84315', linewidth=2.5, label="Slot Macro F1")
    
    plt.xlabel("Layer Index", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("Probing Performance across Transformer Layers\n(Linear L1 Probe, Gemma-2-2b-it)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.xticks(layers_plot)
    plt.ylim(0.0, 1.05)
    
    plt.tight_layout()
    plot_path = out_dir / "layer_sweep_comparison.png"
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
