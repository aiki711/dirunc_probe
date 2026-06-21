import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import random
import sys
import os
import json
import importlib.util

sys.path.append(os.getcwd())
from scripts.common import DIRS

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]

ROLE_TO_DIR = {
    "Agent": "who", "Theme": "what", "Location": "where", "Source": "where",
    "Goal": "where", "Time": "when", "Manner": "how"
}

class DummyZeroClassifier:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0])
    def predict_proba(self, X):
        res = np.zeros((X.shape[0], 2))
        res[:, 0] = 1.0
        return res

def load_script_32():
    import sys
    import os
    import importlib.util
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    scripts_path = os.path.join(os.getcwd(), "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    cache_dir = Path("data/cache")
    prefix = "final_token_aligned_soft"
    layer = 16
    eval_size = 300
    
    train_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_train.pt", map_location="cpu")
    dev_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_dev.pt", map_location="cpu")
    
    # Sample evaluation set (same seed 42)
    dev_y = dev_cache["y"].numpy()
    N_dev = dev_y.shape[0]
    
    # We load natural_dev.jsonl just to get dataset structure and align indices
    # We can reconstruct sampling indices exactly as in scripts/55
    s32 = load_script_32()
    PairedDirUncDataset = s32.PairedDirUncDataset
    
    def read_jsonl(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
            
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs[:N_dev]
    
    class_groups = {c: [] for c in ALL_CLASSES}
    for i, pair in enumerate(dev_pairs):
        role = pair["case_role"]
        if not role or role not in CASE_ROLES: continue
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
            
    all_dev_indices = set(range(len(dev_pairs)))
    cal_indices = list(all_dev_indices - eval_indices)
    
    train_f_hs = train_cache["f_hs"].float().numpy()
    train_m_hs = train_cache["m_hs"].float().numpy()
    train_y = train_cache["y"].numpy()
    N_train = train_f_hs.shape[0]
    
    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()
    
    # Reconstruct ground truths
    y_true_multilabel = []
    y_true_suff = [] # 1 for Insufficient (missing), 0 for Sufficient (filled)
    for idx, cond in sampled_items:
        if cond == "filled":
            y_true_multilabel.append(np.zeros(7))
            y_true_suff.append(0)
        else:
            y_true_multilabel.append(dev_y[idx])
            y_true_suff.append(1)
    y_true_multilabel = np.array(y_true_multilabel)
    y_true_suff = np.array(y_true_suff)
    
    # Train Baseline Probes
    probes_base = []
    for d in range(7):
        X_f = train_f_hs[:, d, :]
        X_m = train_m_hs[:, d, :]
        X_train = np.concatenate([X_f, X_m], axis=0)
        y_f = np.zeros(N_train)
        y_m = train_y[:, d]
        y_train = np.concatenate([y_f, y_m], axis=0)
        
        if len(np.unique(y_train)) <= 1:
            clf = DummyZeroClassifier()
            clf.fit(X_train, y_train)
        else:
            clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=500, random_state=42)
            clf.fit(X_train, y_train)
        probes_base.append(clf)
        
    # Calibrate thresholds
    import importlib.util
    spec_55 = importlib.util.spec_from_file_location("script_55", "scripts/55_compare_multilabel_probing.py")
    s55 = importlib.util.module_from_spec(spec_55)
    spec_55.loader.exec_module(s55)
    calibrate_thresholds = s55.calibrate_thresholds
    run_evaluation = s55.run_evaluation
    # For calibrate_thresholds, we need a device. We will use cpu since it's just numpy/sklearn
    device = torch.device("cpu")
    thresholds = calibrate_thresholds(probes_base, None, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
    
    # Predict multi-label vectors
    y_pred_multilabel = run_evaluation(probes_base, None, dev_f_hs, dev_m_hs, sampled_items, thresholds, device)
    
    # 1. Analyse Verify (Binary Sufficiency) Phase
    # Predict "Insufficient (1)" if any of the 7 slots is predicted as missing (1)
    y_pred_suff = (y_pred_multilabel.sum(axis=1) >= 1).astype(int)
    
    acc_suff = accuracy_score(y_true_suff, y_pred_suff)
    f1_suff = f1_score(y_true_suff, y_pred_suff, pos_label=1)
    cm = confusion_matrix(y_true_suff, y_pred_suff)
    
    print("\n" + "="*50)
    print(" BASELINE INTEGRATED EVALUATION (VERIFY + IDENTIFY) ")
    print("="*50)
    print("\n--- [Phase 1] Verify (Binary Sufficiency) Performance ---")
    print(f"Accuracy: {acc_suff*100:.2f}%")
    print(f"Omission F1-Score: {f1_suff*100:.2f}%")
    print("Confusion Matrix (0: Sufficient, 1: Insufficient):")
    print(f"  True Sufficient (Filled)   matched as:  Sufficient={cm[0,0]} | Insufficient={cm[0,1]}")
    print(f"  True Insufficient (Missing) matched as: Sufficient={cm[1,0]} | Insufficient={cm[1,1]}")
    
    # 2. Analyse Identify (Multi-label per-slot) Performance
    print("\n--- [Phase 2] Identify (Per-Slot Omission) Performance ---")
    print(f"Slot Index\tSlot Name\tPrecision\tRecall\t\tF1-Score")
    print("-"*70)
    for d in range(7):
        slot = DIRS[d]
        y_true_slot = y_true_multilabel[:, d]
        y_pred_slot = y_pred_multilabel[:, d]
        
        # Calculate scores
        # Precision
        p_num = np.sum((y_pred_slot == 1) & (y_true_slot == 1))
        p_denom = np.sum(y_pred_slot == 1)
        precision = p_num / p_denom if p_denom > 0 else 0.0
        
        # Recall
        r_denom = np.sum(y_true_slot == 1)
        recall = p_num / r_denom if r_denom > 0 else 0.0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Slot {d}\t\t{slot:8s}\t{precision*100:6.2f}%\t\t{recall*100:6.2f}%\t\t{f1*100:6.2f}%")
        
if __name__ == "__main__":
    main()
