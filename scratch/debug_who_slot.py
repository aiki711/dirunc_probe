#!/usr/bin/env python3
"""Debug why who-slot achieves 100% precision and recall."""
import os, sys, json, random, warnings
import torch, numpy as np
warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import importlib.util

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]
ROLE_TO_DIR = {
    "Agent": "who", "Theme": "what", "Location": "where",
    "Source": "where", "Goal": "where", "Time": "when", "Manner": "how"
}

def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s32 = load_script_32()
PairedDirUncDataset = s32.PairedDirUncDataset

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

layer = 26
prefix = "final_token_aligned_soft"
cache_dir = Path("data/cache")
eval_size = 300

dev_rows = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
dev_ds = PairedDirUncDataset(dev_rows)
dev_pairs = dev_ds.pairs

dev_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_dev.pt", map_location="cpu")
dev_y = dev_cache["y"].numpy()
dev_f_hs = dev_cache["f_hs"].float().numpy()
dev_m_hs = dev_cache["m_hs"].float().numpy()

if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
    dev_pairs = dev_pairs[:dev_cache["f_hs"].shape[0]]

class_groups = {c: [] for c in ALL_CLASSES}
for i, pair in enumerate(dev_pairs):
    role = pair["case_role"]
    if not role or role not in CASE_ROLES:
        continue
    mapped_dir = ROLE_TO_DIR[role]
    class_groups["None"].append((i, "filled"))
    class_groups[mapped_dir].append((i, "missing"))

# ---- how many Agent samples exist in dev? ----
print(f"\n=== Class group sizes ===")
for c in ALL_CLASSES:
    print(f"  {c:10s}: {len(class_groups[c])} samples")

num_per_class = max(1, eval_size // 6)
random.seed(42)
sampled_items = []
for c in ALL_CLASSES:
    idxs = class_groups[c]
    sampled = random.sample(idxs, min(len(idxs), num_per_class))
    sampled_items.extend(sampled)

# ---- Build y_true_binary ----
y_true_binary = []
for idx, cond in sampled_items:
    if cond == "filled":
        y_true_binary.append(np.zeros(7))
    else:
        y_true_binary.append(dev_y[idx])
y_true_binary = np.array(y_true_binary)

# ---- Fit who probe ----
train_cache = torch.load(cache_dir / f"{prefix}_layer{layer}_train.pt", map_location="cpu")
train_f_hs = train_cache["f_hs"].float().numpy()
train_m_hs = train_cache["m_hs"].float().numpy()
train_y_labels = train_cache["y"].numpy()
N_train = train_f_hs.shape[0]

d_who = DIRS.index("who")  # should be 0
X_f = train_f_hs[:, d_who, :]
X_m = train_m_hs[:, d_who, :]
X_train = np.concatenate([X_f, X_m], axis=0)
y_f = np.zeros(N_train)
y_m = train_y_labels[:, d_who]
y_train = np.concatenate([y_f, y_m], axis=0)

print(f"\n=== Training who-probe ===")
print(f"  Positive (missing=Agent) in train: {int(y_m.sum())} / {len(y_m)}")
print(f"  Total train samples (f+m): {len(y_train)}, positives: {int(y_train.sum())}")

clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
clf.fit(X_train, y_train)

# ---- Check probability distribution on eval set ----
THRESHOLDS = {"who": 0.47, "what": 0.27, "when": 0.37, "where": 0.32, "why": 0.02, "how": 0.37, "which": 0.47}
thresh = THRESHOLDS["who"]

y_true_who = y_true_binary[:, d_who]
probs_who = []
for idx, cond in sampled_items:
    hs = dev_f_hs[idx] if cond == "filled" else dev_m_hs[idx]
    feat = hs[d_who].reshape(1, -1)
    prob = clf.predict_proba(feat)[0, 1]
    probs_who.append(prob)
probs_who = np.array(probs_who)
y_pred_who = (probs_who >= thresh).astype(int)

print(f"\n=== Eval set who-slot breakdown ===")
print(f"  y_true positives (Agent missing): {int(y_true_who.sum())}")
print(f"  y_true negatives (non-Agent):     {int((1 - y_true_who).sum())}")
print(f"  Threshold used: {thresh}")
print(f"  Predicted positives: {int(y_pred_who.sum())}")
print(f"  Predicted negatives: {int((1 - y_pred_who).sum())}")
print(f"  Prob range: min={probs_who.min():.4f}, max={probs_who.max():.4f}, mean={probs_who.mean():.4f}")
print(f"  Prob among true positives: min={probs_who[y_true_who==1].min():.4f}, max={probs_who[y_true_who==1].max():.4f}")
print(f"  Prob among true negatives: min={probs_who[y_true_who==0].min():.4f}, max={probs_who[y_true_who==0].max():.4f}")

cm = confusion_matrix(y_true_who, y_pred_who)
print(f"\n  Confusion Matrix (rows=true, cols=pred):")
print(f"             Pred=0   Pred=1")
print(f"  True=0:   {cm[0,0]:6d}   {cm[0,1]:6d}  (TN / FP)")
print(f"  True=1:   {cm[1,0]:6d}   {cm[1,1]:6d}  (FN / TP)")

# ---- Check if train and dev indices overlap (data leakage check) ----
# Also check if the 'filled' samples for who are truly non-Agent
print(f"\n=== Sanity check on sampled Agent items ===")
agent_items = [(idx, cond) for idx, cond in sampled_items if cond == "missing" and ROLE_TO_DIR[dev_pairs[idx]["case_role"]] == "who"]
print(f"  Sampled Agent-missing items: {len(agent_items)}")
for idx, cond in agent_items[:5]:
    pair = dev_pairs[idx]
    print(f"  idx={idx}, role={pair['case_role']}, y_true_who={dev_y[idx][d_who]}")
