#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
import numpy as np
import importlib.util
import warnings
import traceback

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
        res[:, 0] = 1.0
        return res

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

class BottleneckMLP(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=512, bottleneck_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.classifier = nn.Linear(bottleneck_dim, 1)
        
    def forward(self, x):
        z = self.mlp(x)
        logits = self.classifier(z).squeeze(-1)
        return z, logits

def mine_triplets(y_batch):
    y_np = y_batch.cpu().numpy()
    n = len(y_np)
    
    pos_idxs = np.where(y_np == 1)[0]
    neg_idxs = np.where(y_np == 0)[0]
    
    anchors = []
    positives = []
    negatives = []
    
    if len(pos_idxs) < 2 or len(neg_idxs) < 2:
        return None
        
    for i in range(n):
        label = y_np[i]
        if label == 1:
            same_class_pool = pos_idxs[pos_idxs != i]
            diff_class_pool = neg_idxs
        else:
            same_class_pool = neg_idxs[neg_idxs != i]
            diff_class_pool = pos_idxs
            
        if len(same_class_pool) > 0 and len(diff_class_pool) > 0:
            p = random.choice(same_class_pool)
            n_neg = random.choice(diff_class_pool)
            
            anchors.append(i)
            positives.append(p)
            negatives.append(n_neg)
            
    if not anchors:
        return None
        
    return torch.tensor(anchors, dtype=torch.long), torch.tensor(positives, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

def train_bottleneck_mlp(X_train_tensor, y_train_tensor, device, epochs=12, batch_size=256, lr=1e-3, weight_decay=1e-4):
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = BottleneckMLP(input_dim=X_train_tensor.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    bce_loss_fn = nn.BCEWithLogitsLoss()
    cosine_dist = lambda u, v: 1.0 - nn.functional.cosine_similarity(u, v)
    triplet_loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=cosine_dist, margin=0.4)
    
    model.train()
    for epoch in range(1, epochs + 1):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            triplet_indices = mine_triplets(y_batch)
            if triplet_indices is None:
                z, logits = model(x_batch)
                loss = bce_loss_fn(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                continue
                
            anch_idx, pos_idx, neg_idx = triplet_indices
            anch_idx, pos_idx, neg_idx = anch_idx.to(device), pos_idx.to(device), neg_idx.to(device)
            
            z, logits = model(x_batch)
            loss_bce = bce_loss_fn(logits, y_batch)
            
            z_anch = z[anch_idx]
            z_pos = z[pos_idx]
            z_neg = z[neg_idx]
            loss_triplet = triplet_loss_fn(z_anch, z_pos, z_neg)
            
            loss = loss_bce + loss_triplet
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    return model

def calibrate_thresholds(probes, models, dev_f_hs, dev_m_hs, dev_y, cal_indices, device, d_steps=20):
    thresholds = {}
    for d in range(7):
        slot = DIRS[d]
        X_f = torch.tensor(dev_f_hs[cal_indices, d, :], dtype=torch.float32).to(device)
        X_m = torch.tensor(dev_m_hs[cal_indices, d, :], dtype=torch.float32).to(device)
        
        if models is not None and models[d] is not None:
            models[d].eval()
            with torch.no_grad():
                z_f, _ = models[d](X_f)
                z_m, _ = models[d](X_m)
            X_f_z = z_f.cpu().numpy()
            X_m_z = z_m.cpu().numpy()
        else:
            X_f_z = X_f.cpu().numpy()
            X_m_z = X_m.cpu().numpy()
            
        X = np.concatenate([X_f_z, X_m_z], axis=0)
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

def run_evaluation(probes, models, dev_f_hs, dev_m_hs, sampled_items, thresholds, device):
    y_preds = []
    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
        else:
            hs_7 = dev_m_hs[idx]
            
        pred_vector = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            feat_tensor = torch.tensor(feat, dtype=torch.float32).to(device)
            
            if models is not None and models[d] is not None:
                models[d].eval()
                with torch.no_grad():
                    z_feat, _ = models[d](feat_tensor)
                feat_proj = z_feat.cpu().numpy()
            else:
                feat_proj = feat
                
            prob = probes[d].predict_proba(feat_proj)[0, 1]
            slot = DIRS[d]
            thresh = thresholds.get(slot, 0.5)
            
            pred = 1 if prob >= thresh else 0
            pred_vector.append(pred)
            
        y_preds.append(pred_vector)
    return np.array(y_preds)

def main():
    try:
        # Force CPU to isolate CUDA issues
        device = torch.device("cpu")
        print("Using device: CPU", flush=True)
        
        cache_dir = "data/cache"
        prefix = "final_token_aligned_soft"
        layer = 16
        eval_size = 300
        
        print("Loading cached tensors...", flush=True)
        train_cache = torch.load(Path(cache_dir) / f"{prefix}_layer{layer}_train.pt", map_location="cpu")
        dev_cache = torch.load(Path(cache_dir) / f"{prefix}_layer{layer}_dev.pt", map_location="cpu")
        
        print("Loading natural_dev.jsonl...", flush=True)
        dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
        
        dev_ds = PairedDirUncDataset(dev_rows)
        dev_pairs = dev_ds.pairs[:dev_cache["f_hs"].shape[0]]
        
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
                
        all_dev_indices = set(range(len(dev_pairs)))
        cal_indices = list(all_dev_indices - eval_indices)
        
        train_f_hs = train_cache["f_hs"].float().numpy()
        train_m_hs = train_cache["m_hs"].float().numpy()
        train_y_labels = train_cache["y"].numpy()
        N_train = train_f_hs.shape[0]
        
        # Subsample globally to speed up debugging
        np.random.seed(42)
        sub_indices = np.random.choice(N_train, size=1500, replace=False)
        print(f"Subsampled train set to {len(sub_indices)} pairs.", flush=True)
        
        train_f_hs = train_f_hs[sub_indices]
        train_m_hs = train_m_hs[sub_indices]
        train_y_labels = train_y_labels[sub_indices]
        N_train = len(sub_indices)
        
        dev_f_hs = dev_cache["f_hs"].float().numpy()
        dev_m_hs = dev_cache["m_hs"].float().numpy()
        dev_y = dev_cache["y"].numpy()
        
        y_true_multilabel = []
        for idx, cond in sampled_items:
            if cond == "filled":
                y_true_multilabel.append(np.zeros(7))
            else:
                y_true_multilabel.append(dev_y[idx])
        y_true_multilabel = np.array(y_true_multilabel)
        
        # -----------------------------------------------------------------------
        # Setup A: Baseline (Raw Space)
        # -----------------------------------------------------------------------
        print("\nTraining Setup A: Baseline...", flush=True)
        probes_base = []
        for d in range(7):
            X_f = train_f_hs[:, d, :]
            X_m = train_m_hs[:, d, :]
            X_train_np = np.concatenate([X_f, X_m], axis=0)
            
            y_f = np.zeros(N_train)
            y_m = train_y_labels[:, d]
            y_train_np = np.concatenate([y_f, y_m], axis=0)
            
            if len(np.unique(y_train_np)) <= 1:
                clf = DummyZeroClassifier()
                clf.fit(X_train_np, y_train_np)
            else:
                clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=150, random_state=42)
                clf.fit(X_train_np, y_train_np)
            probes_base.append(clf)
            
        print("Calibrating thresholds for Baseline...", flush=True)
        thresh_base = calibrate_thresholds(probes_base, None, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
        y_pred_base = run_evaluation(probes_base, None, dev_f_hs, dev_m_hs, sampled_items, thresh_base, device)
        
        # -----------------------------------------------------------------------
        # Setup B: Bottleneck MLP
        # -----------------------------------------------------------------------
        print("\nTraining Setup B: MLP Bottleneck...", flush=True)
        models_bt = []
        probes_bt = []
        for d in range(7):
            slot = DIRS[d]
            print(f"Fitting MLP for slot: {slot}...", flush=True)
            X_f = train_f_hs[:, d, :]
            X_m = train_m_hs[:, d, :]
            X_train_np = np.concatenate([X_f, X_m], axis=0)
            
            y_f = np.zeros(N_train)
            y_m = train_y_labels[:, d]
            y_train_np = np.concatenate([y_f, y_m], axis=0)
            
            if len(np.unique(y_train_np)) <= 1:
                models_bt.append(None)
                clf = DummyZeroClassifier()
                clf.fit(X_train_np, y_train_np)
                probes_bt.append(clf)
                continue
                
            X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
            
            mlp_model = train_bottleneck_mlp(X_train_tensor, y_train_tensor, device, epochs=12, batch_size=256, lr=1e-3)
            models_bt.append(mlp_model)
            
            mlp_model.eval()
            with torch.no_grad():
                z_train, _ = mlp_model(X_train_tensor.to(device))
            X_train_z = z_train.cpu().numpy()
            
            clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, tol=1e-2, max_iter=150, random_state=42)
            clf.fit(X_train_z, y_train_np)
            probes_bt.append(clf)
            
        print("Calibrating thresholds for MLP...", flush=True)
        thresh_bt = calibrate_thresholds(probes_bt, models_bt, dev_f_hs, dev_m_hs, dev_y, cal_indices, device)
        y_pred_bt = run_evaluation(probes_bt, models_bt, dev_f_hs, dev_m_hs, sampled_items, thresh_bt, device)
        
        # Metrics
        macro_f1_base = f1_score(y_true_multilabel, y_pred_base, average='macro', zero_division=0)
        subset_acc_base = accuracy_score(y_true_multilabel, y_pred_base)
        
        macro_f1_bt = f1_score(y_true_multilabel, y_pred_bt, average='macro', zero_division=0)
        subset_acc_bt = accuracy_score(y_true_multilabel, y_pred_bt)
        
        y_true_dialogue = (y_true_multilabel.sum(axis=1) > 0).astype(int)
        y_pred_dialogue_base = (y_pred_base.sum(axis=1) > 0).astype(int)
        dialogue_acc_base = accuracy_score(y_true_dialogue, y_pred_dialogue_base)
        dialogue_f1_base = f1_score(y_true_dialogue, y_pred_dialogue_base, zero_division=0)
        
        y_pred_dialogue_bt = (y_pred_bt.sum(axis=1) > 0).astype(int)
        dialogue_acc_bt = accuracy_score(y_true_dialogue, y_pred_dialogue_bt)
        dialogue_f1_bt = f1_score(y_true_dialogue, y_pred_dialogue_bt, zero_division=0)
        
        print("\n" + "="*50, flush=True)
        print(" COMPREHENSIVE DEBUG EVALUATION RESULTS ", flush=True)
        print("="*50, flush=True)
        print(f"Baseline Macro F1:\t{macro_f1_base*100:.2f}%", flush=True)
        print(f"Baseline Dialogue Acc:\t{dialogue_acc_base*100:.2f}%", flush=True)
        print(f"Baseline Dialogue F1:\t{dialogue_f1_base*100:.2f}%", flush=True)
        print("-"*50, flush=True)
        print(f"Approach B Macro F1:\t{macro_f1_bt*100:.2f}%", flush=True)
        print(f"Approach B Dialogue Acc:\t{dialogue_acc_bt*100:.2f}%", flush=True)
        print(f"Approach B Dialogue F1:\t{dialogue_f1_bt*100:.2f}%", flush=True)
        print("="*50, flush=True)
        
    except Exception as e:
        print(f"\nCRITICAL EXCEPTION ENCOUNTERED:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
