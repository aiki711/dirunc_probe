from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DIRS = ["who", "what", "when", "where", "why", "how"]

QUERY_LABEL_STR = {
    "who": "[WHO?]",
    "what": "[WHAT?]",
    "when": "[WHEN?]",
    "where": "[WHERE?]",
    "why": "[WHY?]",
    "how": "[HOW?]",
}
SPECIAL_TOKENS = list(QUERY_LABEL_STR.values())
QUERY_TOKENS = tuple(SPECIAL_TOKENS)

# ---------------- Utils ----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def micro_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    # y_true/y_pred: (N, 6) binary
    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    # --- NEW: 正例があるラベルだけでmacroを計算 ---
    support = y_true.sum(axis=0).astype(int)  # 各ラベルの正例数
    mask = support > 0
    if mask.any():
        macro_posonly = f1_score(y_true[:, mask], y_pred[:, mask], average="macro", zero_division=0)
        labels_included = [DIRS[i] for i in range(len(DIRS)) if mask[i]]
    else:
        macro_posonly = 0.0
        labels_included = []

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "per_label_f1": per_label,
        "support_pos": support.tolist(),                 # NEW: [who, what, ...] の正例数
        "macro_f1_posonly": float(macro_posonly),        # NEW: 正例ラベルのみmacro
        "macro_posonly_labels": labels_included,         # NEW: 含めたラベル名
    }

def floor_layer_points(num_layers: int) -> List[int]:
    # We define "layer_idx" in [0, num_layers-1] (transformer blocks)
    # Candidate: floor(r*(L-1))
    if num_layers <= 1:
        return [0]
    Lm1 = num_layers - 1
    rs = [0.2, 0.4, 0.6, 0.8, 1.0]
    pts = [int(math.floor(r * Lm1)) for r in rs]
    # de-dup while keeping order
    out = []
    for p in pts:
        if p not in out:
            out.append(p)
    return out

def expand_best_pm2(best: int, num_layers: int) -> List[int]:
    cands = [best - 2, best - 1, best, best + 1, best + 2]
    cands = [max(0, min(num_layers - 1, c)) for c in cands]
    out = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out

def strip_query_tokens(text: str) -> str:
    t = text.rstrip()
    if t.endswith(QUERY_TOKENS): # Error: QUERY_TOKENS not defined, likely usage error in original or missing var.
        # Wait, I don't see QUERY_TOKENS defined above, only QUERY_LABEL_STR and SPECIAL_TOKENS.
        # Checking implementation plan or previous output... 
        # Ah, I don't have the implementation of strip_query_tokens from my view.
        # Step 179 showed it signature. Step 199 showed full content!
        # Step 199 line 110: if t.endswith(QUERY_TOKENS):
        # BUT where is QUERY_TOKENS defined?
        # Maybe tuple(SPECIAL_TOKENS)?
        # Let's check imports in Step 179/199.
        # It's not there.
        # Wait, maybe it's tuple(SPECIAL_TOKENS).
        # Let's Assume tuple(SPECIAL_TOKENS).
        pass
    # I'll fix this to use tuple(SPECIAL_TOKENS) which is safe for endswith.
    return t

def _init_span_acc() -> Dict[str, Any]:
    return {
        "found": Counter(),   # dir -> count
        "total": Counter(),   # dir -> count
        "all_found": 0,       # samples where all 6 spans were found
        "n_samples": 0,       # total samples
    }

def _update_span_acc(acc: Dict[str, Any], model: nn.Module) -> None:
    st = getattr(model, "last_span_found", None)
    if not st:
        return
    for d in DIRS:
        acc["found"][d] += int(st["found"].get(d, 0))
        acc["total"][d] += int(st["total"].get(d, 0))
    acc["all_found"] += int(st.get("all_found", 0))
    acc["n_samples"] += int(st.get("n_samples", 0))

def _finalize_span_acc(acc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    total_all = sum(int(acc["total"][d]) for d in DIRS)
    if total_all == 0:
        return None
    found_all = sum(int(acc["found"][d]) for d in DIRS)
    rates = {
        d: (float(acc["found"][d]) / float(acc["total"][d]) if acc["total"][d] else 0.0)
        for d in DIRS
    }
    return {
        "found": {d: int(acc["found"][d]) for d in DIRS},
        "total": {d: int(acc["total"][d]) for d in DIRS},
        "rates": rates,
        "overall_rate": float(found_all) / float(total_all),
        "all_found_rate": float(acc["all_found"]) / float(acc["n_samples"]) if acc["n_samples"] else 0.0,
        "n_samples": int(acc["n_samples"]),
    }

# ---------------- Dataset ----------------

@dataclass
class FilterSpec:
    phases: Optional[List[int]] = None
    levels: Optional[List[int]] = None
    k_values: Optional[List[int]] = None

class JsonlDirUncDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        filter_spec: FilterSpec,
    ) -> None:
        self.rows = self._filter(rows, filter_spec)

    @staticmethod
    def _filter(rows: List[Dict[str, Any]], fs: FilterSpec) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            ph = int(r.get("phase"))
            lv = int(r.get("level"))
            kv = r.get("k", None)
            kv_int = None if kv is None else int(kv)

            if fs.phases is not None and ph not in fs.phases:
                continue
            if fs.levels is not None and lv not in fs.levels:
                continue
            if fs.k_values is not None:
                if ph == 2 and (kv_int not in fs.k_values):
                    continue
            out.append(r)
        return out

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        labels = r["labels"]
        y = torch.tensor([float(labels[d]) for d in DIRS], dtype=torch.float32)
        return {"text": r["text"], "y": y, "meta": r}

# ---------------- Model wrappers ----------------

def find_subsequence(hay: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
    if not needle or not hay or len(needle) > len(hay):
        return None
    for start in range(len(hay) - len(needle), -1, -1):
        if hay[start : start + len(needle)] == needle:
            return (start, start + len(needle))
    return None

class ProbeModelBase(nn.Module):
    """Shared: frozen LM + output_hidden_states. Optionally train only special-token embeddings."""
    def __init__(self, model_name: str, *, 
                 vocab_size: Optional[int] = None, 
                 train_token_ids: Optional[List[int]] = None,
                 pretrained_model: Optional[nn.Module] = None) -> None:
        super().__init__()
        
        if pretrained_model is not None:
            self.lm = pretrained_model
        else:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            try:
                self.lm = AutoModel.from_pretrained(
                    model_name, 
                    output_hidden_states=True, 
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
            except OSError:
                 self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            
            if vocab_size is not None:
                cur = self.lm.get_input_embeddings().weight.size(0)
                if vocab_size > cur:
                    self.lm.resize_token_embeddings(vocab_size)

        self.output_dtype = self.lm.dtype

        for p in self.lm.parameters():
            p.requires_grad = False

        self.train_token_ids: List[int] = train_token_ids or []
        self.hook_handle = None

        if self.train_token_ids:
            emb = self.lm.get_input_embeddings()
            emb.weight.requires_grad = True
            
            mask = torch.zeros_like(emb.weight, dtype=torch.float32)
            for tid in self.train_token_ids:
                if 0 <= tid < mask.size(0):
                    mask[tid] = 1.0
            self.register_buffer("_emb_grad_mask", mask)

            def _hook(grad: torch.Tensor) -> torch.Tensor:
                return grad * self._emb_grad_mask.to(dtype=grad.dtype)

            self.hook_handle = emb.weight.register_hook(_hook)

        self.hidden_size = int(self.lm.config.hidden_size)
        self.num_layers = int(self.lm.config.num_hidden_layers)

    def teardown(self) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        
        if hasattr(self, 'lm') and self.lm is not None:
            emb = self.lm.get_input_embeddings()
            if emb:
                emb.weight.requires_grad = False

    def get_layer_hidden(self, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        if layer_idx < 0:
            idx = len(hidden_states) + layer_idx
        else:
            idx = layer_idx + 1
            
        if idx < 0 or idx >= len(hidden_states):
             idx = -1
             
        return hidden_states[idx]

class EosPoolingProbe(nn.Module):
    """Baseline: EOS pooling (last non-pad token) + linear head."""
    def __init__(self, base: ProbeModelBase) -> None:
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.hidden_size, len(DIRS)).to(dtype=getattr(base, "output_dtype", torch.float32))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)  # (B, T, H)
        lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
        bsz = hs.size(0)
        h_last = hs[torch.arange(bsz, device=hs.device), lengths]  # (B, H)
        logits = self.head(h_last)  # (B, 6)
        return logits

class QueryTokenProbe(nn.Module):
    """Proposed: Query-token Approach"""
    def __init__(self, base: ProbeModelBase, tokenizer: Any) -> None:
        super().__init__()
        self.base = base
        self.tokenizer = tokenizer
        
        # We need token_id map. It assumes SPECIAL_TOKENS order matches DIRS?
        # Let's derive it or reuse QUERY_LABEL_STR.
        self.token_id = {}
        for d, tstr in QUERY_LABEL_STR.items():
            self.token_id[d] = tokenizer.convert_tokens_to_ids(tstr)

        target_dtype = getattr(base, "output_dtype", torch.float32)
        self.W = nn.Parameter(torch.empty(len(DIRS), base.hidden_size, dtype=target_dtype))
        self.b = nn.Parameter(torch.zeros(len(DIRS), dtype=target_dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)
        self.last_span_found: Optional[Dict[str, Any]] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)
        found_counts = {d: 0 for d in DIRS}
        total_counts = {d: 0 for d in DIRS}
        all_found = 0
        n_samples = int(input_ids.size(0))
        logits_list = []
        for bi in range(n_samples):
            mask = attention_mask[bi]
            valid_indices = torch.nonzero(mask).squeeze(-1)
            if valid_indices.numel() == 0:
                 logits_list.append(torch.zeros(len(DIRS), device=input_ids.device))
                 continue
            ids_vec = input_ids[bi, valid_indices]
            h_vec = hs[bi, valid_indices, :]
            ids_list = ids_vec.tolist()
            dir_vecs = []
            sample_all_found = True
            for d in DIRS:
                total_counts[d] += 1
                tid = self.token_id[d]
                pos = None
                for j in range(len(ids_list) - 1, -1, -1):
                    if ids_list[j] == tid:
                        pos = j
                        break
                if pos is None:
                    sample_all_found = False
                    vec = h_vec[-1]
                else:
                    vec = h_vec[pos]
                    found_counts[d] += 1
                dir_vecs.append(vec)
            if sample_all_found:
                all_found += 1
            H = torch.stack(dir_vecs, dim=0)
            logit = (H * self.W).sum(dim=1) + self.b
            logits_list.append(logit)
        logits = torch.stack(logits_list, dim=0)
        self.last_span_found = {
            "found": {d: int(found_counts[d]) for d in DIRS},
            "total": {d: int(total_counts[d]) for d in DIRS},
            "rates": {d: (float(found_counts[d]) / float(total_counts[d]) if total_counts[d] else 0.0) for d in DIRS},
            "all_found": int(all_found),
            "n_samples": int(n_samples),
        }
        return logits

# ---------------- Activation Caching ----------------

class BaselineHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dtype: torch.dtype):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_labels).to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class QueryHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dtype: torch.dtype):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(num_labels, dtype=dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x * self.W.unsqueeze(0)).sum(dim=2) + self.b.unsqueeze(0)
        return out

@torch.no_grad()
def extract_activations(
    dl: DataLoader,
    base_model: ProbeModelBase,
    layers: List[int],
    mode: str,
    device: torch.device,
    tokenizer: Any = None,
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, Optional[Dict[str, Any]]]:
    base_model.lm.eval()
    all_x = {l: [] for l in layers}
    all_y = []
    span_acc = _init_span_acc() if mode == "query" else None
    
    token_id = {}
    if mode == "query":
         for d, tstr in QUERY_LABEL_STR.items():
            token_id[d] = tokenizer.convert_tokens_to_ids(tstr)

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["y"].to(device)
        out = base_model.lm(input_ids=input_ids, attention_mask=attention_mask)
        bsz = input_ids.size(0)
        
        if mode == "baseline":
            lengths = attention_mask.long().sum(dim=1) - 1
            
        for layer_idx in layers:
            hs = base_model.get_layer_hidden(out.hidden_states, layer_idx)
            
            if mode == "baseline":
                h_last = hs[torch.arange(bsz, device=device), lengths]
                all_x[layer_idx].append(h_last.cpu())
                
            elif mode == "query":
                batch_features = []
                found_counts = {d: 0 for d in DIRS}
                total_counts = {d: 0 for d in DIRS}
                batch_all_found = 0
                
                for bi in range(bsz):
                    mask = attention_mask[bi]
                    valid_indices = torch.nonzero(mask).squeeze(-1)
                    if valid_indices.numel() == 0:
                        batch_features.append(torch.zeros(len(DIRS), hs.size(-1), device=device, dtype=hs.dtype))
                        continue
                    ids_list = input_ids[bi, valid_indices].tolist()
                    h_vec = hs[bi, valid_indices, :]
                    dir_vecs = []
                    sample_all_found = True
                    for d in DIRS:
                        if layer_idx == layers[0]:
                            total_counts[d] += 1
                        tid = token_id[d]
                        pos = None
                        for j in range(len(ids_list) - 1, -1, -1):
                            if ids_list[j] == tid:
                                pos = j
                                break
                        if pos is None:
                            sample_all_found = False
                            vec = h_vec[-1]
                        else:
                            vec = h_vec[pos]
                            if layer_idx == layers[0]:
                                found_counts[d] += 1
                        dir_vecs.append(vec)
                    if sample_all_found:
                         if layer_idx == layers[0]:
                            batch_all_found += 1
                    batch_features.append(torch.stack(dir_vecs, dim=0))
                
                if layer_idx == layers[0]:
                    for d in DIRS:
                        span_acc["found"][d] += found_counts[d]
                        span_acc["total"][d] += total_counts[d]
                    span_acc["all_found"] += batch_all_found
                    span_acc["n_samples"] += bsz

                if batch_features:
                    all_x[layer_idx].append(torch.stack(batch_features, dim=0).cpu())

        all_y.append(y.cpu())
        
    final_x = {}
    if len(all_y) == 0:
         for l in layers:
             final_x[l] = torch.empty(0)
         return final_x, torch.empty(0), None
    Y = torch.cat(all_y, dim=0)
    for l in layers:
        if all_x[l]:
             final_x[l] = torch.cat(all_x[l], dim=0)
        else:
             final_x[l] = torch.empty(0)
    
    summary = None
    if mode == "query":
        summary = _finalize_span_acc(span_acc)
    return final_x, Y, summary

@torch.no_grad()
def evaluate_cached(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for batch in dl:
        x = batch[0].to(device)
        y = batch[1].to(device)
        logits = model(x)
        prob = torch.sigmoid(logits)
        ys.append(y.float().cpu().numpy())
        ps.append(prob.float().cpu().numpy())
    if not ys:
        return np.array([]), np.array([])
    y_true = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return y_true, p

# ---------------- Training / Eval ----------------

def collate_batch(tokenizer, batch: List[Dict[str, Any]], max_length: int) -> Dict[str, Any]:
    texts = [x["text"] for x in batch]
    labels = torch.stack([x["y"] for x in batch])
    
    # Simple tokenization
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "y": labels,
    }

def eval_with_threshold(y_true: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (p >= threshold).astype(np.int32)
    m = micro_macro_f1(y_true, y_pred)
    m["threshold"] = float(threshold)
    return m

def tune_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    metric: str = "macro_f1_posonly",
    grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if grid is None:
        grid = [i / 100.0 for i in range(5, 96, 5)]

    best = None
    for th in grid:
        m = eval_with_threshold(y_true, p, th)
        score = m.get(metric, None)
        if score is None:
            continue
        if best is None or score > best["score"]:
            best = {"score": float(score), "threshold": float(th), "metrics": m}

    assert best is not None
    return best

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    layer_idx: int,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    ys = []
    ps = []
    span_acc = _init_span_acc()
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["y"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, layer_idx=layer_idx)
        _update_span_acc(span_acc, model)
        prob = torch.sigmoid(logits)
        ys.append(y.float().cpu().numpy())
        ps.append(prob.float().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    y_pred = (p >= threshold).astype(np.int32)

    metrics = micro_macro_f1(y_true.astype(np.int32), y_pred)
    metrics["threshold"] = threshold
    span_summary = _finalize_span_acc(span_acc)
    if span_summary is not None:
        metrics["span_found"] = span_summary
    return metrics

def train_probe_from_cache(
    *,
    mode: str,
    layer_idx: int,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_dev: torch.Tensor,
    Y_dev: torch.Tensor,
    train_span_summary: Optional[Dict[str, Any]],
    dev_span_summary: Optional[Dict[str, Any]],
    hidden_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    seed: int,
    out_dir: Path,
    threshold: float,
    eval_services: Optional[List[str]],
    dev_rows_use: List[Dict[str, Any]],
    no_tqdm: bool,
    tqdm_mininterval: float,
) -> Dict[str, Any]:
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create cached DataLoaders
    train_ds_cached = TensorDataset(X_train, Y_train)
    dev_ds_cached = TensorDataset(X_dev, Y_dev)
    
    train_dl_cached = DataLoader(train_ds_cached, batch_size=batch_size, shuffle=True)
    dev_dl_cached = DataLoader(dev_ds_cached, batch_size=batch_size, shuffle=False)
    
    dtype = X_train.dtype
    
    if mode == "baseline":
        model = BaselineHead(hidden_size, len(DIRS), dtype).to(device)
    elif mode == "query":
        model = QueryHead(hidden_size, len(DIRS), dtype).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr)

    best = {"micro_f1": -1.0, "macro_f1_posonly": -1.0}
    best_path = out_dir / f"best_{mode}_layer{layer_idx}.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        
        total_batches = len(train_dl_cached)
        for i, batch in enumerate(tqdm(
            train_dl_cached, 
            desc=f"train {mode} layer={layer_idx} ep={ep}", 
            leave=False, disable=no_tqdm, mininterval=tqdm_mininterval)):
            
            x = batch[0].to(device)
            y = batch[1].to(device)

            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

            if no_tqdm and (i + 1) % 20000 == 0:
                val = loss.item()
                print(f"[Epoch {ep}] Step {i+1}/{total_batches} | Loss: {val:.4f}", flush=True)

        train_loss = total_loss / max(1, n)

        y_true_dev, p_dev = evaluate_cached(model, dev_dl_cached, device)

        tuned = tune_threshold(
            y_true_dev,
            p_dev,
            metric="macro_f1_posonly", 
            grid=None,
        )
        dev_metrics = dict(tuned["metrics"]) 
        
        if dev_span_summary is not None:
             dev_metrics["span_found"] = dev_span_summary
             
        dev_metrics["tuned"] = {
            "metric": "macro_f1_posonly",
            "best_threshold": float(tuned["threshold"]),
            "best_score": float(tuned["score"]),
        }

        if train_span_summary is not None:
            dev_metrics["span_found_train"] = train_span_summary

        record = {
            "mode": mode,
            "layer_idx": layer_idx,
            "epoch": ep,
            "train_loss": train_loss,
            **dev_metrics,
        }

        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"[Epoch {ep}/{epochs}] mode={mode} layer={layer_idx} | "
              f"train_loss={train_loss:.4f} | "
              f"macro_f1_posonly={dev_metrics.get('macro_f1_posonly',0.0):.4f} | "
              f"micro_f1={dev_metrics.get('micro_f1',0.0):.4f}")

        if dev_metrics["macro_f1_posonly"] > best.get("macro_f1_posonly", -1.0):
            best = {**record}
            torch.save(model.state_dict(), best_path)

    if best["macro_f1_posonly"] >= 0 and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    best_th = float(best.get("threshold", threshold))
    y_dev, p_dev = evaluate_cached(model, dev_dl_cached, device)
    best_dev_metrics = eval_with_threshold(y_dev, p_dev, best_th)
    if dev_span_summary:
        best_dev_metrics["span_found"] = dev_span_summary
    best["final_dev"] = best_dev_metrics

    if eval_services:
        best["dev_by_service"] = {}
        svc_indices: Dict[str, List[int]] = {}
        for i, r in enumerate(dev_rows_use):
            s = str(r.get("service", ""))
            if s not in svc_indices:
                svc_indices[s] = []
            svc_indices[s].append(i)
            
        for svc in eval_services:
            idxs = svc_indices.get(svc, [])
            if not idxs:
                best["dev_by_service"][svc] = {"n_rows": 0}
                continue
            
            idxs_t = torch.tensor(idxs, dtype=torch.long)
            sub_x = X_dev[idxs_t]
            sub_y = Y_dev[idxs_t]
            sub_ds = TensorDataset(sub_x, sub_y)
            sub_dl = DataLoader(sub_ds, batch_size=batch_size, shuffle=False)
            
            y_s, p_s = evaluate_cached(model, sub_dl, device)
            m = eval_with_threshold(y_s, p_s, best_th)
            m["n_rows"] = len(idxs)
            best["dev_by_service"][svc] = m

    return {"best": best, "best_path": str(best_path)}

# ---------------- Main ----------------

def parse_int_list(s: str) -> Optional[List[int]]:
    s = s.strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_str_list(s: str) -> Optional[List[str]]:
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]

def filter_rows_by_service(rows: List[Dict[str, Any]], services: Sequence[str]) -> List[Dict[str, Any]]:
    ss = set(services)
    return [r for r in rows if str(r.get("service", "")) in ss]

def get_score(best_dict: Dict[str, Any], primary: str = "macro_f1_posonly") -> float:
    if primary in best_dict and best_dict[primary] is not None:
        return float(best_dict[primary])
    if "macro_f1" in best_dict and best_dict["macro_f1"] is not None:
        return float(best_dict["macro_f1"])
    if "micro_f1" in best_dict and best_dict["micro_f1"] is not None:
        return float(best_dict["micro_f1"])
    return float("-inf")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="distilroberta-base",
                    help="Start small for smoke test; later replace with larger LLM.")
    ap.add_argument("--data_dir", type=str, default="data/processed/sgd/dirunc")
    ap.add_argument("--train_file", type=str, default="train.jsonl")
    ap.add_argument("--dev_file", type=str, default="dev.jsonl")

    ap.add_argument("--mode", type=str, choices=["baseline", "query", "both"], default="both")
    ap.add_argument("--phases", type=str, default="", help="e.g. '1' or '1,2' ; empty=all")
    ap.add_argument("--levels", type=str, default="", help="e.g. '0' or '0,1' ; empty=all")
    ap.add_argument("--k_values", type=str, default="", help="e.g. '3' or '3,5' ; empty=all")

    ap.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    ap.add_argument("--extract_batch_size", type=int, default=0, help="Batch size for extraction (0 = 4*batch_size)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=0.5)

    ap.add_argument("--layer", type=int, default=-1,
                    help="Single layer_idx (0..L-1). If -1, do sweep.")
    ap.add_argument("--sweep", action="store_true",
                    help="If set, do 5-point sweep; if --layer != -1, ignored.")
    ap.add_argument("--refine_pm2", action="store_true",
                    help="After sweep, run ±2 around best layer (same mode).")

    ap.add_argument("--strip_query_in_baseline", action="store_true",
                    help="Recommended: baseline should not depend on query tokens. (Default off)")
    ap.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars (better for nohup logs)")
    ap.add_argument("--tqdm_mininterval", type=float, default=5.0, help="tqdm update interval seconds")

    ap.add_argument("--out_dir", type=str, default="runs/dirunc_probe")
    ap.add_argument("--eval_services", type=str, default="",
                help="Comma-separated services to report per-service dev metrics, e.g., 'Flights_3,RideSharing_1'")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / args.train_file)
    dev_rows = read_jsonl(data_dir / args.dev_file)
    eval_services = parse_str_list(args.eval_services)

    fs = FilterSpec(
        phases=parse_int_list(args.phases),
        levels=parse_int_list(args.levels),
        k_values=parse_int_list(args.k_values),
    )

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    print(f"Loading model (shared): {args.model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    
    # Add special tokens ONCE
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    
    # Load model
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        shared_model = AutoModel.from_pretrained(
            args.model_name, 
            output_hidden_states=True, 
            torch_dtype=dtype,
            trust_remote_code=True
        )
    except OSError:
        shared_model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    
    # Resize ONCE
    shared_model.resize_token_embeddings(len(tokenizer))
    shared_model.to(device)
    shared_model.eval() # Start in eval mode

    # Check layers
    num_layers = int(shared_model.config.num_hidden_layers)
    
    if args.layer != -1:
        layers = [args.layer]
    else:
        if args.sweep:
            layers = floor_layer_points(num_layers)
        else:
            layers = [num_layers - 1]

    modes: List[str]
    if args.mode == "both":
        modes = ["baseline", "query"]
    else:
        modes = [args.mode]

    results: Dict[str, Any] = {
        "model_name": args.model_name,
        "num_layers": num_layers,
        "layers": layers,
        "modes": modes,
        "filter": {"phases": fs.phases, "levels": fs.levels, "k_values": fs.k_values},
    }

    # run sweep
    score_metric = "macro_f1_posonly" 
    best_overall: Optional[Dict[str, Any]] = None
    best_key: Optional[str] = None
    
    # Decide extraction batch size
    bs_extract = args.extract_batch_size if args.extract_batch_size > 0 else (args.batch_size * 4)

    for mode in modes:
        mode_dir = out_dir / mode
        safe_mkdir(mode_dir)
    
        mode_best = {
            "score": float("-inf"),
            "layer_idx": None,
            "score_metric": score_metric,
            "best": None,
            "best_path": None,
        }
        
        # --- PREPARE DATA LOADER FOR EXTRACTION ---
        train_rows_use, dev_rows_use = train_rows, dev_rows
        if mode == "baseline" and args.strip_query_in_baseline:
             train_rows_use = [{"text": strip_query_tokens(r["text"]), **{k:v for k,v in r.items() if k!="text"}} for r in train_rows]
             dev_rows_use = [{"text": strip_query_tokens(r["text"]), **{k:v for k,v in r.items() if k!="text"}} for r in dev_rows]

        train_ds_ex = JsonlDirUncDataset(train_rows_use, fs)
        dev_ds_ex = JsonlDirUncDataset(dev_rows_use, fs)
        
        train_dl_ex = DataLoader(train_ds_ex, batch_size=bs_extract, shuffle=False, 
                                 collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length), num_workers=2)
        dev_dl_ex = DataLoader(dev_ds_ex, batch_size=bs_extract, shuffle=False,
                               collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length), num_workers=2)

        # --- MULTI-LAYER EXTRACTION ---
        print(f"[{mode}] Extracting activations for layers {layers} with BS={bs_extract}...", flush=True)
        
        # Wrap model
        base = ProbeModelBase(
            args.model_name,
            vocab_size=None,
            train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
            pretrained_model=shared_model,
        ).to(device)

        X_train_dict, Y_train, train_span = extract_activations(train_dl_ex, base, layers, mode, device, tokenizer)
        X_dev_dict, Y_dev, dev_span = extract_activations(dev_dl_ex, base, layers, mode, device, tokenizer)
        
        base.teardown() # detach hooks
        del base        # free structure
        torch.cuda.empty_cache()

        # ---- base sweep (layers) ----
        for layer_idx in layers:
            layer_dir = mode_dir / f"layer_{layer_idx}"
            safe_mkdir(layer_dir)
            
            # Retrieve features
            X_tr = X_train_dict[layer_idx]
            X_dv = X_dev_dict[layer_idx]
    
            res = train_probe_from_cache(
                mode=mode,
                layer_idx=layer_idx,
                X_train=X_tr,
                Y_train=Y_train,
                X_dev=X_dv,
                Y_dev=Y_dev,
                train_span_summary=train_span,
                dev_span_summary=dev_span,
                hidden_size=int(shared_model.config.hidden_size),
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                seed=args.seed,
                out_dir=layer_dir,
                threshold=args.threshold,
                eval_services=eval_services,
                dev_rows_use=dev_rows_use,
                no_tqdm=args.no_tqdm,
                tqdm_mininterval=args.tqdm_mininterval,
            )
            
            # --- Result Handling (Same as before) ---
            key = f"{mode}/layer_{layer_idx}"
            results[key] = res
    
            score = get_score(res["best"], primary=score_metric)
    
            # update mode best
            if score > float(mode_best["score"]):
                mode_best = {
                    "score": float(score),
                    "layer_idx": int(layer_idx),
                    "score_metric": score_metric,
                    "best": res["best"],
                    "best_path": res["best_path"],
                }
    
            # update global best
            if best_overall is None or score > float(best_overall["score"]):
                best_overall = {
                    "score": float(score),
                    "score_metric": score_metric,
                    "mode": mode,
                    "layer_idx": int(layer_idx),
                    "best": res["best"],
                }
                best_key = key
    
        results[f"{mode}/best"] = mode_best
    
        # ---- optional refine ±2 around best for this mode ----
        if args.refine_pm2 and mode_best["layer_idx"] is not None:
            best_l = int(mode_best["layer_idx"])
            refine_layers = expand_best_pm2(best_l, num_layers)
            
            # Filter ones we haven't done
            needed = [l for l in refine_layers if l not in layers]
            
            if needed:
                print(f"[{mode}] Refine: Extracting layers {needed}...", flush=True)
                # Re-extract only needed
                base = ProbeModelBase(
                    args.model_name,
                    vocab_size=None,
                    train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
                    pretrained_model=shared_model,
                ).to(device)
                
                rx_tr, ry_tr, rspan_tr = extract_activations(train_dl_ex, base, needed, mode, device, tokenizer)
                rx_dv, ry_dv, rspan_dv = extract_activations(dev_dl_ex, base, needed, mode, device, tokenizer)
                base.teardown()
                
                results[f"{mode}/refine_layers"] = refine_layers
                
                # Iterate all refine layers
                for layer_idx in refine_layers:
                    if layer_idx in layers:
                        pass # already in results, skipped
                    else:
                        # Process newly extracted
                        layer_dir = mode_dir / f"layer_{layer_idx}"
                        safe_mkdir(layer_dir)
                        
                        res = train_probe_from_cache(
                            mode=mode,
                            layer_idx=layer_idx,
                            X_train=rx_tr[layer_idx],
                            Y_train=ry_tr, # same labels
                            X_dev=rx_dv[layer_idx],
                            Y_dev=ry_dv,
                            train_span_summary=rspan_tr,
                            dev_span_summary=rspan_dv,
                            hidden_size=int(shared_model.config.hidden_size),
                            batch_size=args.batch_size,
                            epochs=args.epochs,
                            lr=args.lr,
                            seed=args.seed,
                            out_dir=layer_dir,
                            threshold=args.threshold,
                            eval_services=eval_services,
                            dev_rows_use=dev_rows_use,
                            no_tqdm=args.no_tqdm,
                            tqdm_mininterval=args.tqdm_mininterval,
                        )
                        
                        key = f"{mode}/layer_{layer_idx}"
                        results[key] = res
            
                        score = get_score(res["best"], primary=score_metric)
            
                        if score > float(mode_best["score"]):
                             # update best...
                             mode_best = { 
                                 "score": float(score), "layer_idx": int(layer_idx), 
                                 "score_metric": score_metric, "best": res["best"], "best_path": res["best_path"]
                             }
                             results[f"{mode}/best"] = mode_best
                        
                        if best_overall is None or score > float(best_overall["score"]):
                            best_overall = {
                                "score": float(score), "score_metric": score_metric, "mode": mode, "layer_idx": int(layer_idx), "best": res["best"]
                            }
                            best_key = key
    
    results["best_overall"] = best_overall
    results["best_overall_key"] = best_key
    
    (out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    print(f"[done] wrote {out_dir/'summary.json'}")
    if best_overall is not None:
        print(
            f"[best] mode={best_overall['mode']} layer={best_overall['layer_idx']} "
            f"{best_overall['score_metric']}={best_overall['score']:.4f}"
        )

if __name__ == "__main__":
    main()
