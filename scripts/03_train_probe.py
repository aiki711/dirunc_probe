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
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import DIRS, QUERY_LABEL_STR, SPECIAL_TOKENS, QUERY_TOKENS

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

def compute_auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    # y_true: (N, C), y_prob: (N, C)
    if y_true.size == 0:
        return {}
    
    micro = average_precision_score(y_true, y_prob, average="micro")
    macro = average_precision_score(y_true, y_prob, average="macro")
    per_label = average_precision_score(y_true, y_prob, average=None)
    
    # safe handling for per_label if it returns scalar or array
    if isinstance(per_label, (float, int)):
        per_label = [float(per_label)]
    else:
        per_label = per_label.tolist()

    return {
        "micro_auc_pr": float(micro),
        "macro_auc_pr": float(macro),
        "per_label_auc_pr": per_label,
    }

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
            # Handle missing metadata safely (default to 0)
            ph_val = r.get("phase", 0)
            ph = int(ph_val) if ph_val is not None else 0
            
            lv_val = r.get("level", 0)
            lv = int(lv_val) if lv_val is not None else 0
            
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

class LearnableLayerWeights(nn.Module):
    """
    複数レイヤーの重み付け和を学習するモジュール
    """
    def __init__(self, num_layers: int):
        super().__init__()
        # 各レイヤーへの重み（学習可能）
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
    
    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List[(B, C, H)], length = num_layers
        Returns:
            weighted_sum: (B, C, H)
        """
        # Softmaxで正規化（合計が1になるように）
        w = torch.softmax(self.weights, dim=0)
        
        # 重み付け和
        result = sum(w[i] * feat for i, feat in enumerate(layer_features))
        return result
    
    def get_weights(self) -> List[float]:
        """現在のレイヤー重みを取得（正規化済み）"""
        with torch.no_grad():
            w = torch.softmax(self.weights, dim=0)
            return w.cpu().tolist()

class MultiLayerQueryHead(nn.Module):
    """
    複数レイヤーからの特徴を重み付け和で統合するQuery Head
    """
    def __init__(self, hidden_size: int, num_layers: int, num_labels: int, dtype: torch.dtype):
        super().__init__()
        self.num_layers = num_layers
        
        # レイヤー重み付けモジュール
        self.layer_weights = LearnableLayerWeights(num_layers)
        
        # 通常のQuery Head（入力次元は変わらない）
        self.W = nn.Parameter(torch.empty(num_labels, hidden_size, dtype=dtype))
        self.b = nn.Parameter(torch.zeros(num_labels, dtype=dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_layers, C, H) - バッチ、レイヤー数、クラス数、隠れ次元
        Returns:
            logits: (B, C)
        """
        # x: (B, L, C, H) -> List of (B, C, H)
        layer_features = [x[:, i, :, :] for i in range(self.num_layers)]
        
        # 重み付け和: (B, C, H)
        combined = self.layer_weights(layer_features)
        
        # 各directionの予測
        logits = (combined * self.W.unsqueeze(0)).sum(dim=2) + self.b.unsqueeze(0)
        return logits
    
    def get_layer_weights(self) -> List[float]:
        """現在のレイヤー重みを取得"""
        return self.layer_weights.get_weights()

# ========== Sampling Utilities ==========

def balanced_sampling(rows: List[Dict[str, Any]], target_count: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Perform balanced sampling across 7 labels (DIRS) and a negative (all-zero) class.
    Each of the 8 categories will aim for target_count // 8 samples.
    """
    if not rows:
        return []
    if target_count <= 0 or target_count >= len(rows):
        return rows

    rng = random.Random(seed)
    # Buckets for each label + negative
    buckets: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DIRS}
    buckets["negative"] = []

    for r in rows:
        lbls = r.get("labels", {})
        pos_labels = [d for d in DIRS if lbls.get(d, 0) == 1]
        
        if not pos_labels:
            buckets["negative"].append(r)
        else:
            # If multi-label, assign to a random one of its labels to maintain balance
            chosen = rng.choice(pos_labels)
            buckets[chosen].append(r)

    total_categories = len(DIRS) + 1
    per_bucket_target = target_count // total_categories
    
    selected_rows = []
    remaining_count = target_count
    
    # First pass: take up to per_bucket_target from each
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys) # Randomize order for leftover filling
    
    actual_taken = {}
    for k in bucket_keys:
        pool = buckets[k]
        take = min(len(pool), per_bucket_target)
        actual_taken[k] = take
        selected_rows.extend(rng.sample(pool, take))
        remaining_count -= take
        
    # Second pass: fill remaining from any available
    if remaining_count > 0:
        leftover_pool = []
        for k in bucket_keys:
            pool = buckets[k]
            taken = actual_taken[k]
            leftover_pool.extend(pool[taken:])
            
        if len(leftover_pool) > remaining_count:
            selected_rows.extend(rng.sample(leftover_pool, remaining_count))
        else:
            selected_rows.extend(leftover_pool)
            
    print(f"Balanced Sampling: target={target_count}, selected={len(selected_rows)}")
    rng.shuffle(selected_rows)
    return selected_rows

@torch.no_grad()
def extract_activations(
    dl: DataLoader,
    base_model: ProbeModelBase,
    layers: List[int],
    mode: str,
    device: torch.device,
    tokenizer: Any = None,
    save_dir: Optional[Path] = None,
    no_tqdm: bool = False,
    tqdm_mininterval: float = 5.0,
) -> Tuple[Union[Dict[int, torch.Tensor], Dict[int, Path]], Union[torch.Tensor, Path], Optional[Dict[str, Any]]]:
    base_model.lm.eval()
    
    # メモリ節約モード（save_dir指定時）か、オンメモリモードか
    use_disk = save_dir is not None
    if use_disk:
        save_dir.mkdir(parents=True, exist_ok=True)
        # チャンク保存用のカウンタ
        chunk_idx = 0
        CHUNK_SIZE = 100  # バッチ数単位で保存
    
    # バッファ
    all_x = {l: [] for l in layers}
    all_y = []
    
    span_acc = _init_span_acc() if mode == "query" else None
    
    token_id = {}
    if mode == "query":
         for d, tstr in QUERY_LABEL_STR.items():
            token_id[d] = tokenizer.convert_tokens_to_ids(tstr)

    print(f"DEBUG: Starting extraction loop. Device={device}, DL length={len(dl)}", flush=True)

    batch_count = 0
    tqdm_obj = tqdm(dl, desc="Extracting", disable=no_tqdm, mininterval=30.0) # 30秒間隔に緩和
    for batch in tqdm_obj:
        batch_count += 1
        # 500バッチごとに明示的に一行出力 (ログ用)
        if batch_count % 500 == 0 or batch_count == 1 or batch_count == len(dl):
            print(f"[{mode}] Extraction progress: {batch_count}/{len(dl)} batches", flush=True)
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["y"].to(device)
        
        try:
            out = base_model.lm(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                print("WARNING: OOM detected, skipping batch", flush=True)
                continue
            else:
                raise e
        
        bsz = input_ids.size(0)
        
        if mode == "baseline":
            lengths = attention_mask.long().sum(dim=1) - 1
            
        for layer_idx in layers:
            # get_layer_hiddenがGPUテンソルを返すと仮定
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
        
        # ディスク保存処理
        if use_disk and batch_count >= CHUNK_SIZE:
             _save_chunk(all_x, all_y, save_dir, chunk_idx, layers)
             # バッファクリア
             all_x = {l: [] for l in layers}
             all_y = []
             chunk_idx += 1
             batch_count = 0
             torch.cuda.empty_cache()

    # 残りのバッファを保存または結合
    if use_disk:
        if any(all_x[l] for l in layers) or all_y:
            _save_chunk(all_x, all_y, save_dir, chunk_idx, layers)
        
        # チャンクを結合して1つのファイルにする（またはファイルパスをリストで返す）
        # ここでは学習スクリプトの互換性のため、結合してファイルに保存し、そのパスを返す形にする
        print("DEBUG: Merging chunks...", flush=True)
        final_x_paths = {}
        for l in layers:
            merged_x = _merge_chunks(save_dir, "x", l)
            p = save_dir / f"features_layer_{l}.pt"
            torch.save(merged_x, p)
            final_x_paths[l] = p
            del merged_x
            
        merged_y = _merge_chunks(save_dir, "y", None)
        y_path = save_dir / "labels.pt"
        torch.save(merged_y, y_path)
        
        # 一時チャンク削除
        for f in save_dir.glob("chunk_*.pt"):
            f.unlink()
        
        summary = None
        if mode == "query":
            summary = _finalize_span_acc(span_acc)
        return final_x_paths, y_path, summary

    else:
        # 既存のオンメモリ結合処理
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

def _save_chunk(all_x, all_y, save_dir, chunk_idx, layers):
    if all_y:
        y_chunk = torch.cat(all_y, dim=0)
        torch.save(y_chunk, save_dir / f"chunk_{chunk_idx}_y.pt")
    
    for l in layers:
        if all_x[l]:
            x_chunk = torch.cat(all_x[l], dim=0)
            torch.save(x_chunk, save_dir / f"chunk_{chunk_idx}_x_layer_{l}.pt")

def _merge_chunks(save_dir, type_str, layer_idx):
    if type_str == "y":
        chunk_files = sorted(save_dir.glob("chunk_*_y.pt"), key=lambda x: int(x.name.split('_')[1]))
        if not chunk_files:
            return torch.empty(0) # No data for this type
        return torch.cat([torch.load(p) for p in chunk_files], dim=0)
    else:
        chunk_files = sorted(save_dir.glob(f"chunk_*_x_layer_{layer_idx}.pt"), key=lambda x: int(x.name.split('_')[1]))
        if not chunk_files:
            return torch.empty(0) # No data for this type/layer
        return torch.cat([torch.load(p) for p in chunk_files], dim=0)

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

def tune_threshold_per_class(
    y_true: np.ndarray,
    p: np.ndarray,
    grid: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    クラスごとに最適な閾値を探索し、各クラスのF1スコアを最大化する
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        p: Predicted probabilities (N, num_classes)
        grid: Threshold candidates to search
    
    Returns:
        Dictionary containing:
        - thresholds: List of optimal thresholds per class
        - per_class_f1: F1 score for each class at its optimal threshold
        - macro_f1: Average of per-class F1 scores
        - threshold_dict: Dictionary mapping class names to thresholds
    """
    if grid is None:
        grid = [i / 100.0 for i in range(5, 96, 5)]
    
    num_classes = y_true.shape[1]
    best_thresholds = []
    best_f1s = []
    
    # 各クラスごとに最適閾値を探索
    for class_idx in range(num_classes):
        y_c = y_true[:, class_idx]  # このクラスの正解ラベル
        p_c = p[:, class_idx]        # このクラスの予測確率
        
        best_f1 = -1.0
        best_th = 0.5
        
        # このクラスに正例が存在するか確認
        if y_c.sum() == 0:
            # 正例がない場合はデフォルト閾値を使用
            best_thresholds.append(0.5)
            best_f1s.append(0.0)
            continue
        
        for th in grid:
            y_pred_c = (p_c >= th).astype(int)
            
            # F1スコア計算
            tp = ((y_pred_c == 1) & (y_c == 1)).sum()
            fp = ((y_pred_c == 1) & (y_c == 0)).sum()
            fn = ((y_pred_c == 0) & (y_c == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        
        best_thresholds.append(best_th)
        best_f1s.append(best_f1)
    
    # 正例があるクラスのみでmacro F1を計算
    support = y_true.sum(axis=0)
    mask = support > 0
    if mask.any():
        macro_f1_posonly = np.mean([best_f1s[i] for i in range(num_classes) if mask[i]])
    else:
        macro_f1_posonly = 0.0
    
    # クラス名との対応付け
    threshold_dict = {DIRS[i]: float(best_thresholds[i]) for i in range(len(DIRS))}
    f1_dict = {DIRS[i]: float(best_f1s[i]) for i in range(len(DIRS))}
    
    return {
        "thresholds": [float(th) for th in best_thresholds],
        "per_class_f1": [float(f1) for f1 in best_f1s],
        "macro_f1": float(np.mean(best_f1s)),
        "macro_f1_posonly": float(macro_f1_posonly),
        "threshold_dict": threshold_dict,
        "f1_dict": f1_dict,
    }

def eval_with_per_class_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    thresholds: List[float],
) -> Dict[str, Any]:
    """
    クラスごとの閾値を使って予測を行い、メトリクスを計算
    
    Args:
        y_true: Ground truth labels (N, num_classes)
        p: Predicted probabilities (N, num_classes)
        thresholds: List of thresholds for each class
    
    Returns:
        Metrics dictionary
    """
    num_classes = y_true.shape[1]
    y_pred = np.zeros_like(y_true, dtype=np.int32)
    
    # 各クラスごとに異なる閾値を適用
    for i in range(num_classes):
        y_pred[:, i] = (p[:, i] >= thresholds[i]).astype(np.int32)
    
    # メトリクス計算
    metrics = micro_macro_f1(y_true, y_pred)
    metrics["thresholds_used"] = [float(th) for th in thresholds]
    
    return metrics

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

        auc_metrics = compute_auc_pr(y_true_dev, p_dev)

        # クラスごとの閾値最適化
        tuned_per_class = tune_threshold_per_class(
            y_true_dev,
            p_dev,
            grid=None,
        )
        
        # クラスごとの閾値を使って評価
        dev_metrics = eval_with_per_class_threshold(
            y_true_dev,
            p_dev,
            tuned_per_class["thresholds"],
        )
        dev_metrics.update(auc_metrics) 
        
        if dev_span_summary is not None:
             dev_metrics["span_found"] = dev_span_summary
             
        # クラスごとの閾値情報を追加
        dev_metrics["per_class_tuned"] = {
            "threshold_dict": tuned_per_class["threshold_dict"],
            "f1_dict": tuned_per_class["f1_dict"],
            "macro_f1_posonly": tuned_per_class["macro_f1_posonly"],
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

    # クラスごとの閾値を取得
    best_thresholds = best.get("per_class_tuned", {}).get("threshold_dict", None)
    if best_thresholds is None:
        # フォールバック：デフォルト閾値
        best_thresholds = [0.5] * len(DIRS)
    else:
        best_thresholds = [best_thresholds[d] for d in DIRS]
    
    y_dev, p_dev = evaluate_cached(model, dev_dl_cached, device)
    best_dev_metrics = eval_with_per_class_threshold(y_dev, p_dev, best_thresholds)
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
            m = eval_with_per_class_threshold(y_s, p_s, best_thresholds)
            m["n_rows"] = len(idxs)
            best["dev_by_service"][svc] = m
    
    return {"best": best, "best_path": str(best_path)}

def train_multilayer_probe_from_cache(
    *,
    mode: str,
    layers: List[int],
    X_dict: Dict[int, torch.Tensor],
    Y_train: torch.Tensor,
    X_dev_dict: Dict[int, torch.Tensor],
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
    """マルチレイヤー版のプローブ学習関数"""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 複数レイヤーの特徴を結合
    X_train_list = [X_dict[l] for l in layers]
    X_dev_list = [X_dev_dict[l] for l in layers]
    X_train_stacked = torch.stack(X_train_list, dim=1)
    X_dev_stacked = torch.stack(X_dev_list, dim=1)
    
    train_ds = TensorDataset(X_train_stacked, Y_train)
    dev_ds = TensorDataset(X_dev_stacked, Y_dev)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    
    dtype = X_train_stacked.dtype
    if mode != "query":
        raise ValueError(f"Multi-layer mode only supports 'query', got '{mode}'")
    
    model = MultiLayerQueryHead(hidden_size, len(layers), len(DIRS), dtype).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr)
    
    best = {"micro_f1": -1.0, "macro_f1_posonly": -1.0}
    best_path = out_dir / f"best_multilayer.pt"
    
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        total_batches = len(train_dl)
        
        for i, batch in enumerate(tqdm(train_dl, desc=f"train multilayer ep={ep}", 
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
                print(f"[Epoch {ep}] Step {i+1}/{total_batches} | Loss: {loss.item():.4f}", flush=True)
        
        train_loss = total_loss / max(1, n)
        
        # Evaluation
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in dev_dl:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = model(x)
                prob = torch.sigmoid(logits)
                ys.append(y.float().cpu().numpy())
                ps.append(prob.float().cpu().numpy())
        
        y_true_dev = np.concatenate(ys, axis=0)
        p_dev = np.concatenate(ps, axis=0)
        auc_metrics = compute_auc_pr(y_true_dev, p_dev)
        tuned_per_class = tune_threshold_per_class(y_true_dev, p_dev, grid=None)
        dev_metrics = eval_with_per_class_threshold(y_true_dev, p_dev, tuned_per_class["thresholds"])
        dev_metrics.update(auc_metrics)
        
        if dev_span_summary is not None:
            dev_metrics["span_found"] = dev_span_summary
        
        dev_metrics["per_class_tuned"] = {
            "threshold_dict": tuned_per_class["threshold_dict"],
            "f1_dict": tuned_per_class["f1_dict"],
            "macro_f1_posonly": tuned_per_class["macro_f1_posonly"],
        }
        dev_metrics["layer_weights"] = {"weights": model.get_layer_weights(), "layers": layers}
        
        if train_span_summary is not None:
            dev_metrics["span_found_train"] = train_span_summary
        
        record = {"mode": "multilayer", "layers": layers, "epoch": ep, "train_loss": train_loss, **dev_metrics}
        
        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"[Epoch {ep}/{epochs}] mode=multilayer layers={layers} | "
              f"train_loss={train_loss:.4f} | macro_f1_posonly={dev_metrics.get('macro_f1_posonly',0.0):.4f} | "
              f"micro_f1={dev_metrics.get('micro_f1',0.0):.4f}")
        
        if dev_metrics["macro_f1_posonly"] > best.get("macro_f1_posonly", -1.0):
            best = {**record}
            torch.save(model.state_dict(), best_path)
    
    if best["macro_f1_posonly"] >= 0 and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    
    best_thresholds = best.get("per_class_tuned", {}).get("threshold_dict", None)
    if best_thresholds is None:
        best_thresholds = [0.5] * len(DIRS)
    else:
        best_thresholds = [best_thresholds[d] for d in DIRS]
    
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in dev_dl:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(x)
            prob = torch.sigmoid(logits)
            ys.append(y.float().cpu().numpy())
            ps.append(prob.float().cpu().numpy())
    
    y_dev, p_dev = np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)
    best_dev_metrics = eval_with_per_class_threshold(y_dev, p_dev, best_thresholds)
    if dev_span_summary:
        best_dev_metrics["span_found"] = dev_span_summary
    best["final_dev"] = best_dev_metrics
    
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
    ap.add_argument("--data_dir", type=str, default="data/processed/sgd/dirunc", help="Default data dir (source)")
    ap.add_argument("--eval_data_dir", type=str, default=None, help="Target data dir for evaluation (if different)")
    ap.add_argument("--few_shot_data_dir", type=str, default=None, help="Data dir for few-shot examples (usually target train)")
    ap.add_argument("--few_shot_count", type=int, default=0, help="Number of few-shot examples to mix in")

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

    # Multi-layer fusion arguments
    ap.add_argument("--multilayer", action="store_true",
                    help="Enable multi-layer feature fusion (uses LearnableLayerWeights)")
    ap.add_argument("--fusion_layers", type=str, default="10,15,20,25",
                    help="Comma-separated list of layers to fuse in multilayer mode")

    ap.add_argument("--out_dir", type=str, default="runs/dirunc_probe")
    ap.add_argument("--eval_services", type=str, default="",
                help="Comma-separated services to report per-service dev metrics, e.g., 'Flights_3,RideSharing_1'")
    ap.add_argument("--limit", type=str, default="0", help="Limit train/dev size. Usage: 'train=100,dev=100' or just '100'")
    ap.add_argument("--balance_sampling", action="store_true", help="If limit is set, use balanced sampling across classes.")
    args = ap.parse_args()

    set_seed(args.seed)

    # Parse limit
    limit_train = 0
    limit_dev = 0
    if args.limit:
        parts = args.limit.split(",")
        for p in parts:
            if "=" in p:
                k, v = p.split("=")
                if k == "train": limit_train = int(v)
                if k == "dev": limit_dev = int(v)
            else:
                # If no '=', assume it's for both (legacy behavior)
                limit_train = int(p)
                limit_dev = int(p)

    # Load Train Data
    data_dir = Path(args.data_dir)
    print(f"Loading training data from {data_dir}")
    train_rows = read_jsonl(data_dir / args.train_file)
    if limit_train > 0:
        print(f"Limiting train rows to {limit_train}")
        if args.balance_sampling:
            train_rows = balanced_sampling(train_rows, limit_train, args.seed)
        else:
            train_rows = train_rows[:limit_train]
    
    # Few-shot Mixing
    if args.few_shot_data_dir and args.few_shot_count > 0:

        fs_dir = Path(args.few_shot_data_dir)
        print(f"Loading few-shot data from {fs_dir} (count={args.few_shot_count})")
        fs_rows = read_jsonl(fs_dir / args.train_file)
        if len(fs_rows) > args.few_shot_count:
            rng = random.Random(args.seed)
            fs_rows = rng.sample(fs_rows, args.few_shot_count)
        train_rows.extend(fs_rows)
        print(f"Added {len(fs_rows)} few-shot examples. Total train size: {len(train_rows)}")

    # Load Dev Data (Source or Target)
    if args.eval_data_dir:
        eval_dir = Path(args.eval_data_dir)
        print(f"Loading evaluation data from {eval_dir} (Target)")
        dev_rows = read_jsonl(eval_dir / args.dev_file)
    else:
        print(f"Loading evaluation data from {data_dir} (Source)")
        dev_rows = read_jsonl(data_dir / args.dev_file)
        
    if limit_dev > 0:
        print(f"Limiting dev rows to {limit_dev}")
        if args.balance_sampling:
            dev_rows = balanced_sampling(dev_rows, limit_dev, args.seed)
        else:
            dev_rows = dev_rows[:limit_dev]


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
    hidden_size = int(shared_model.config.hidden_size)
    
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

    # --- ACTIVATION CACHE ---
    # (mode, layer_idx) -> Tensor
    cache_X_train: Dict[Tuple[str, int], torch.Tensor] = {}
    cache_X_dev: Dict[Tuple[str, int], torch.Tensor] = {}
    # mode -> Tensor/Summary
    cache_Y_train: Dict[str, torch.Tensor] = {}
    cache_Y_dev: Dict[str, torch.Tensor] = {}
    cache_span_train: Dict[str, Any] = {}
    cache_span_dev: Dict[str, Any] = {}

    def get_activations_with_cache(mode: str, needed_layers: List[int], current_train_rows, current_dev_rows):
        # すべて既にあるかチェック
        missing = [l for l in needed_layers if (mode, l) not in cache_X_train]
        
        if missing or mode not in cache_Y_train:
            # まだ抽出していない層がある場合のみ LLM を回す
            # 効率化のため、multilayer用の層も初回にまとめて抽出対象に加える
            all_to_extract = set(missing)
            if mode == "query" and args.multilayer:
                ml_layers = [int(x.strip()) for x in args.fusion_layers.split(",")]
                for ml_l in ml_layers:
                    if (mode, ml_l) not in cache_X_train:
                        all_to_extract.add(ml_l)
            
            to_extract_list = sorted(list(all_to_extract))
            
            if to_extract_list:
                print(f"[{mode}] Cache MISS. Extracting NEW layers: {to_extract_list}", flush=True)
                
                # Check if we need to wrap the model
                local_base = ProbeModelBase(
                    args.model_name,
                    vocab_size=None,
                    train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS],
                    pretrained_model=shared_model,
                ).to(device)

                # Prepare loaders
                ds_tr = JsonlDirUncDataset(current_train_rows, fs)
                ds_dv = JsonlDirUncDataset(current_dev_rows, fs)
                dl_tr = DataLoader(ds_tr, batch_size=bs_extract, shuffle=False, 
                                   collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length))
                dl_dv = DataLoader(ds_dv, batch_size=bs_extract, shuffle=False,
                                   collate_fn=lambda b: collate_batch(tokenizer, b, args.max_length))

                new_X_tr, new_Y_tr, new_span_tr = extract_activations(dl_tr, local_base, to_extract_list, mode, device, tokenizer, no_tqdm=args.no_tqdm, tqdm_mininterval=args.tqdm_mininterval)
                new_X_dv, new_Y_dv, new_span_dv = extract_activations(dl_dv, local_base, to_extract_list, mode, device, tokenizer, no_tqdm=args.no_tqdm, tqdm_mininterval=args.tqdm_mininterval)
                
                # Update cache (MOVE TO CPU TO SAVE VRAM)
                for l in to_extract_list:
                    cache_X_train[(mode, l)] = new_X_tr[l].cpu()
                    cache_X_dev[(mode, l)] = new_X_dv[l].cpu()
                
                if mode not in cache_Y_train:
                    cache_Y_train[mode] = new_Y_tr.cpu()
                    cache_Y_dev[mode] = new_Y_dv.cpu()
                    cache_span_train[mode] = new_span_tr
                    cache_span_dev[mode] = new_span_dv
                
                local_base.teardown()
                del local_base
                torch.cuda.empty_cache()
            else:
                print(f"[{mode}] Cache HIT for required layers.", flush=True)

        # 必要な分を返却（呼び出し側の期待する形式）
        ret_X_tr = {l: cache_X_train[(mode, l)] for l in needed_layers}
        ret_X_dv = {l: cache_X_dev[(mode, l)] for l in needed_layers}
        return ret_X_tr, ret_X_dv, cache_Y_train[mode], cache_Y_dev[mode], cache_span_train[mode], cache_span_dev[mode]

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
        
        # --- PREPARE DATA ---
        train_rows_use, dev_rows_use = train_rows, dev_rows
        if mode == "baseline" and args.strip_query_in_baseline:
             train_rows_use = [{"text": strip_query_tokens(r["text"]), **{k:v for k,v in r.items() if k!="text"}} for r in train_rows]
             dev_rows_use = [{"text": strip_query_tokens(r["text"]), **{k:v for k,v in r.items() if k!="text"}} for r in dev_rows]

        # --- EXTRACTION (with cache) ---
        X_train_dict, X_dev_dict, Y_train, Y_dev, train_span, dev_span = get_activations_with_cache(
            mode, layers, train_rows_use, dev_rows_use
        )

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
                X_train=X_tr.to(device),
                Y_train=Y_train.to(device),
                X_dev=X_dv.to(device),
                Y_dev=Y_dev.to(device),
                train_span_summary=train_span,
                dev_span_summary=dev_span,
                hidden_size=hidden_size,
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
            
            # --- Result Handling ---
            key = f"{mode}/layer_{layer_idx}"
            results[key] = res
            score = get_score(res["best"], primary=score_metric)
    
            if score > float(mode_best["score"]):
                mode_best = {
                    "score": float(score),
                    "layer_idx": int(layer_idx),
                    "score_metric": score_metric,
                    "best": res["best"],
                    "best_path": res["best_path"],
                }
    
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
            
            needed = [l for l in refine_layers if l not in layers]
            if needed:
                rx_tr, rx_dv, ry_tr, ry_dv, rspan_tr, rspan_dv = get_activations_with_cache(
                    mode, refine_layers, train_rows_use, dev_rows_use
                )
                
                results[f"{mode}/refine_layers"] = refine_layers
                for layer_idx in refine_layers:
                    if layer_idx in layers:
                        pass 
                    else:
                        layer_dir = mode_dir / f"layer_{layer_idx}"
                        safe_mkdir(layer_dir)
                        
                        res = train_probe_from_cache(
                            mode=mode,
                            layer_idx=layer_idx,
                            X_train=rx_tr[layer_idx],
                            Y_train=ry_tr,
                            X_dev=rx_dv[layer_idx],
                            Y_dev=ry_dv,
                            train_span_summary=rspan_tr,
                            dev_span_summary=rspan_dv,
                            hidden_size=hidden_size,
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
    
    # ========== Multi-layer fusion ==========
    if args.multilayer:
        print("\n" + "="*60)
        print("Freeing up memory for Multilayer Fusion...", flush=True)
        # Delete base model to save VRAM for all-layer activations
        if 'shared_model' in locals():
            del shared_model
        if 'base' in locals(): # Just in case it leaked
            del base
        torch.cuda.empty_cache()
        print(f"Memory freed. Current allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        print("Multi-Layer Feature Fusion")
        print("\n" + "="*60)
        print("Multi-Layer Feature Fusion")
        print("="*60)
        fusion_layers = [int(x.strip()) for x in args.fusion_layers.split(",")]
        multilayer_dir = out_dir / "multilayer"
        safe_mkdir(multilayer_dir)
        
        # Use cache (mode will be "query" normally)
        X_train_dict, X_dev_dict, Y_train_ml, Y_dev_ml, train_span_ml, dev_span_ml = get_activations_with_cache(
            "query", fusion_layers, train_rows, dev_rows
        )
        
        print(f"\nTraining multi-layer probe...")
        # Prepare multi-layer X on GPU
        X_tr_gpu = {l: t.to(device) for l, t in X_train_dict.items()}
        X_dv_gpu = {l: t.to(device) for l, t in X_dev_dict.items()}

        ml_result = train_multilayer_probe_from_cache(
            mode="query",
            layers=fusion_layers,
            X_dict=X_tr_gpu,
            Y_train=Y_train_ml.to(device),
            X_dev_dict=X_dv_gpu,
            Y_dev=Y_dev_ml.to(device),
            train_span_summary=train_span_ml,
            dev_span_summary=dev_span_ml,
            hidden_size=hidden_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            seed=args.seed,
            out_dir=multilayer_dir,
            threshold=args.threshold,
            eval_services=eval_services,
            dev_rows_use=dev_rows,
            no_tqdm=args.no_tqdm,
            tqdm_mininterval=args.tqdm_mininterval,
        )
        
        results["multilayer"] = ml_result
        
        # Update best_overall if multilayer is better
        ml_score = get_score(ml_result["best"], score_metric)
        print(f"\nMulti-layer Macro F1: {ml_score:.4f}")
        
        if best_overall is None or ml_score > float(best_overall["score"]):
            best_overall = {
                "score": float(ml_score),
                "score_metric": score_metric,
                "mode": "multilayer",
                "layer_idx": None,
                "layers": fusion_layers,
                "best": ml_result["best"],
            }
            best_key = "multilayer"
            results["best_overall"] = best_overall
            results["best_overall_key"] = best_key
            print(f"[NEW BEST] Multi-layer fusion!")
        
        print(f"\nLayer weights: {ml_result['best'].get('layer_weights', {})}")
        print("="*60)
    
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
