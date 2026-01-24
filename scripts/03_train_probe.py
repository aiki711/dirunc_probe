# scripts/03_train_probe.py
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Must match dataset script
QUERY_TOKENS = " [WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?]"
DIRS = ["who", "what", "when", "where", "why", "how"]
DIR2IDX = {d: i for i, d in enumerate(DIRS)}

QUERY_LABEL_STR = {
    "who": "[WHO?]",
    "what": "[WHAT?]",
    "when": "[WHEN?]",
    "where": "[WHERE?]",
    "why": "[WHY?]",
    "how": "[HOW?]",
}
SPECIAL_TOKENS = list(QUERY_LABEL_STR.values())

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
    if t.endswith(QUERY_TOKENS):
        return t[: -len(QUERY_TOKENS)].rstrip()
    return text

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
    k_values: Optional[List[int]] = None  # only for phase2; None means keep all

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
                # keep Phase1 always; filter k only for Phase2
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
    """Return (start, end_exclusive) of the last occurrence of needle in hay, else None."""
    if not needle or not hay or len(needle) > len(hay):
        return None
    # search from end (query tokens are near the end)
    for start in range(len(hay) - len(needle), -1, -1):
        if hay[start : start + len(needle)] == needle:
            return (start, start + len(needle))
    return None

    # ---- NEW: make query tokens single special tokens ----
    # For LLMs, we usually need left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    special_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
    
    # Resize model embeddings if needed happens inside ProbeModelBase now
    
    # ... (dataset prep code omitted, assumes same) ...

    # When creating ProbeModelBase, pass torch_dtype if available
    # We'll modify ProbeModelBase to handle loading arguments better
    
    base = ProbeModelBase(
        model_name,
        vocab_size=len(tokenizer),
        train_token_ids=special_token_ids,
    ).to(device)

    # ...
    
class ProbeModelBase(nn.Module):
    """Shared: frozen LM + output_hidden_states. Optionally train only special-token embeddings."""
    def __init__(self, model_name: str, *, 
                 vocab_size: Optional[int] = None, 
                 train_token_ids: Optional[List[int]] = None,
                 pretrained_model: Optional[nn.Module] = None) -> None:
        super().__init__()
        
        if pretrained_model is not None:
            self.lm = pretrained_model
            # Assume the model is already on the correct device/dtype if strictly managed,
            # but we can check or trust the caller.
            # We assume vocab resizing is already done on the shared model if needed.
        else:
            # Use bfloat16 or float16 for LLMs to save memory if CUDA available
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
                 # Fallback or strict fail
                 self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True)
            
            if vocab_size is not None:
                cur = self.lm.get_input_embeddings().weight.size(0)
                if vocab_size > cur:
                    self.lm.resize_token_embeddings(vocab_size)

        # Ensure model is in expected dtype
        self.output_dtype = self.lm.dtype

        # freeze everything (reset to frozen state)
        for p in self.lm.parameters():
            p.requires_grad = False

        self.train_token_ids: List[int] = train_token_ids or []
        self.hook_handle = None

        if self.train_token_ids:
            emb = self.lm.get_input_embeddings()
            emb.weight.requires_grad = True
            
            # grad mask logic from before
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
        """Clean up hooks and reset grad status for shared model re-use."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        
        # Reset embeddings requires_grad to False to match 'frozen' baseline state
        if hasattr(self, 'lm') and self.lm is not None:
            emb = self.lm.get_input_embeddings()
            if emb:
                emb.weight.requires_grad = False

    def get_layer_hidden(self, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        # Decoder models (like Llama) output tuple of hidden states.
        # usually index 0 is embedding, last is final.
        # But 'hidden_states' tuple length = num_layers + 1 (embeddings).
        
        # Robust layer indexing handling negative index
        if layer_idx < 0:
            # -1 means last layer
            idx = len(hidden_states) + layer_idx
        else:
            # +1 because 0 is embeddings
            idx = layer_idx + 1
            
        if idx < 0 or idx >= len(hidden_states):
             # Fallback to last
             idx = -1
             
        return hidden_states[idx]

class EosPoolingProbe(nn.Module):
    """Baseline: EOS pooling (last non-pad token) + linear head."""
    def __init__(self, base: ProbeModelBase) -> None:
        super().__init__()
        self.base = base
        # Match head dtype to base model
        self.head = nn.Linear(base.hidden_size, len(DIRS)).to(dtype=getattr(base, "output_dtype", torch.float32))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)  # (B, T, H)
        # last non-pad token index per batch
        lengths = attention_mask.long().sum(dim=1) - 1  # (B,)
        bsz = hs.size(0)
        h_last = hs[torch.arange(bsz, device=hs.device), lengths]  # (B, H)
        logits = self.head(h_last)  # (B, 6)
        return logits

class QueryTokenProbe(nn.Module):
    """
    Proposed: Query-token Approach (Left-Padding Compatible)
    """
    def __init__(self, base: ProbeModelBase, tokenizer: Any) -> None:
        super().__init__()
        self.base = base
        self.tokenizer = tokenizer

        # safety
        for d, tid in self.token_id.items():
            if tid < 0:
                raise ValueError(f"Special token id not found for {d}: {QUERY_LABEL_STR[d]}")

        target_dtype = getattr(base, "output_dtype", torch.float32)

        self.W = nn.Parameter(torch.empty(len(DIRS), base.hidden_size, dtype=target_dtype))
        self.b = nn.Parameter(torch.zeros(len(DIRS), dtype=target_dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

        self.last_span_found: Optional[Dict[str, Any]] = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)  # (B, T, H)

        found_counts = {d: 0 for d in DIRS}
        total_counts = {d: 0 for d in DIRS}
        all_found = 0
        n_samples = int(input_ids.size(0))

        logits_list = []
        
        # Iterate batch
        for bi in range(n_samples):
            # Extract valid tokens based on attention_mask (handle Left or Right padding)
            # attention_mask is 1 for valid, 0 for pad.
            mask = attention_mask[bi] # (T,)
            valid_indices = torch.nonzero(mask).squeeze(-1) # Indices where mask is 1
            
            # If no valid tokens (edge case?), skip
            if valid_indices.numel() == 0:
                 logits_list.append(torch.zeros(len(DIRS), device=input_ids.device))
                 continue
                 
            # Extract valid sequence
            ids_vec = input_ids[bi, valid_indices] # (Tv,)
            h_vec = hs[bi, valid_indices, :]       # (Tv, H)
            
            ids_list = ids_vec.tolist()
            
            dir_vecs = []
            sample_all_found = True
            
            for d in DIRS:
                total_counts[d] += 1
                tid = self.token_id[d]
                
                # Search from end of valid sequence
                pos = None
                for j in range(len(ids_list) - 1, -1, -1):
                    if ids_list[j] == tid:
                        pos = j
                        break
                
                if pos is None:
                    sample_all_found = False
                    # Fallback: use last token of valid sequence
                    vec = h_vec[-1]
                else:
                    vec = h_vec[pos]
                    found_counts[d] += 1
                dir_vecs.append(vec)

            if sample_all_found:
                all_found += 1

            H = torch.stack(dir_vecs, dim=0)  # (6, H)
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

# ---------------- Training / Eval ----------------

def collate_batch(tokenizer, batch: List[Dict[str, Any]], max_length: int) -> Dict[str, Any]:
    texts = [b["text"] for b in batch]
    ys = torch.stack([b["y"] for b in batch], dim=0)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "y": ys}

@torch.no_grad()
def collect_probs(
    model: nn.Module, dl: DataLoader, device: torch.device, layer_idx: int
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    model.eval()
    ys, ps = [], []
    span_acc = _init_span_acc()
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["y"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, layer_idx=layer_idx)
        _update_span_acc(span_acc, model)
        prob = torch.sigmoid(logits)
        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
    y_true = np.concatenate(ys, axis=0).astype(np.int32)
    p = np.concatenate(ps, axis=0)
    span_summary = _finalize_span_acc(span_acc)
    return y_true, p, span_summary

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
        grid = [i / 100.0 for i in range(5, 96, 5)]  # 0.05..0.95 step 0.05

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
        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    y_pred = (p >= threshold).astype(np.int32)

    metrics = micro_macro_f1(y_true.astype(np.int32), y_pred)
    metrics["threshold"] = threshold
    span_summary = _finalize_span_acc(span_acc)
    if span_summary is not None:
        metrics["span_found"] = span_summary
    return metrics

def train_one_layer(
    *,
    mode: str,  # "baseline" or "query"
    model_name: str,
    train_rows: List[Dict[str, Any]],
    dev_rows: List[Dict[str, Any]],
    filter_spec: FilterSpec,
    layer_idx: int,
    batch_size: int,
    epochs: int,
    lr: float,
    max_length: int,
    seed: int,
    out_dir: Path,
    threshold: float,
    strip_query_in_baseline: bool,
    eval_services: Optional[List[str]],          
    no_tqdm: bool,                               
    tqdm_mininterval: float,
    shared_model: nn.Module,     # NEW
    shared_tokenizer: Any,       # NEW
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use shared tokenizer
    tokenizer = shared_tokenizer

    # Special tokens are already added in main, but we need the IDs here
    special_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
    
    # prepare datasets (optionally strip query for baseline)
    if mode == "baseline" and strip_query_in_baseline:
        train_rows2 = []
        for r in train_rows:
            r2 = dict(r)
            r2["text"] = strip_query_tokens(r2["text"])
            train_rows2.append(r2)
        dev_rows2 = []
        for r in dev_rows:
            r2 = dict(r)
            r2["text"] = strip_query_tokens(r2["text"])
            dev_rows2.append(r2)
        train_rows_use, dev_rows_use = train_rows2, dev_rows2
    else:
        train_rows_use, dev_rows_use = train_rows, dev_rows

    train_ds = JsonlDirUncDataset(train_rows_use, filter_spec)
    dev_ds = JsonlDirUncDataset(dev_rows_use, filter_spec)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(tokenizer, b, max_length),
    )
    dev_dl = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(tokenizer, b, max_length),
    )

    # Instantiate Wrapper
    # Pass shared_model
    # Note: ProbeModelBase init will handle freezing and hooks
    base = ProbeModelBase(
        model_name,
        vocab_size=None, # Already handled in main
        train_token_ids=special_token_ids,
        pretrained_model=shared_model,
    ).to(device) # model already on device, but .to checks internally

    if mode == "baseline":
        model = EosPoolingProbe(base).to(device)
    elif mode == "query":
        model = QueryTokenProbe(base, tokenizer).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Train only head parameters (base is frozen)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr)

    best = {"micro_f1": -1.0, "macro_f1_posonly": -1.0}
    best_path = out_dir / f"best_{mode}_layer{layer_idx}.pt"

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        span_acc_train = _init_span_acc()

        total_batches = len(train_dl)
        for i, batch in enumerate(tqdm(
            train_dl, 
            desc=f"train {mode} layer={layer_idx} ep={ep}", 
            leave=False, disable=no_tqdm, mininterval=tqdm_mininterval)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, layer_idx=layer_idx)
            _update_span_acc(span_acc_train, model)
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
                if math.isnan(val):
                     print(f"WARNING: Loss is NaN at step {i+1}", flush=True)

        train_loss = total_loss / max(1, n)

        # ====== NEW: dev上で threshold をチューニングして評価 ======
        y_true_dev, p_dev, span_summary_dev = collect_probs(model, dev_dl, device, layer_idx)

        tuned = tune_threshold(
            y_true_dev,
            p_dev,
            metric="macro_f1_posonly",   # <- "micro_f1" に変えてもOK
            grid=None,           # <- Noneなら 0.05..0.95 step 0.05（関数側デフォルト）
        )
        dev_metrics = dict(tuned["metrics"])  # includes micro_f1/macro_f1/per_label_f1/threshold

        # span found 率（query モードのときだけ入る想定）
        if span_summary_dev is not None:
            dev_metrics["span_found"] = span_summary_dev

        # どの指標でチューニングしたかもログへ
        dev_metrics["tuned"] = {
            "metric": "macro_f1_posonly",
            "best_threshold": float(tuned["threshold"]),
            "best_score": float(tuned["score"]),
        }

        # train側 span found（あなたの既存処理を維持）
        span_summary_train = _finalize_span_acc(span_acc_train)
        if span_summary_train is not None:
            dev_metrics["span_found_train"] = span_summary_train

        record = {
            "mode": mode,
            "layer_idx": layer_idx,
            "epoch": ep,
            "train_loss": train_loss,
            **dev_metrics,
        }

        # save per-epoch log
        with (out_dir / "log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Print progress for logging
        print(f"[Epoch {ep}/{epochs}] mode={mode} layer={layer_idx} | "
              f"train_loss={train_loss:.4f} | "
              f"macro_f1_posonly={dev_metrics.get('macro_f1_posonly',0.0):.4f} | "
              f"micro_f1={dev_metrics.get('micro_f1',0.0):.4f}")

        # ====== NEW: best 更新基準を「チューニング後macro_f1」に ======
        # もし micro で選びたければ dev_metrics["micro_f1"] に変えてください
        if dev_metrics["macro_f1_posonly"] > best.get("macro_f1_posonly", -1.0):
            best = {**record}
            torch.save(model.state_dict(), best_path)

            # --- 追加: best modelをロードして、全体＆サービス別で評価 ---
    # best_path は学習中に更新されるので、最後に読み直して確定評価
    if best["macro_f1_posonly"] >= 0 and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    best_th = float(best.get("threshold", threshold))
    best_dev_metrics = evaluate(model, dev_dl, device, layer_idx, threshold=best_th)
    best["final_dev"] = best_dev_metrics  # 参考：最終的にbest重みで測った値

    if eval_services:
        best["dev_by_service"] = {}
        for svc in eval_services:
            dev_rows_s = [r for r in dev_rows_use if str(r.get("service", "")) == svc]
            if not dev_rows_s:
                best["dev_by_service"][svc] = {"n_rows": 0}
                continue

            dev_ds_s = JsonlDirUncDataset(dev_rows_s, filter_spec)
            dev_dl_s = DataLoader(
                dev_ds_s,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda b: collate_batch(tokenizer, b, max_length),
            )
            m = evaluate(model, dev_dl_s, device, layer_idx, threshold=best_th)
            m["n_rows"] = len(dev_ds_s)
            best["dev_by_service"][svc] = m

    # CLEANUP here to reuse shared model cleanly
    if 'base' in locals() and hasattr(base, 'teardown'):
        base.teardown()

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
    # Prefer primary metric; fallback to macro_f1; then micro_f1; else -inf
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

    ap.add_argument("--batch_size", type=int, default=16)
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
    score_metric = "macro_f1_posonly"  # <- ここだけ変えれば基準を差し替えられる
    best_overall: Optional[Dict[str, Any]] = None
    best_key: Optional[str] = None

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
    
        # ---- base sweep (layers) ----
        for layer_idx in layers:
            layer_dir = mode_dir / f"layer_{layer_idx}"
            safe_mkdir(layer_dir)
    
            res = train_one_layer(
                mode=mode,
                model_name=args.model_name,
                train_rows=train_rows,
                dev_rows=dev_rows,
                filter_spec=fs,
                layer_idx=layer_idx,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                max_length=args.max_length,
                seed=args.seed,
                out_dir=layer_dir,
                threshold=args.threshold,
                strip_query_in_baseline=args.strip_query_in_baseline,
                eval_services=eval_services,
                no_tqdm=args.no_tqdm,
                tqdm_mininterval=args.tqdm_mininterval,
                shared_model=shared_model,
                shared_tokenizer=tokenizer,
            )
    
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
            refine_layers = expand_best_pm2(int(mode_best["layer_idx"]), num_layers)
            results[f"{mode}/refine_layers"] = refine_layers
    
            for layer_idx in refine_layers:
                if layer_idx in layers:
                    continue  # already done in base sweep
    
                layer_dir = mode_dir / f"layer_{layer_idx}"
                safe_mkdir(layer_dir)
    
                res = train_one_layer(
                    mode=mode,
                    model_name=args.model_name,
                    train_rows=train_rows,
                    dev_rows=dev_rows,
                    filter_spec=fs,
                    layer_idx=layer_idx,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    max_length=args.max_length,
                    seed=args.seed,
                    out_dir=layer_dir,
                    threshold=args.threshold,
                    strip_query_in_baseline=args.strip_query_in_baseline,
                    eval_services=eval_services,
                    no_tqdm=args.no_tqdm,
                    tqdm_mininterval=args.tqdm_mininterval,
                    shared_model=shared_model,
                    shared_tokenizer=tokenizer,
                )
    
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
                    results[f"{mode}/best"] = mode_best
    
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
