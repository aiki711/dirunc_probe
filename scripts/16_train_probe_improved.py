import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import DIRS, QUERY_LABEL_STR, SPECIAL_TOKENS, QUERY_TOKENS_STR, strip_query_tokens
from common import WHO_KWS, WHEN_KWS, WHERE_KWS, HOW_KWS, WHICH_KWS

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
            if not line: continue
            rows.append(json.loads(line))
    return rows

def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def extract_domain_from_id(sample_id: str) -> Optional[str]:
    parts = sample_id.split("::")
    source = parts[0]
    if source == "sgd" and len(parts) > 3:
        return f"sgd_{parts[3]}"
    elif source == "multiwoz" and len(parts) > 2:
        dom_name = parts[2].replace("domain_", "")
        return f"multiwoz_{dom_name}"
    return None

# ---------------- Keyword Masking Logic ----------------

KW_MAP = {
    "who": WHO_KWS,
    "when": WHEN_KWS,
    "where": WHERE_KWS,
    "how": HOW_KWS,
    "which": WHICH_KWS,
    "what": ["something", "stuff", "detail"] # 'what' is generic
}

def mask_keywords(text: str, rng: random.Random, prob: float = 0.5) -> str:
    """Randomly replaces slot keywords with generic [SLOT] tokens."""
    if rng.random() > prob:
        return text
    
    out = text
    for d, kws in KW_MAP.items():
        # Sort by length descending to catch longer phrases first
        for kw in sorted(kws, key=len, reverse=True):
            if not kw: continue
            # Case-insensitive replacement with word boundaries
            pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
            out = pattern.sub(f"[{d.upper()}]", out)
    return out

# ---------------- Dataset & DataLoader ----------------

class ImprovedDirUncLodoDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        test_domain: str,
        negative_sample_prob: float = 0.2,
        mask_prob: float = 0.0,
        seed: int = 42
    ) -> None:
        self.rows = []
        for r in rows:
            dom = extract_domain_from_id(r.get("id", ""))
            if dom != test_domain:
                self.rows.append(r)
        
        self.negative_sample_prob = negative_sample_prob
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        text = strip_query_tokens(r["text"]).strip()
        labels = r["labels"]
        
        # Keyword Masking (Data Augmentation)
        if self.mask_prob > 0:
            text = mask_keywords(text, self.rng, self.mask_prob)
        
        # Dynamic Negative Sampling
        if self.negative_sample_prob > 0 and self.rng.random() < self.negative_sample_prob:
            text = ""
            labels = {d: 0 for d in DIRS}
            
        y = torch.tensor([float(labels.get(d, 0)) for d in DIRS], dtype=torch.float32)
        return {"text": text, "y": y}

def collate_batch(tokenizer, batch: List[Dict[str, Any]], max_length: int) -> Dict[str, Any]:
    texts = [x["text"] for x in batch]
    labels = torch.stack([x["y"] for x in batch])
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "y": labels}

# ---------------- Model components ----------------

class ProbeModelBase(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=dtype, trust_remote_code=True)
        self.output_dtype = self.lm.dtype
        for p in self.lm.parameters(): p.requires_grad = False
        self.hidden_size = int(self.lm.config.hidden_size)

    def get_layer_hidden(self, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        idx = layer_idx + 1 if layer_idx >= 0 else len(hidden_states) + layer_idx
        return hidden_states[idx]

class FinalTokenProbe(nn.Module):
    def __init__(self, base: ProbeModelBase) -> None:
        super().__init__()
        self.base = base
        target_dtype = getattr(base, "output_dtype", torch.float32)
        self.W = nn.Parameter(torch.empty(len(DIRS), base.hidden_size, dtype=target_dtype))
        self.b = nn.Parameter(torch.zeros(len(DIRS), dtype=target_dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)
        n_samples = input_ids.size(0)
        logits_list = []
        for bi in range(n_samples):
            mask = attention_mask[bi]
            valid_indices = torch.nonzero(mask).squeeze(-1)
            if valid_indices.numel() == 0:
                 logits_list.append(torch.zeros(len(DIRS), device=input_ids.device, dtype=self.W.dtype))
                 continue
            vec = hs[bi, valid_indices[-1], :]
            H = vec.unsqueeze(0).expand(len(DIRS), -1)
            logit = (H * self.W).sum(dim=1) + self.b
            logits_list.append(logit)
        return torch.stack(logits_list, dim=0)

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--data_jsonl", type=str, required=True)
    parser.add_argument("--test_domain", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--neg_sample_prob", type=float, default=0.20)
    parser.add_argument("--mask_prob", type=float, default=0.50)
    parser.add_argument("--pos_weight_mult", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_mkdir(Path(args.save_dir))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(Path(args.data_jsonl))
    dataset = ImprovedDirUncLodoDataset(rows, args.test_domain, args.neg_sample_prob, args.mask_prob, args.seed)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_batch(tokenizer, b, max_length=1024))

    # Calculate Pos Weight for BCE Loss
    all_y = [r["y"] for r in dataset]
    pos_counts = torch.stack(all_y).sum(dim=0)
    neg_counts = len(dataset) - pos_counts
    # Small epsilon to avoid div by zero
    pos_weight = (neg_counts + 1e-6) / (pos_counts + 1e-6)
    pos_weight = pos_weight * args.pos_weight_mult
    print(f"Calculated pos_weight (0:1 ratio) with mult {args.pos_weight_mult}: {pos_weight.tolist()}")

    base = ProbeModelBase(args.model_name)
    base.lm.eval()
    base.to(device)
    probe = FinalTokenProbe(base).to(device)
    probe.train()
    
    optimizer = torch.optim.AdamW([probe.W, probe.b], lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device).to(probe.W.dtype))

    print(f"Starting Balanced Training on {len(dataset)} samples...")
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["y"].to(device).to(probe.W.dtype)

            optimizer.zero_grad()
            logits = probe(input_ids, attention_mask, args.layer_idx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if (step + 1) % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step+1}/{len(dl)} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dl):.4f}")

    out_path = Path(args.save_dir) / f"improved_lodo_layer{args.layer_idx}_{args.test_domain}.pt"
    torch.save({"W": probe.W.detach().cpu(), "b": probe.b.detach().cpu(), "pos_weight": pos_weight}, out_path)
    print(f"Saved improved probe to {out_path}")

if __name__ == "__main__":
    main()
