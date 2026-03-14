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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from common import DIRS, NATURAL_QUERY_MAP, NATURAL_QUERY_STR, strip_query_tokens

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

# ---------------- Dataset & DataLoader ----------------

class JsonlDirUncNQDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        test_domain: str,
        negative_sample_prob: float = 0.2,
        seed: int = 42
    ) -> None:
        self.rows = []
        for r in rows:
            dom = extract_domain_from_id(r.get("id", ""))
            if dom != test_domain:
                self.rows.append(r)
        
        self.negative_sample_prob = negative_sample_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        # Ensure we strip existing special tokens and append natural language queries
        text = strip_query_tokens(r["text"]).strip() + NATURAL_QUERY_STR
        labels = r["labels"]
        
        if self.negative_sample_prob > 0 and self.rng.random() < self.negative_sample_prob:
            text = NATURAL_QUERY_STR.strip()
            labels = {d: 0 for d in DIRS}
            
        y = torch.tensor([float(labels.get(d, 0)) for d in DIRS], dtype=torch.float32)
        return {"text": text, "y": y, "meta": r}

def collate_batch(tokenizer, batch: List[Dict[str, Any]], max_length: int) -> Dict[str, Any]:
    texts = [x["text"] for x in batch]
    labels = torch.stack([x["y"] for x in batch])
    
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

# ---------------- Model components ----------------

class ProbeModelBase(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.lm = AutoModel.from_pretrained(
            model_name, 
            output_hidden_states=True, 
            torch_dtype=dtype,
            trust_remote_code=True
        )
        self.output_dtype = self.lm.dtype
        for p in self.lm.parameters():
            p.requires_grad = False
        self.hidden_size = int(self.lm.config.hidden_size)

    def get_layer_hidden(self, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        idx = layer_idx + 1 if layer_idx >= 0 else len(hidden_states) + layer_idx
        return hidden_states[idx]

class NaturalQueryProbe(nn.Module):
    """Refers to the '?' token hidden state in 'Who?', 'What?', etc."""
    def __init__(self, base: ProbeModelBase, tokenizer: Any) -> None:
        super().__init__()
        self.base = base
        
        self.query_token_ids = {}
        for d, qstr in NATURAL_QUERY_MAP.items():
            # tokenize the natural query "Who?" and find the ID for "?" (usually the last token)
            ids = tokenizer.encode(qstr, add_special_tokens=False)
            # In Gemma, "Who?" -> ["Who", "?"]
            self.query_token_ids[d] = ids[-1] # Target the '?'

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
                 logits_list.append(torch.zeros(len(DIRS), device=input_ids.device))
                 continue
                 
            ids_list = input_ids[bi, valid_indices].tolist()
            h_vec = hs[bi, valid_indices, :]
            
            dir_vecs = []
            for d in DIRS:
                tid = self.query_token_ids[d]
                # Find the position of '?' for this query from the end
                pos = None
                # We search for the occurrence that matches the natural sequence "Who?" (ids_list)
                # But to keep it simple and robust, we look for the expected tid in the tail area
                # Since we appended " Who? What? ...", they should be near the end.
                for j in range(len(ids_list) - 1, -1, -1):
                    if ids_list[j] == tid:
                        pos = j
                        break
                
                vec = h_vec[pos] if pos is not None else h_vec[-1]
                dir_vecs.append(vec)
            
            H = torch.stack(dir_vecs, dim=0)
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_mkdir(Path(args.save_dir))

    print(f"=======================================")
    print(f"Experiment 6f: Natural Query LODO Training (Domain: {args.test_domain})")
    print(f"=======================================\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    rows = read_jsonl(Path(args.data_jsonl))
    dataset = JsonlDirUncNQDataset(rows, args.test_domain, args.neg_sample_prob, args.seed)
    
    dl = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_batch(tokenizer, b, max_length=1024)
    )

    base = ProbeModelBase(args.model_name)
    base.lm.eval()
    base.to(device)
    
    probe = NaturalQueryProbe(base, tokenizer).to(device)
    probe.train()
    
    optimizer = torch.optim.AdamW([probe.W, probe.b], lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_weights = None

    total_steps = len(dl)
    print_interval = max(1, total_steps // 5)

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad()
            logits = probe(input_ids, attention_mask, args.layer_idx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if (step + 1) % print_interval == 0 or (step + 1) == total_steps:
                pct = ((step + 1) / total_steps) * 100
                print(f"Epoch {epoch+1} | Step {step+1}/{total_steps} ({pct:.0f}%) | Loss: {loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / total_steps
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = {"W": probe.W.detach().cpu(), "b": probe.b.detach().cpu()}

    out_path = Path(args.save_dir) / f"lodo_nq_layer{args.layer_idx}_{args.test_domain}.pt"
    torch.save(best_weights, out_path)
    print(f"Saved optimized probe to {out_path}")

if __name__ == "__main__":
    main()
