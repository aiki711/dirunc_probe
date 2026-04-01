import argparse
import json
import random
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
            rows.append(json.loads(line))
    return rows

# ---------------- Dataset & DataLoader ----------------

class ContrastivePairedDataset(Dataset):
    """
    Returns pairs of (Text_A, Text_B) where A is Missing and B is Filled.
    """
    def __init__(self, rows: List[Dict[str, Any]], seed: int = 42) -> None:
        self.rows = rows
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        return {
            "text_A": r["text_A"],
            "text_B": r["text_B"],
            "label_idx": r["label_idx"]
        }

def collate_paired_batch(tokenizer, batch: List[Dict[str, Any]], max_length: int) -> Dict[str, Any]:
    texts_A = [x["text_A"] for x in batch]
    texts_B = [x["text_B"] for x in batch]
    label_idxs = torch.tensor([x["label_idx"] for x in batch], dtype=torch.long)
    
    enc_A = tokenizer(texts_A, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_B = tokenizer(texts_B, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    return {
        "input_ids_A": enc_A["input_ids"],
        "mask_A": enc_A["attention_mask"],
        "input_ids_B": enc_B["input_ids"],
        "mask_B": enc_B["attention_mask"],
        "label_idxs": label_idxs
    }

# ---------------- Model components ----------------

class MeanPoolingProbe(nn.Module):
    def __init__(self, model_name: str, n_dirs: int) -> None:
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=dtype, trust_remote_code=True)
        for p in self.lm.parameters(): p.requires_grad = False
        
        self.hidden_size = int(self.lm.config.hidden_size)
        target_dtype = self.lm.dtype
        # Linear layer for all directions
        self.W = nn.Parameter(torch.empty(n_dirs, self.hidden_size, dtype=target_dtype))
        self.b = nn.Parameter(torch.zeros(n_dirs, dtype=target_dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        # Hidden states of the specified layer
        hs = out.hidden_states[layer_idx + 1] # [B, L, H]
        
        # Mean Pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hs.size()).to(hs.dtype)
        sum_hs = torch.sum(hs * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_hs = sum_hs / sum_mask # [B, H]
        
        # Apply probe weights
        H = mean_hs.unsqueeze(1).expand(-1, self.W.size(0), -1) # [B, N, H]
        logits = (H * self.W).sum(dim=2) + self.b # [B, N]
        return logits

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--data_jsonl", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4) # Smaller batch because of pairs
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(Path(args.data_jsonl))
    # Subsample for faster training in this demo if needed
    if len(rows) > 50000:
        rows = random.sample(rows, 50000)
        
    dataset = ContrastivePairedDataset(rows, args.seed)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                    collate_fn=lambda b: collate_paired_batch(tokenizer, b, max_length=512))

    # Import DIRS from common.py
    sys.path.append(os.getcwd())
    from scripts.common import DIRS

    probe = MeanPoolingProbe(args.model_name, len(DIRS)).to(device)
    probe.train()
    
    optimizer = torch.optim.AdamW([probe.W, probe.b], lr=args.learning_rate)
    
    # Loss: BCE for each + Contrastive Pairwise Loss
    bce_criterion = nn.BCEWithLogitsLoss()

    print(f"Starting Accuracy-Focused Training on {len(dataset)} pairs...")
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(dl)):
            ids_A = batch["input_ids_A"].to(device)
            mask_A = batch["mask_A"].to(device)
            ids_B = batch["input_ids_B"].to(device)
            mask_B = batch["mask_B"].to(device)
            label_idxs = batch["label_idxs"].to(device)
            
            optimizer.zero_grad()
            
            # Predict A (Missing=1)
            logits_A = probe(ids_A, mask_A, args.layer_idx)
            # Predict B (Filled=0)
            logits_B = probe(ids_B, mask_B, args.layer_idx)
            
            # 1. Classification Loss (BCE)
            # We only supervise the specific slot being toggled for the contrastive effect
            target_labels_A = torch.zeros_like(logits_A)
            target_labels_B = torch.zeros_like(logits_B)
            for i, l_idx in enumerate(label_idxs):
                target_labels_A[i, l_idx] = 1.0
                target_labels_B[i, l_idx] = 0.0
            
            loss_bce = bce_criterion(logits_A, target_labels_A) + bce_criterion(logits_B, target_labels_B)
            
            # 2. Contrastive Pairwise Loss
            # We want P(A) > P(B) + Margin for the target slot
            target_logits_A = torch.stack([logits_A[i, l_idx] for i, l_idx in enumerate(label_idxs)])
            target_logits_B = torch.stack([logits_B[i, l_idx] for i, l_idx in enumerate(label_idxs)])
            
            # Soft version of margin loss
            # loss_margin = max(0, margin - (logits_A - logits_B))
            # Here we use ReLU for a similar effect
            loss_margin = torch.mean(torch.relu(args.margin - (target_logits_A - target_logits_B)))
            
            total_loss = loss_bce + loss_margin
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

            if (step + 1) % 100 == 0:
                tqdm.write(f"Step {step+1}: Loss {total_loss.item():.4f} (BCE: {loss_bce.item():.4f}, Margin: {loss_margin.item():.4f})")

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / len(dl):.4f}")

    out_path = Path(args.save_dir) / f"contrastive_mean_layer{args.layer_idx}.pt"
    torch.save({"W": probe.W.detach().cpu(), "b": probe.b.detach().cpu()}, out_path)
    print(f"Saved accuracy-focused probe to {out_path}")

if __name__ == "__main__":
    main()
