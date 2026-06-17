#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS, strip_query_tokens
import importlib.util

# Dynamically import components from existing files to ensure architecture compatibility
spec = importlib.util.spec_from_file_location("nq_probe", "scripts/32_train_contrastive_nq_probe.py")
nq_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_probe)
ProbeModelBase = nq_probe.ProbeModelBase

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]

def find_subsequence(sequence, subsequence):
    n, m = len(sequence), len(subsequence)
    for i in range(n - m + 1):
        if sequence[i:i+m] == subsequence:
            return i, i + m
    return None

def locate_span(tokenizer, text, dropped_span):
    # Remove query tokens just in case
    clean_text = strip_query_tokens(text)
    input_ids = tokenizer.encode(clean_text, add_special_tokens=True)
    
    # Try different tokenization formats because SentencePiece is sensitive to spaces
    # 1. With leading space
    span_ids_space = tokenizer.encode(" " + dropped_span.strip(), add_special_tokens=False)
    pos = find_subsequence(input_ids, span_ids_space)
    if pos is not None:
        return pos
        
    # 2. As-is
    span_ids_asis = tokenizer.encode(dropped_span, add_special_tokens=False)
    pos = find_subsequence(input_ids, span_ids_asis)
    if pos is not None:
        return pos
        
    # 3. Trimmed
    span_ids_trim = tokenizer.encode(dropped_span.strip(), add_special_tokens=False)
    pos = find_subsequence(input_ids, span_ids_trim)
    if pos is not None:
        return pos
        
    return None

class SlotDataset(Dataset):
    def __init__(self, rows, tokenizer, limit=0):
        # Resolve filled/missing pairs
        pairs = {}
        for r in rows:
            pid = r["id"].rsplit("::", 1)[0]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][r["condition"]] = r

        self.items = []
        for pid, p in pairs.items():
            if "filled" in p and "missing" in p:
                meta = p["missing"].get("metadata", {})
                role = meta.get("case_role", "")
                dropped_span = meta.get("dropped_span", "")
                
                if not role or role not in CASE_ROLES or not dropped_span:
                    continue
                    
                self.items.append({
                    "filled_text":  p["filled"]["text"],
                    "case_role":    role,
                    "dropped_span": dropped_span,
                    "base_id":      pid,
                    "dataset_name": pid.split("::")[0] if "::" in pid else "unknown"
                })

        if limit > 0 and len(self.items) > limit:
            # Keep it deterministic
            self.items = self.items[:limit]

        # Locate slot spans in tokens
        self.processed = []
        skipped = 0
        
        for item in self.items:
            span_range = locate_span(tokenizer, item["filled_text"], item["dropped_span"])
            if span_range is None:
                skipped += 1
                continue
                
            clean_filled = strip_query_tokens(item["filled_text"])
            
            self.processed.append({
                "filled_text": clean_filled,
                "span_start": span_range[0],
                "span_end": span_range[1],
                "case_role": item["case_role"],
                "dataset_name": item["dataset_name"]
            })
            
        print(f"Dataset loaded: {len(self.processed)} valid slot spans found. Skipped {skipped} spans due to tokenization mismatch.")

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        return self.processed[idx]

def collate_slot_batch(tokenizer, batch):
    texts = [b["filled_text"] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "span_start": torch.tensor([b["span_start"] for b in batch], dtype=torch.long),
        "span_end": torch.tensor([b["span_end"] for b in batch], dtype=torch.long),
        "case_role": [b["case_role"] for b in batch],
        "dataset_name": [b["dataset_name"] for b in batch]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/cache")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,26")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=2500, help="Max training samples to cache")
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base LM: {args.model_name}...")
    base = ProbeModelBase(args.model_name).to(device)
    base.eval()

    def load_rows(path_str):
        rows = []
        for p in path_str.split(","):
            rows.extend(nq_probe.read_jsonl(Path(p.strip())))
        return rows

    train_rows = load_rows(args.train_data)
    dev_rows = load_rows(args.dev_data)

    print("Processing Train Set...")
    train_ds = SlotDataset(train_rows, tokenizer, limit=args.limit)
    print("Processing Dev Set...")
    dev_ds = SlotDataset(dev_rows, tokenizer, limit=0) # Do not limit dev set

    collate_fn = lambda b: collate_slot_batch(tokenizer, b)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_save(dl, split_name):
        print(f"Extracting slot representations for {split_name}...")
        
        meta_list = []
        layer_slot_hs = {L: [] for L in layers}

        for batch in tqdm(dl, desc=split_name):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            span_starts = batch["span_start"]
            span_ends = batch["span_end"]
            
            for bi in range(len(span_starts)):
                meta_list.append({
                    "case_role": batch["case_role"][bi],
                    "dataset_name": batch["dataset_name"][bi],
                })

            with torch.no_grad():
                out = base.lm(input_ids=input_ids, attention_mask=attention_mask)

                for L in layers:
                    hs = base.get_layer_hidden(out.hidden_states, L) # [B, SeqLen, D]
                    
                    batch_slot_hs = []
                    for bi in range(hs.size(0)):
                        start = span_starts[bi].item()
                        end = span_ends[bi].item()
                        # Extract the tokens corresponding to the slot and average them (mean pool)
                        slot_tokens = hs[bi, start:end, :] # [Len, D]
                        slot_avg = slot_tokens.mean(dim=0).cpu() # [D]
                        batch_slot_hs.append(slot_avg)
                        
                    layer_slot_hs[L].append(torch.stack(batch_slot_hs))

        for L in layers:
            slot_tensor = torch.cat(layer_slot_hs[L], dim=0) # [N, D]
            
            # Replicate the structure of other cache files
            # For CRC we only need filled hidden states (f_hs) and metadata
            save_path = out_dir / f"slot_aligned_soft_layer{L}_{split_name}.pt"
            print(f"Saving {save_path} (Tensor Shape: {slot_tensor.shape})...")
            torch.save({
                "f_hs": slot_tensor,
                "metadata": meta_list,
                # Placeholders to match cached datasets
                "m_hs": torch.zeros_like(slot_tensor),
                "y": torch.zeros(slot_tensor.size(0), len(DIRS)) 
            }, save_path)

    extract_and_save(train_dl, "train")
    extract_and_save(dev_dl, "dev")
    print("Caching completed successfully.")

if __name__ == "__main__":
    main()
