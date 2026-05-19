import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())
from scripts.common import DIRS

# ---------------------------------------------------------------------------
# Model Classes (same as 32_train_contrastive_probe.py)
# ---------------------------------------------------------------------------
class ProbeModelBase(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True, torch_dtype=dtype, trust_remote_code=True)
        for p in self.lm.parameters():
            p.requires_grad = False
        self.hidden_size = int(self.lm.config.hidden_size)

    def get_layer_hidden(self, hidden_states, layer_idx: int) -> torch.Tensor:
        idx = layer_idx + 1 if layer_idx >= 0 else len(hidden_states) + layer_idx
        return hidden_states[idx]

class EosPoolingProbe(nn.Module):
    def __init__(self, base: ProbeModelBase):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.hidden_size, len(DIRS)).to(dtype=torch.float32)

    def forward(self, input_ids, attention_mask, layer_idx):
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)
        lengths = attention_mask.long().sum(dim=1) - 1
        h_last = hs[torch.arange(hs.size(0)), lengths]
        return self.head(h_last.to(torch.float32))

class InferenceDataset(Dataset):
    def __init__(self, rows):
        pairs = {}
        for r in rows:
            pid = r["id"].rsplit("::", 1)[0]
            if pid not in pairs: pairs[pid] = {}
            pairs[pid][r["condition"]] = r
        self.pairs = []
        for pid, p in pairs.items():
            if "filled" in p and "missing" in p:
                self.pairs.append({
                    "id": pid,
                    "filled_text": p["filled"]["text"],
                    "missing_text": p["missing"]["text"],
                    "labels": p["missing"]["labels"],
                    "metadata": p["missing"].get("metadata", {}),
                    "dataset": p["missing"].get("dataset", "unknown")
                })
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx): return self.pairs[idx]

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model
    base = ProbeModelBase(args.model_name)
    model = EosPoolingProbe(base)
    model.head.load_state_dict(torch.load(args.probe_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    dataset = InferenceDataset(rows)
    
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), args.batch_size)):
            batch = dataset.pairs[i:i+args.batch_size]
            
            # Prepare inputs
            filled_texts = [p["filled_text"] for p in batch]
            missing_texts = [p["missing_text"] for p in batch]
            
            enc_f = tokenizer(filled_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            enc_m = tokenizer(missing_texts, padding=True, truncation=True, return_tensors="pt").to(device)
            
            logits_f = model(enc_f.input_ids, enc_f.attention_mask, args.layer_idx)
            logits_m = model(enc_m.input_ids, enc_m.attention_mask, args.layer_idx)
            
            probs_f = torch.sigmoid(logits_f).cpu().numpy()
            probs_m = torch.sigmoid(logits_m).cpu().numpy()
            
            for j, p in enumerate(batch):
                res = {
                    "id": p["id"],
                    "case_role": p["metadata"].get("case_role", "unknown"),
                    "is_saturated": p["metadata"].get("is_saturated", False),
                    "strength": p["metadata"].get("strength", "unknown"),
                    "dataset": p["dataset"],
                    "labels": p["labels"],
                    "probs_filled": probs_f[j].tolist(),
                    "probs_missing": probs_m[j].tolist(),
                    "diff": (probs_m[j] - probs_f[j]).tolist()
                }
                results.append(res)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved results to {args.out_path}")

if __name__ == "__main__":
    run_inference()
