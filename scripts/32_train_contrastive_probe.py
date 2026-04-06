import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.append(os.getcwd())
try:
    from scripts.common import DIRS
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_probe", "scripts/03_train_probe.py")
    train_probe = importlib.util.module_from_spec(spec)
    sys.modules["train_probe"] = train_probe
    spec.loader.exec_module(train_probe)
    micro_macro_f1 = train_probe.micro_macro_f1
    tune_threshold_per_class = train_probe.tune_threshold_per_class
    eval_with_per_class_threshold = train_probe.eval_with_per_class_threshold
except Exception as e:
    print(f"Cannot import requirements: {e}")
    sys.exit(1)
    sys.exit(1)

class ProbeModelBase(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
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

        for p in self.lm.parameters():
            p.requires_grad = False

        self.hidden_size = int(self.lm.config.hidden_size)

    def get_layer_hidden(self, hidden_states, layer_idx: int) -> torch.Tensor:
        idx = layer_idx + 1 if layer_idx >= 0 else len(hidden_states) + layer_idx
        idx = max(0, min(idx, len(hidden_states) - 1))
        return hidden_states[idx]

class EosPoolingProbe(nn.Module):
    def __init__(self, base: ProbeModelBase):
        super().__init__()
        self.base = base
        target_dtype = torch.float32 # ensure head is fp32 for stability
        self.head = nn.Linear(base.hidden_size, len(DIRS)).to(dtype=target_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx) # (B, T, H)
        
        # Last non-pad token extraction
        lengths = attention_mask.long().sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)
        bsz = hs.size(0)
        h_last = hs[torch.arange(bsz, device=hs.device), lengths]
        
        return self.head(h_last.to(torch.float32))

class PairedDirUncDataset(Dataset):
    def __init__(self, rows):
        pairs = {}
        for r in rows:
            pid = r["id"].rsplit("::", 1)[0]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][r["condition"]] = r
            
        self.pairs = []
        for pid, p in pairs.items():
            if "filled" in p and "missing" in p:
                self.pairs.append({
                    "filled_text": p["filled"]["text"],
                    "missing_text": p["missing"]["text"],
                    "y_missing": p["missing"]["labels"],
                    "base_id": pid
                })
                
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        item = self.pairs[idx]
        y_vec = torch.tensor([float(item["y_missing"][d]) for d in DIRS], dtype=torch.float32)
        return {
            "filled_text": item["filled_text"],
            "missing_text": item["missing_text"],
            "y": y_vec
        }

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def collate_paired_batch(tokenizer, batch, max_length):
    text_f = [b["filled_text"] for b in batch]
    text_m = [b["missing_text"] for b in batch]
    y = torch.stack([b["y"] for b in batch])
    
    enc_f = tokenizer(text_f, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_m = tokenizer(text_m, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    return {
        "f_input_ids": enc_f["input_ids"],
        "f_attention_mask": enc_f["attention_mask"],
        "m_input_ids": enc_m["input_ids"],
        "m_attention_mask": enc_m["attention_mask"],
        "y": y
    }

@torch.no_grad()
def evaluate(model, dl, device, layer_idx):
    model.eval()
    ys = []
    ps_m = []
    ps_f = []
    
    total_loss = 0.0
    n = 0
    
    for batch in dl:
        f_ids = batch["f_input_ids"].to(device)
        f_mask = batch["f_attention_mask"].to(device)
        m_ids = batch["m_input_ids"].to(device)
        m_mask = batch["m_attention_mask"].to(device)
        y = batch["y"].to(device)
        
        logits_f = model(f_ids, f_mask, layer_idx)
        logits_m = model(m_ids, m_mask, layer_idx)
        
        prob_f = torch.sigmoid(logits_f)
        prob_m = torch.sigmoid(logits_m)
        
        ys.append(y.cpu().numpy())
        ps_m.append(prob_m.cpu().numpy())
        ps_f.append(prob_f.cpu().numpy())
        
        # Loss computation (eval only)
        y_zero = torch.zeros_like(y)
        loss_f = F.binary_cross_entropy_with_logits(logits_f, y_zero)
        loss_m = F.binary_cross_entropy_with_logits(logits_m, y)
        total_loss += float((loss_f + loss_m).item()) * y.size(0)
        n += y.size(0)
            
    if not ys:
        return {}
        
    y_true = np.concatenate(ys, axis=0)         # (N, C)
    p_pred_m = np.concatenate(ps_m, axis=0)       # (N, C)
    p_pred_f = np.concatenate(ps_f, axis=0)       # (N, C)
    
    # Tune Thresholds
    tuned = tune_threshold_per_class(y_true, p_pred_m, grid=None)
    metrics = eval_with_per_class_threshold(y_true, p_pred_m, tuned["thresholds"])
    
    thresholds = np.array(tuned["thresholds"])    # (C,)
    mask = (y_true == 1)                          # (N, C)
    total_active = mask.sum()
    
    if total_active > 0:
        correct_std = (p_pred_m[mask] > p_pred_f[mask]).sum()
        thresh_broadcast = np.broadcast_to(thresholds, y_true.shape)
        correct_str = ((p_pred_m[mask] >= thresh_broadcast[mask]) & (p_pred_f[mask] < thresh_broadcast[mask])).sum()
        
        metrics["pair_accuracy_standard"] = float(correct_std / total_active)
        metrics["pair_accuracy_strict"] = float(correct_str / total_active)
    else:
        metrics["pair_accuracy_standard"] = 0.0
        metrics["pair_accuracy_strict"] = 0.0
    
    metrics["loss"] = total_loss / max(1, n)
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--layer_idx", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--train_data", type=str, required=True, help="Comma separated paths to jsonl")
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="runs/contrastive_probe")
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    base = ProbeModelBase(args.model_name).to(device)
    model = EosPoolingProbe(base).to(device)
    
    train_rows = []
    for path in args.train_data.split(","):
        train_rows.extend(read_jsonl(Path(path.strip())))
        
    dev_rows = []
    for path in args.dev_data.split(","):
        dev_rows.extend(read_jsonl(Path(path.strip())))
        
    train_ds = PairedDirUncDataset(train_rows)
    dev_ds = PairedDirUncDataset(dev_rows)
    
    if len(train_ds) == 0 or len(dev_ds) == 0:
        print("Empty dataset!")
        return
        
    collate_fn = lambda b: collate_paired_batch(tokenizer, b, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    optim = torch.optim.AdamW(model.head.parameters(), lr=args.lr)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"best_probe_layer{args.layer_idx}.pt"
    best_score = -1.0
    
    print(f"Training pairs: {len(train_ds)}, Dev pairs: {len(dev_ds)}")
    
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        
        for batch in tqdm(train_dl, desc=f"Ep {ep}"):
            f_ids = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            m_ids = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            y = batch["y"].to(device)
            
            logits_f = model(f_ids, f_mask, args.layer_idx)
            logits_m = model(m_ids, m_mask, args.layer_idx)
            
            # BCE Loss
            y_zero = torch.zeros_like(y)
            loss_bce_f = F.binary_cross_entropy_with_logits(logits_f, y_zero)
            loss_bce_m = F.binary_cross_entropy_with_logits(logits_m, y)
            
            # Margin Loss
            prob_f = torch.sigmoid(logits_f)
            prob_m = torch.sigmoid(logits_m)
            
            mask = (y == 1)
            if mask.any():
                diff = prob_m[mask] - prob_f[mask]
                loss_margin = torch.mean(F.relu(args.margin - diff))
            else:
                loss_margin = torch.tensor(0.0).to(device)
                
            loss = loss_bce_f + loss_bce_m + args.lambda_margin * loss_margin
            
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            
            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)
            
        train_loss = total_loss / max(1, n)
        
        # Evaluate
        dev_metrics = evaluate(model, dev_dl, device, args.layer_idx)
        print(f"Ep {ep} | Train: {train_loss:.4f} | Dev: {dev_metrics['loss']:.4f} | MacroF1(PosOnly): {dev_metrics.get('macro_f1_posonly', 0):.4f} | StdAcc: {dev_metrics['pair_accuracy_standard']:.4f} | StrictAcc: {dev_metrics['pair_accuracy_strict']:.4f}")
        
        # Model Selection strategy: Standard Pair Accuracy + MacroF1 validation score
        score = dev_metrics['pair_accuracy_standard'] + dev_metrics.get('macro_f1_posonly', 0)
        if score > best_score:
            best_score = score
            torch.save(model.head.state_dict(), best_path)
            
        with (out_dir / "log.jsonl").open("a") as f:
            rec = {"epoch": ep, "train_loss": train_loss, **dev_metrics}
            f.write(json.dumps(rec) + "\n")
            
    print(f"Saved best model to {best_path}")

if __name__ == "__main__":
    main()
