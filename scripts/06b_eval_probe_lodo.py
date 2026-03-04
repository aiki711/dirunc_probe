import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common import DIRS, QUERY_LABEL_STR, QUERY_TOKENS_STR

# ---------------- Utils ----------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line.strip()))
    return rows

def extract_domain_from_id(sample_id: str) -> Optional[str]:
    parts = sample_id.split("::")
    source = parts[0]
    if source == "sgd" and len(parts) > 3:
        return f"sgd_{parts[3]}"
    elif source == "multiwoz" and len(parts) > 2:
        return f"multiwoz_{parts[2]}"
    return None

def find_best_threshold_for_delta(y_true: np.ndarray, delta_p: np.ndarray) -> Dict[str, Any]:
    """Tunes threshold for delta_p which ranges from [-1, 1]"""
    # Create grid from -1.0 to 1.0
    grid = [i / 100.0 for i in range(-100, 101, 5)]
    num_classes = y_true.shape[1]
    
    best_thresholds = []
    best_f1s = []
    
    for class_idx in range(num_classes):
        y_c = y_true[:, class_idx]
        dp_c = delta_p[:, class_idx]
        
        best_f1 = -1.0
        best_th = 0.0 # Default delta threshold
        
        if y_c.sum() == 0:
            best_thresholds.append(0.0)
            best_f1s.append(0.0)
            continue
            
        for th in grid:
            y_pred_c = (dp_c >= th).astype(int)
            
            # Fast F1 calculation
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
        
    support = y_true.sum(axis=0)
    mask = support > 0
    macro_f1_posonly = float(np.mean([best_f1s[i] for i in range(num_classes) if mask[i]])) if mask.any() else 0.0
    
    return {
        "thresholds": [float(th) for th in best_thresholds],
        "per_class_f1": [float(f1) for f1 in best_f1s],
        "macro_f1": float(np.mean(best_f1s)),
        "macro_f1_posonly": macro_f1_posonly,
        "threshold_dict": {DIRS[i]: float(best_thresholds[i]) for i in range(num_classes)},
        "f1_dict": {DIRS[i]: float(best_f1s[i]) for i in range(num_classes)},
    }

def evaluate_with_thresholds(y_true: np.ndarray, delta_p: np.ndarray, thresholds: List[float]) -> Dict[str, Any]:
    num_classes = y_true.shape[1]
    y_pred = np.zeros_like(y_true, dtype=np.int32)
    for i in range(num_classes):
        y_pred[:, i] = (delta_p[:, i] >= thresholds[i]).astype(np.int32)
        
    micro = f1_score(y_true.reshape(-1), y_pred.reshape(-1), average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_label = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    support = y_true.sum(axis=0)
    mask = support > 0
    macro_posonly = f1_score(y_true[:, mask], y_pred[:, mask], average="macro", zero_division=0) if mask.any() else 0.0

    return {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "macro_f1_posonly": float(macro_posonly),
        "per_label_f1": per_label,
        "support": support.tolist()
    }

# ---------------- Dataset & DataLoader ----------------

class EvaluatorDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        y = torch.tensor([float(r["labels"].get(d, 0)) for d in DIRS], dtype=torch.float32)
        return {"text": r["text"], "y": y}

def collate_dual_batch(tokenizer, batch: List[Dict[str, Any]], max_length: int) -> Dict[str, Any]:
    texts_real = [x["text"] for x in batch]
    # Zero-input text (only query tokens)
    texts_null = [QUERY_TOKENS_STR.strip()] * len(batch)
    
    labels = torch.stack([x["y"] for x in batch])
    
    enc_real = tokenizer(texts_real, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_null = tokenizer(texts_null, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    return {
        "real_input_ids": enc_real["input_ids"],
        "real_attention_mask": enc_real["attention_mask"],
        "null_input_ids": enc_null["input_ids"],
        "null_attention_mask": enc_null["attention_mask"],
        "y": labels,
    }

# ---------------- Model components ----------------

class ProbeModelBase(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)

        self.lm = AutoModel.from_pretrained(
            model_name, output_hidden_states=True, torch_dtype=dtype, trust_remote_code=True
        )
        self.output_dtype = self.lm.dtype
        for p in self.lm.parameters(): p.requires_grad = False
        self.hidden_size = int(self.lm.config.hidden_size)

    def get_layer_hidden(self, hidden_states: Tuple[torch.Tensor, ...], layer_idx: int) -> torch.Tensor:
        idx = layer_idx + 1 if layer_idx >= 0 else len(hidden_states) + layer_idx
        return hidden_states[idx]

class QueryTokenProbe(nn.Module):
    def __init__(self, base: ProbeModelBase, tokenizer: Any, weights_path: str) -> None:
        super().__init__()
        self.base = base
        self.token_id = {d: tokenizer.convert_tokens_to_ids(tstr) for d, tstr in QUERY_LABEL_STR.items()}
        
        w_dict = torch.load(weights_path, map_location="cpu")
        self.W = nn.Parameter(w_dict["W"].to(base.output_dtype))
        self.b = nn.Parameter(w_dict["b"].to(base.output_dtype))

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
                tid = self.token_id[d]
                pos = None
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
    parser.add_argument("--data_jsonl", type=str, required=True, help="Full dataset path (contains train/dev/test usually)")
    parser.add_argument("--test_domain", type=str, required=True, help="Domain to evaluate on (e.g. multiwoz_hotel)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained probe weights")
    parser.add_argument("--layer_idx", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_json", type=str, required=True, help="Output JSON for metrics")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=======================================")
    print(f"Experiment 6: LODO Evaluation (Domain: {args.test_domain})")
    print(f"Layer: {args.layer_idx} | Probe: {args.model_path}")
    print(f"=======================================\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    all_rows = read_jsonl(Path(args.data_jsonl))
    
    # Filter ONLY the test domain rows
    test_rows = [r for r in all_rows if extract_domain_from_id(r.get("id", "")) == args.test_domain]
    print(f"Found {len(test_rows)} samples for test domain '{args.test_domain}'")
    
    if len(test_rows) == 0:
        print("No test samples found. Exiting.")
        return

    # In a rigorous setup, we might tune threshold on a held-out dev set form OTHER domains, 
    # but here we tune on the test domain itself to see its upper bound capacity or 
    # ideally split test_rows into dev/test. Let's split 50/50 for tuning/scoring.
    random.seed(42)
    random.shuffle(test_rows)
    split_idx = len(test_rows) // 2
    dev_rows = test_rows[:split_idx]
    eval_rows = test_rows[split_idx:]

    def get_predictions(rows: List[Dict[str, Any]], probe: QueryTokenProbe) -> Tuple[np.ndarray, np.ndarray]:
        ds = EvaluatorDataset(rows)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, 
                        collate_fn=lambda b: collate_dual_batch(tokenizer, b, max_length=1024))
        
        y_true, delta_p_list = [], []
        probe.eval()
        with torch.no_grad():
            for batch in tqdm(dl, desc="Evaluating"):
                y = batch["y"].to(device)
                
                # Real text inference
                logits_real = probe(batch["real_input_ids"].to(device), batch["real_attention_mask"].to(device), args.layer_idx)
                p_real = torch.sigmoid(logits_real)
                
                # Null text inference (Bias baseline)
                logits_null = probe(batch["null_input_ids"].to(device), batch["null_attention_mask"].to(device), args.layer_idx)
                p_null = torch.sigmoid(logits_null)
                
                # Information Gain (Delta P)
                delta_p = p_real - p_null
                
                y_true.append(y.to(torch.float32).cpu().numpy())
                delta_p_list.append(delta_p.to(torch.float32).cpu().numpy())
                
        return np.concatenate(y_true, axis=0), np.concatenate(delta_p_list, axis=0)

    print("Initializing model...")
    base = ProbeModelBase(args.model_name).to(device)
    base.lm.eval()
    probe = QueryTokenProbe(base, tokenizer, args.model_path).to(device)
    
    print("\n[Phase 1] Tuning Thresholds on Dev Split...")
    y_dev, dp_dev = get_predictions(dev_rows, probe)
    tune_res = find_best_threshold_for_delta(y_dev, dp_dev)
    best_thresholds = tune_res["thresholds"]
    print(f"Tuned Thresholds (Delta P): {tune_res['threshold_dict']}")
    print(f"Dev Split Macro F1 (pos only): {tune_res['macro_f1_posonly']:.4f}")

    print("\n[Phase 2] Evaluating on Test Split...")
    y_test, dp_test = get_predictions(eval_rows, probe)
    final_metrics = evaluate_with_thresholds(y_test, dp_test, best_thresholds)
    
    print(f"Test Split Macro F1 (pos only): {final_metrics['macro_f1_posonly']:.4f}")
    
    # Save final results
    out_dict = {
        "domain": args.test_domain,
        "samples_tuned": len(dev_rows),
        "samples_evaluated": len(eval_rows),
        "thresholds_used": {DIRS[i]: best_thresholds[i] for i in range(len(DIRS))},
        "metrics": final_metrics
    }
    
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
         json.dump(out_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {args.out_json}")

if __name__ == "__main__":
    main()
