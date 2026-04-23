import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np

import importlib.util
spec = importlib.util.spec_from_file_location("train_probe", "scripts/32_train_contrastive_probe.py")
train_probe = importlib.util.module_from_spec(spec)
sys.modules["train_probe"] = train_probe
spec.loader.exec_module(train_probe)

from scripts.common import DIRS
ProbeModelBase = train_probe.ProbeModelBase
EosPoolingProbe = train_probe.EosPoolingProbe
PairedDirUncDataset = train_probe.PairedDirUncDataset
collate_paired_batch = train_probe.collate_paired_batch

# Configuration from log.jsonl
THRESHOLDS = {
    "who": 0.05,
    "what": 0.05,
    "when": 0.05,
    "where": 0.1,
    "why": 0.05,
    "how": 0.05,
    "which": 0.05
}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 16
    checkpoint_path = Path("runs/cg_probe/best_probe_layer16.pt")
    dev_data_path = Path("data/processed/case_grammar/natural_dev_gemini.jsonl")
    out_path = Path("analysis_results/inspection_results.jsonl")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    base = ProbeModelBase(model_name).to(device)
    model = EosPoolingProbe(base).to(device)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
    
    print(f"Loading weights from {checkpoint_path}")
    model.head.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Load Dataset
    print(f"Loading data from {dev_data_path}")
    with dev_data_path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    
    ds = PairedDirUncDataset(rows)
    collate_fn = lambda b: collate_paired_batch(tokenizer, b, max_length=256)
    dl = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    results = []
    
    print("Running inference...")
    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(dl):
            f_ids  = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            m_ids  = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            y      = batch["y"].to(device)
            
            logits_f = model(f_ids, f_mask, layer_idx)
            logits_m = model(m_ids, m_mask, layer_idx)
            
            p_f = torch.sigmoid(logits_f).cpu().numpy()
            p_m = torch.sigmoid(logits_m).cpu().numpy()
            y_true = y.cpu().numpy()
            
            for i in range(len(batch["f_input_ids"])):
                target_indices = np.where(y_true[i] == 1)[0]
                pair = ds.pairs[global_idx]
                
                for idx in target_indices:
                    slot = DIRS[idx]
                    thresh = THRESHOLDS.get(slot, 0.5)
                    val_m = p_m[i, idx]
                    val_f = p_f[i, idx]
                    
                    results.append({
                        "dataset": batch["dataset_name"][i],
                        "case_role": batch["case_role"][i],
                        "slot": slot,
                        "filled_text": pair["filled_text"],
                        "missing_text": pair["missing_text"],
                        "prob_filled": float(val_f),
                        "prob_missing": float(val_m),
                        "confidence_diff": float(val_m - val_f),
                        "is_correct": bool((val_m >= thresh) and (val_f < thresh)),
                        "threshold": thresh
                    })
                global_idx += 1

    print(f"Saving results to {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
