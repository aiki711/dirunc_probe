#!/usr/bin/env python3
import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import stats
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import importlib.util
def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s32 = load_script_32()
ProbeModelBase = s32.ProbeModelBase
EosPoolingProbe = s32.EosPoolingProbe
PairedDirUncDataset = s32.PairedDirUncDataset
collate_paired_batch = s32.collate_paired_batch
DIRS = s32.DIRS

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 16
    checkpoint_path = "runs/natural_contrastive_probe/best_probe_layer16.pt"
    data_path = "data/processed/case_grammar/natural_dev.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not Path(checkpoint_path).exists():
        print(f"Error: {checkpoint_path} not found.")
        return

    print("Loading models and data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base = ProbeModelBase(model_name).to(device)
    model = EosPoolingProbe(base).to(device)
    model.head.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with open(data_path, "r") as f:
        rows = [json.loads(line) for line in f]
    
    ds = PairedDirUncDataset(rows)
    collate_fn = lambda b: collate_paired_batch(tokenizer, b, max_length=256)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    results = []
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(dl):
            f_ids = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            m_ids = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            y = batch["y"].to(device)

            logits_f = model(f_ids, f_mask, layer_idx)
            logits_m = model(m_ids, m_mask, layer_idx)

            prob_f = torch.sigmoid(logits_f).cpu().numpy()
            prob_m = torch.sigmoid(logits_m).cpu().numpy()
            y_np = y.cpu().numpy()

            for i in range(len(prob_f)):
                # Get the label (which slot was dropped)
                # In contrastive probe, y has 1 at the dropped slot direction
                labels = y_np[i]
                target_idx = np.where(labels == 1)[0]
                if len(target_idx) == 0: continue
                idx = target_idx[0]

                res = {
                    "f_len": len(batch["f_input_ids"][i]),
                    "m_len": len(batch["m_input_ids"][i]),
                    "f_char_len": len(ds.pairs[len(results)]["filled_text"]),
                    "m_char_len": len(ds.pairs[len(results)]["missing_text"]),
                    "f_prob": float(prob_f[i, idx]),
                    "m_prob": float(prob_m[i, idx]),
                    "case_role": batch["case_role"][i],
                    "saturation": batch["saturation_score"][i],
                    "is_saturated": batch["is_saturated"][i]
                }
                results.append(res)

    # --- Quantitative Analysis ---
    print("\n" + "="*30)
    print(" QUANTITATIVE BIAS ANALYSIS ")
    print("="*30)

    m_probs = [r["m_prob"] for r in results]
    m_lens = [r["m_char_len"] for r in results]
    
    rho, p = stats.spearmanr(m_lens, m_probs)
    print(f"Correlation (Length vs Score): {rho:.4f} (p-value: {p:.4g})")

    # Pair Accuracy analysis
    pair_acc = np.mean([1 if r["m_prob"] > r["f_prob"] else 0 for r in results])
    print(f"Overall Pair Accuracy: {pair_acc:.4f}")

    # Subgroup: Length-Controlled (Diff < 10 chars)
    lc_pairs = [r for r in results if abs(r["f_char_len"] - r["m_char_len"]) < 10]
    if lc_pairs:
        lc_acc = np.mean([1 if r["m_prob"] > r["f_prob"] else 0 for r in lc_pairs])
        print(f"Length-Controlled Pair Accuracy (diff < 10 chars, N={len(lc_pairs)}): {lc_acc:.4f}")
    
    # Subgroup: Same Token Length
    tok_pairs = [r for r in results if r["f_len"] == r["m_len"]]
    if tok_pairs:
        tok_acc = np.mean([1 if r["m_prob"] > r["f_prob"] else 0 for r in tok_pairs])
        print(f"Same-Token-Length Pair Accuracy (N={len(tok_pairs)}): {tok_acc:.4f}")

    # --- Saturation Analysis ---
    print("\n" + "="*30)
    print(" SATURATION VS CONFIDENCE ")
    print("="*30)
    
    sat_bins = defaultdict(list)
    for r in results:
        b = round(int(r["saturation"] * 5) / 5, 1)
        sat_bins[b].append(r["m_prob"])

    for b in sorted(sat_bins.keys()):
        avg_prob = np.mean(sat_bins[b])
        print(f"Saturation [{b:.1f}] Average score: {avg_prob:.4f} (N={len(sat_bins[b])})")

if __name__ == "__main__":
    main()
