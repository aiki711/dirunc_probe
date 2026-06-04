import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

import torch
import torch.nn.functional as F
import json
import importlib.util
from transformers import AutoTokenizer
from pathlib import Path
import numpy as np
from scripts.common import DIRS, NATURAL_QUERY_STR, strip_query_tokens

# Load NQ classes dynamically
spec = importlib.util.spec_from_file_location("nq_probe", "scripts/32_train_contrastive_nq_probe.py")
nq_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_probe)
ProbeModelBase = nq_probe.ProbeModelBase
NaturalQueryProbe = nq_probe.NaturalQueryProbe

def main():
    model_name = "google/gemma-2-2b-it"
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base = ProbeModelBase(model_name).to(device)
    model = NaturalQueryProbe(base, tokenizer).to(device)

    checkpoint_path = Path("runs/layer_sweep_gemini_nq/soft_layer_0/best_probe_layer0.pt")
    if not checkpoint_path.exists():
        print(f"Error: checkpoint {checkpoint_path} not found.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dev_path = Path("data/processed/case_grammar/paired_dev_gemini_soft.jsonl")
    if not dev_path.exists():
        print(f"Error: dev data {dev_path} not found.")
        return

    print("Loading samples...")
    rows = []
    with dev_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    # Group into pairs
    pairs = {}
    for r in rows:
        pid = r["id"].rsplit("::", 1)[0]
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][r["condition"]] = r

    paired_list = []
    for pid, p in pairs.items():
        if "filled" in p and "missing" in p:
            paired_list.append((p["filled"], p["missing"]))

    print(f"Loaded {len(paired_list)} pairs. Selecting first 5 pairs...")
    paired_list = paired_list[:5]

    for idx, (filled, missing) in enumerate(paired_list):
        pid = missing["id"].rsplit("::", 1)[0]
        case_role = missing["metadata"].get("case_role", "unknown")
        dropped_span = missing["metadata"].get("dropped_span", "unknown")
        labels = missing["labels"]
        missing_dir = [d for d in DIRS if labels[d] == 1][0]

        text_f_base = strip_query_tokens(filled["text"]).strip()
        text_m_base = strip_query_tokens(missing["text"]).strip()

        tok_f = tokenizer.encode(text_f_base, add_special_tokens=False)
        tok_m = tokenizer.encode(text_m_base, add_special_tokens=False)
        diff_len = len(tok_f) - len(tok_m)

        print(f"\n==================================================")
        print(f"PAIR {idx+1}: {pid}")
        print(f"Omitted Slot: {missing_dir} (Role: {case_role}, Span: '{dropped_span}')")
        print(f"Filled text length (tokens): {len(tok_f)}")
        print(f"Missing text length (tokens): {len(tok_m)} (diff: {diff_len})")
        print(f"--------------------------------------------------")

        # Scenarios
        scenarios = {
            "Standard": (
                text_f_base + NATURAL_QUERY_STR,
                text_m_base + NATURAL_QUERY_STR
            ),
            "Len-Aligned (Periods)": (
                text_f_base + NATURAL_QUERY_STR,
                text_m_base + " ." * diff_len + NATURAL_QUERY_STR if diff_len > 0 else text_m_base + NATURAL_QUERY_STR
            )
        }

        for sc_name, (tf, tm) in scenarios.items():
            enc_f = tokenizer([tf], return_tensors="pt").to(device)
            enc_m = tokenizer([tm], return_tensors="pt").to(device)

            with torch.no_grad():
                logits_f = model(enc_f["input_ids"], enc_f["attention_mask"], 0)
                logits_m = model(enc_m["input_ids"], enc_m["attention_mask"], 0)
                
                prob_f = torch.sigmoid(logits_f)[0].float().cpu().numpy()
                prob_m = torch.sigmoid(logits_m)[0].float().cpu().numpy()

            print(f"\nScenario: {sc_name}")
            print(f"{'Class':8s} | {'Prob (Filled)':13s} | {'Prob (Missing)':13s} | {'Diff (M - F)':13s}")
            print(f"--------------------------------------------------")
            for c_idx, d in enumerate(DIRS):
                pf = float(prob_f[c_idx])
                pm = float(prob_m[c_idx])
                diff = pm - pf
                marker = "★" if d == missing_dir else " "
                print(f"{d:8s} | {pf:13.4f} | {pm:13.4f} | {diff:+13.4f} {marker}")

if __name__ == "__main__":
    main()
