import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

import torch
import json
import importlib.util
from transformers import AutoTokenizer
from pathlib import Path
from scripts.common import DIRS, NATURAL_QUERY_STR

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

    # Let's load a few actual validation pairs from paired_dev_gemini_soft.jsonl
    dev_path = Path("data/processed/case_grammar/paired_dev_gemini_soft.jsonl")
    if not dev_path.exists():
        print(f"Error: dev data {dev_path} not found.")
        return

    print("Loading samples...")
    samples = []
    with dev_path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= 6: # load 3 pairs (6 rows)
                break

    # Process pairs
    for i in range(0, len(samples), 2):
        filled_row = samples[i]
        missing_row = samples[i+1]

        pid = missing_row["id"].rsplit("::", 1)[0]
        case_role = missing_row["metadata"].get("case_role", "unknown")
        dropped_span = missing_row["metadata"].get("dropped_span", "unknown")
        labels = missing_row["labels"] # ground truth for missing
        missing_dir = [d for d in DIRS if labels[d] == 1][0]

        # Append NATURAL_QUERY_STR
        text_f = filled_row["text"].strip() + NATURAL_QUERY_STR
        text_m = missing_row["text"].strip() + NATURAL_QUERY_STR

        enc_f = tokenizer([text_f], return_tensors="pt").to(device)
        enc_m = tokenizer([text_m], return_tensors="pt").to(device)

        with torch.no_grad():
            logits_f = model(enc_f["input_ids"], enc_f["attention_mask"], 0)
            logits_m = model(enc_m["input_ids"], enc_m["attention_mask"], 0)

            prob_f = torch.sigmoid(logits_f)[0]
            prob_m = torch.sigmoid(logits_m)[0]

        print(f"\n=========================================")
        print(f"Pair ID      : {pid}")
        print(f"Omitted Slot : {missing_dir} (Role: {case_role}, Span: '{dropped_span}')")
        print(f"-----------------------------------------")
        print(f"{'Class':8s} | {'Prob (Filled)':13s} | {'Prob (Missing)':13s} | {'Diff (M - F)':13s}")
        print(f"-----------------------------------------")
        for idx, d in enumerate(DIRS):
            pf = float(prob_f[idx])
            pm = float(prob_m[idx])
            diff = pm - pf
            marker = "★" if d == missing_dir else " "
            print(f"{d:8s} | {pf:13.4f} | {pm:13.4f} | {diff:+13.4f} {marker}")

if __name__ == "__main__":
    main()
