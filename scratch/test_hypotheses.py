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

def pair_accuracy(y_true, p_pred_m, p_pred_f, thresholds):
    mask = (y_true == 1)
    total = mask.sum()
    if total == 0:
        return 0.0, 0.0
    thresh_bc = np.broadcast_to(thresholds, y_true.shape)
    correct_std = (p_pred_m[mask] > p_pred_f[mask]).sum()
    correct_str = (
        (p_pred_m[mask] >= thresh_bc[mask]) &
        (p_pred_f[mask] <  thresh_bc[mask])
    ).sum()
    return float(correct_std / total), float(correct_str / total)

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

    print("Loading all validation samples...")
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

    print(f"Loaded {len(paired_list)} pairs.")
    paired_list = paired_list[:200]
    print(f"Evaluating on a subset of {len(paired_list)} pairs for speed.")

    scenarios = ["standard", "len_aligned_period", "len_aligned_comma"]
    results = {s: {"ps_m": [], "ps_f": [], "ys": []} for s in scenarios}

    with torch.no_grad():
        for idx, (filled, missing) in enumerate(paired_list):
            if idx % 100 == 0:
                print(f"Processing pair {idx}/{len(paired_list)}...")

            y_vec = np.array([float(missing["labels"][d]) for d in DIRS])
            
            text_f_base = strip_query_tokens(filled["text"]).strip()
            text_m_base = strip_query_tokens(missing["text"]).strip()

            # Tokenize base texts to find length difference
            tok_f = tokenizer.encode(text_f_base, add_special_tokens=False)
            tok_m = tokenizer.encode(text_m_base, add_special_tokens=False)

            diff_len = len(tok_f) - len(tok_m)

            for scenario in scenarios:
                if scenario == "standard":
                    tf = text_f_base + NATURAL_QUERY_STR
                    tm = text_m_base + NATURAL_QUERY_STR
                elif scenario == "len_aligned_period":
                    tf = text_f_base + NATURAL_QUERY_STR
                    if diff_len > 0:
                        # Pad missing sentence with periods to match filled length
                        tm = text_m_base + " ." * diff_len + NATURAL_QUERY_STR
                    else:
                        tm = text_m_base + NATURAL_QUERY_STR
                elif scenario == "len_aligned_comma":
                    tf = text_f_base + NATURAL_QUERY_STR
                    if diff_len > 0:
                        # Pad missing sentence with commas to match filled length
                        tm = text_m_base + " ," * diff_len + NATURAL_QUERY_STR
                    else:
                        tm = text_m_base + NATURAL_QUERY_STR

                enc_f = tokenizer([tf], return_tensors="pt").to(device)
                enc_m = tokenizer([tm], return_tensors="pt").to(device)

                logits_f = model(enc_f["input_ids"], enc_f["attention_mask"], 0)
                logits_m = model(enc_m["input_ids"], enc_m["attention_mask"], 0)

                prob_f = torch.sigmoid(logits_f)[0].float().cpu().numpy()
                prob_m = torch.sigmoid(logits_m)[0].float().cpu().numpy()

                results[scenario]["ps_f"].append(prob_f)
                results[scenario]["ps_m"].append(prob_m)
                results[scenario]["ys"].append(y_vec)

    print("\n--- RESULTS SUMMARY ---")
    for s in scenarios:
        ys = np.array(results[s]["ys"])
        ps_m = np.array(results[s]["ps_m"])
        ps_f = np.array(results[s]["ps_f"])

        # Tune thresholds on standard dev set (or per-scenario)
        # To be consistent with standard evaluation, we can tune thresholds on standard or on the scenario itself.
        # Let's tune thresholds on the scenario itself to see the best possible strict accuracy.
        # We'll use a simple threshold of 0.5 or tune it. Let's tune it.
        tuned = tune_threshold_per_class(ys, ps_m, grid=None)
        thresholds = np.array(tuned["thresholds"])

        std_acc, str_acc = pair_accuracy(ys, ps_m, ps_f, thresholds)
        print(f"Scenario: {s}")
        print(f"  Standard Pair Accuracy: {std_acc:.4%}")
        print(f"  Strict Pair Accuracy:   {str_acc:.4%}")
        print(f"  Thresholds:            {thresholds}")
        print("-" * 30)

if __name__ == "__main__":
    main()
