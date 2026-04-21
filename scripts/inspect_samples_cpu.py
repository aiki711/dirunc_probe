#!/usr/bin/env python3
import os
import sys
import json
import torch
import importlib.util
from tqdm import tqdm
from transformers import AutoTokenizer

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
    
    # Use CPU to avoid CUDA errors during sample inspection
    device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading probe model...")
    base = ProbeModelBase(model_name).to(device)
    model = EosPoolingProbe(base).to(device)
    model.head.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with open(data_path, "r") as f:
        rows = [json.loads(line) for line in f]
    
    ds = PairedDirUncDataset(rows)
    
    roles = ["Agent", "Time", "Source", "Goal", "Theme", "Manner"]
    correct_samples = {r: None for r in roles}
    incorrect_samples = {r: None for r in roles}

    ROLE_MAP = {
        "Agent": "who",
        "Time": "when",
        "Goal": "where",
        "Source": "where",
        "Theme": "what",
        "Manner": "how"
    }

    print("Analyzing samples (one by one until all roles are covered)...")
    with torch.no_grad():
        for i, pair in tqdm(enumerate(ds.pairs), total=len(ds.pairs)):
            role = pair["case_role"]
            if role not in ROLE_MAP: continue
            
            dir_name = ROLE_MAP[role]
            
            # Check if we still need a sample for this role
            if correct_samples[role] and incorrect_samples[role]:
                if all(c is not None for c in correct_samples.values()) and \
                   all(inc is not None for inc in incorrect_samples.values()):
                    break
                continue

            # Inference
            f_inputs = tokenizer(pair["filled_text"], return_tensors="pt", max_length=256, truncation=True).to(device)
            m_inputs = tokenizer(pair["missing_text"], return_tensors="pt", max_length=256, truncation=True).to(device)

            logit_f = model(f_inputs["input_ids"], f_inputs["attention_mask"], layer_idx)[0]
            logit_m = model(m_inputs["input_ids"], m_inputs["attention_mask"], layer_idx)[0]

            prob_f = torch.sigmoid(logit_f).numpy()
            prob_m = torch.sigmoid(logit_m).numpy()

            idx = DIRS.index(dir_name)
            is_correct = prob_m[idx] > prob_f[idx]

            sample_info = {
                "filled": pair["filled_text"],
                "missing": pair["missing_text"],
                "prob_f": round(float(prob_f[idx]), 4),
                "prob_m": round(float(prob_m[idx]), 4),
                "diff": round(float(prob_m[idx] - prob_f[idx]), 4)
            }

            if is_correct and not correct_samples[role]:
                correct_samples[role] = sample_info
            elif not is_correct and not incorrect_samples[role]:
                incorrect_samples[role] = sample_info

    # Final presentation in a nice format
    print("\n" + "="*80)
    print(" CASE ROLE ANALYSIS SAMPLES (Grid View) ")
    print("="*80)
    
    final_output = {"correct": correct_samples, "incorrect": incorrect_samples}
    with open("detailed_case_samples.json", "w") as f:
        json.dump(final_output, f, indent=2)

if __name__ == "__main__":
    main()
