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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    # We want to pick examples for each case role
    roles = ["Agent", "Time", "Source", "Goal", "Theme", "Manner"]
    correct_samples = {r: [] for r in roles}
    incorrect_samples = {r: [] for r in roles}

    print("Analyzing samples...")
    with torch.no_grad():
        for i, pair in tqdm(enumerate(ds.pairs), total=len(ds.pairs)):
            # Inference for one pair at a time for simplicity of example extraction
            f_inputs = tokenizer(pair["filled_text"], return_tensors="pt", max_length=256, truncation=True).to(device)
            m_inputs = tokenizer(pair["missing_text"], return_tensors="pt", max_length=256, truncation=True).to(device)

            logit_f = model(f_inputs["input_ids"], f_inputs["attention_mask"], layer_idx)[0]
            logit_m = model(m_inputs["input_ids"], f_inputs["attention_mask"], layer_idx)[0] # Note: reuse mask length? no, should use m_inputs

            # Re-do properly
            logit_m = model(m_inputs["input_ids"], m_inputs["attention_mask"], layer_idx)[0]

            prob_f = torch.sigmoid(logit_f).cpu().numpy()
            prob_m = torch.sigmoid(logit_m).cpu().numpy()

            role = pair["case_role"]
            if role not in roles: continue
            
            idx = DIRS.index(role.lower())
            is_correct = prob_m[idx] > prob_f[idx]

            sample_info = {
                "filled": pair["filled_text"],
                "missing": pair["missing_text"],
                "prob_f": float(prob_f[idx]),
                "prob_m": float(prob_m[idx]),
                "base_id": pair["base_id"]
            }

            if is_correct:
                if len(correct_samples[role]) < 2:
                    correct_samples[role].append(sample_info)
            else:
                if len(incorrect_samples[role]) < 2:
                    incorrect_samples[role].append(sample_info)

    # Save results
    output = {"correct": correct_samples, "incorrect": incorrect_samples}
    with open("case_analysis_samples.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSamples saved to case_analysis_samples.json")

if __name__ == "__main__":
    main()
