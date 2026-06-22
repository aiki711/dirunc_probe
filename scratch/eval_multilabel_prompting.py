#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]

ROLE_TO_DIR = {
    "Agent": "who", "Theme": "what", "Location": "where", "Source": "where",
    "Goal": "where", "Time": "when", "Manner": "how"
}

# Mapping of slot name to its index in DIRS (["who", "what", "when", "where", "why", "how", "which"])
SLOT_TO_IDX = {slot: i for i, slot in enumerate(DIRS)}

# Dynamically import PairedDirUncDataset
def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

import importlib.util
s32 = load_script_32()
PairedDirUncDataset = s32.PairedDirUncDataset

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def parse_slots_response(response_str):
    res_clean = response_str.lower()
    if "none" in res_clean:
        return []
    found_slots = []
    # Match candidate slots
    possible_slots = ["who", "what", "when", "where", "how", "which"]
    for slot in possible_slots:
        # Check if slot name exists in response
        if slot in res_clean:
            found_slots.append(slot)
    return found_slots

def run_onestep_multilabel(model, tokenizer, device, text):
    prompt = f"""You are an assistant analyzing context completeness in dialogue.
Identify which of the following semantic slots (who, what, when, where, how, which) are missing or omitted in the last user utterance, based on the conversational context.

If multiple slots are missing, output ALL of them as a comma-separated list (e.g. "when, where").
If no crucial slots are missing (the text is complete), answer "None".

Available choices:
- who: The entity performing the action (e.g. person, customer, agent).
- what: The main object or entity of the action (e.g. phone number, hotel name, food type).
- when: The date, time, day, or duration (e.g. travel time, check-in date).
- where: The location, destination, or origin (e.g. departure city, hotel address).
- how: Quantities, price ranges, payment methods, or ratings (e.g. price range, number of people, number of seats).
- which: Selection categories, choices, amenities (e.g. hotel type, has internet, has parking, star rating).

Text:
\"\"\"
{text}
\"\"\"

Based on the context, output exactly "None" or a comma-separated list of the missing slots. Do not write any other explanation or intro.
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    except:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
    with torch.no_grad():
        gen_out = model.generate(
            input_ids,
            max_new_tokens=20,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_tokens = gen_out[0][input_ids.shape[1]:]
    resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return parse_slots_response(resp)

def run_verify_slot(model, tokenizer, device, text, slot):
    prompt = f"""Text:
\"\"\"
{text}
\"\"\"

Check if the semantic slot "{slot}" is indeed missing or omitted from the last user utterance in the text.
Answer "Sufficient" if the slot is present (filled) or not needed.
Answer "Insufficient" if the slot is truly missing or omitted.
Answer with exactly "Sufficient" or "Insufficient". Do not write any other explanation.
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    except:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
    with torch.no_grad():
        gen_out = model.generate(
            input_ids,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_tokens = gen_out[0][input_ids.shape[1]:]
    resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()
    
    if "insufficient" in resp:
        return "Insufficient"
    else:
        return "Sufficient"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--prefix", type=str, default="final_token_aligned_soft")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--eval_size", type=int, default=300)
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--out_file", type=str, default="runs/identify_verify_comparison/multilabel_prompting_results.json")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading datasets and aligning index...")
    dev_cache = torch.load(Path(args.cache_dir) / f"{args.prefix}_layer{args.layer}_dev.pt", map_location="cpu")
    dev_y = dev_cache["y"].numpy()
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
        dev_pairs = dev_pairs[:dev_cache["f_hs"].shape[0]]
        
    # Sample balanced evaluation set (same seed 42)
    class_groups = {c: [] for c in ALL_CLASSES}
    for i, pair in enumerate(dev_pairs):
        role = pair["case_role"]
        if not role or role not in CASE_ROLES: continue
        mapped_dir = ROLE_TO_DIR[role]
        class_groups["None"].append((i, "filled"))
        class_groups[mapped_dir].append((i, "missing"))
        
    num_per_class = max(1, args.eval_size // 6)
    random.seed(42)
    sampled_items = []
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled = random.sample(idxs, min(len(idxs), num_per_class))
        sampled_items.extend(sampled)
        
    eval_texts = []
    y_true_multilabel = []
    y_true_suff = [] # 1 for Insufficient, 0 for Sufficient
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        if cond == "filled":
            eval_texts.append(pair["filled_text"])
            y_true_multilabel.append(np.zeros(7))
            y_true_suff.append(0)
        else:
            eval_texts.append(pair["missing_text"])
            y_true_multilabel.append(dev_y[idx])
            y_true_suff.append(1)
            
    y_true_multilabel = np.array(y_true_multilabel)
    y_true_suff = np.array(y_true_suff)
            
    print(f"Loading local LLM: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    # 1. Evaluate One-step Prompting (Multi-label)
    onestep_pred_multilabel = []
    print("\n[Evaluating 1/2] One-step Prompting (Multi-label)...")
    for text in tqdm(eval_texts):
        predicted_slots = run_onestep_multilabel(model, tokenizer, device, text)
        pred_vec = np.zeros(7)
        for slot in predicted_slots:
            if slot in SLOT_TO_IDX:
                pred_vec[SLOT_TO_IDX[slot]] = 1
        onestep_pred_multilabel.append(pred_vec)
    onestep_pred_multilabel = np.array(onestep_pred_multilabel)
    
    # Verify: 1 if any slot is missing, 0 otherwise
    onestep_pred_suff = (onestep_pred_multilabel.sum(axis=1) >= 1).astype(int)
    
    # Calculate One-step metrics
    onestep_acc_suff = accuracy_score(y_true_suff, onestep_pred_suff)
    onestep_f1_suff = f1_score(y_true_suff, onestep_pred_suff, pos_label=1)
    onestep_f1_macro_id = f1_score(y_true_multilabel, onestep_pred_multilabel, average="macro")
    
    # 2. Evaluate Identify-then-Verify (Multi-label)
    twostep_pred_multilabel_before_verify = []
    twostep_pred_multilabel = []
    print("\n[Evaluating 2/2] Identify-then-Verify (2-stage Multi-label)...")
    for text in tqdm(eval_texts):
        # Step 1: Identify Candidates (from 1-step prompt)
        candidates = run_onestep_multilabel(model, tokenizer, device, text)
        
        pred_vec_before = np.zeros(7)
        for slot in candidates:
            if slot in SLOT_TO_IDX:
                pred_vec_before[SLOT_TO_IDX[slot]] = 1
        twostep_pred_multilabel_before_verify.append(pred_vec_before)
        
        pred_vec_after = np.zeros(7)
        # Step 2: Verify each candidate
        for slot in candidates:
            verdict = run_verify_slot(model, tokenizer, device, text, slot)
            if verdict == "Insufficient":
                if slot in SLOT_TO_IDX:
                    pred_vec_after[SLOT_TO_IDX[slot]] = 1
        twostep_pred_multilabel.append(pred_vec_after)
        
    twostep_pred_multilabel_before_verify = np.array(twostep_pred_multilabel_before_verify)
    twostep_pred_multilabel = np.array(twostep_pred_multilabel)
    
    twostep_pred_suff = (twostep_pred_multilabel.sum(axis=1) >= 1).astype(int)
    
    # Calculate Two-step metrics
    twostep_acc_suff = accuracy_score(y_true_suff, twostep_pred_suff)
    twostep_f1_suff = f1_score(y_true_suff, twostep_pred_suff, pos_label=1)
    
    # Identify Macro F1: both before and after verification
    twostep_f1_macro_id_before = f1_score(y_true_multilabel, twostep_pred_multilabel_before_verify, average="macro")
    twostep_f1_macro_id_after = f1_score(y_true_multilabel, twostep_pred_multilabel, average="macro")
    
    print("\n" + "="*60)
    print(" UNIFIED MULTI-LABEL COMPARISON RESULTS ")
    print("="*60)
    print("One-step Prompting (Multi-label):")
    print(f"  Verify Accuracy   : {onestep_acc_suff*100:.2f}%")
    print(f"  Verify Omission F1: {onestep_f1_suff*100:.2f}%")
    print(f"  Identify Macro F1 : {onestep_f1_macro_id*100:.2f}%")
    print("-" * 60)
    print("Identify-then-Verify (2-stage Multi-label):")
    print(f"  Verify Accuracy   : {twostep_acc_suff*100:.2f}%")
    print(f"  Verify Omission F1: {twostep_f1_suff*100:.2f}%")
    print(f"  Identify Macro F1 (Before Verify) : {twostep_f1_macro_id_before*100:.2f}%")
    print(f"  Identify Macro F1 (After Verify)  : {twostep_f1_macro_id_after*100:.2f}%")
    print("="*60)
    
    # Save metrics
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "onestep": {
                "verify_accuracy": float(onestep_acc_suff),
                "verify_f1_omission": float(onestep_f1_suff),
                "identify_f1_macro": float(onestep_f1_macro_id)
            },
            "twostep": {
                "verify_accuracy": float(twostep_acc_suff),
                "verify_f1_omission": float(twostep_f1_suff),
                "identify_f1_macro_before_verify": float(twostep_f1_macro_id_before),
                "identify_f1_macro_after_verify": float(twostep_f1_macro_id_after)
            }
        }, f, indent=2)
    print(f"Unified results successfully saved to {out_path}")

if __name__ == "__main__":
    main()
