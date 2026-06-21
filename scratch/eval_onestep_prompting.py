#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import importlib.util
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

# Dynamically import PairedDirUncDataset
def load_script_32():
    path = "scripts/32_train_contrastive_probe.py"
    spec = importlib.util.spec_from_file_location("script_32", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s32 = load_script_32()
PairedDirUncDataset = s32.PairedDirUncDataset

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def run_onestep_prompting(model, tokenizer, device, text):
    prompt = f"""You are an assistant analyzing context completeness in dialogue.
Analyze the following text and determine if the last user utterance is complete (Sufficient) or missing crucial information (Insufficient) based on the context.

Text:
\"\"\"
{text}
\"\"\"

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
    parser.add_argument("--out_file", type=str, default="runs/identify_verify_comparison/onestep_results.json")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading datasets and aligning index...")
    dev_cache = torch.load(Path(args.cache_dir) / f"{args.prefix}_layer{args.layer}_dev.pt", map_location="cpu")
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
    y_true_suff = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        if cond == "filled":
            eval_texts.append(pair["filled_text"])
            y_true_suff.append("Sufficient")
        else:
            eval_texts.append(pair["missing_text"])
            y_true_suff.append("Insufficient")
            
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
    
    pred_suff = []
    print("Running One-step Prompting Baseline on 300 eval samples...")
    for text in tqdm(eval_texts):
        pred = run_onestep_prompting(model, tokenizer, device, text)
        pred_suff.append(pred)
        
    acc = accuracy_score(y_true_suff, pred_suff)
    f1 = f1_score(y_true_suff, pred_suff, pos_label="Insufficient")
    
    print(f"\nOne-step Prompting Verify Acc: {acc*100:.2f}% | Omission F1: {f1*100:.2f}%")
    
    # Save results
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({
            "accuracy": float(acc),
            "f1_omission": float(f1)
        }, f, indent=2)
    print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
