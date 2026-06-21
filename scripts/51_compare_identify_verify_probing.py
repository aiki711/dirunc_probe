#!/usr/bin/env python3
import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS

CASE_ROLES = ["Agent", "Theme", "Location", "Source", "Goal", "Time", "Manner"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]

ROLE_TO_DIR = {
    "Agent": "who",
    "Theme": "what",
    "Location": "where",
    "Source": "where",
    "Goal": "where",
    "Time": "when",
    "Manner": "how"
}

class DummyZeroClassifier:
    def fit(self, X, y):
        pass
    def predict(self, X):
        return np.zeros(X.shape[0])
    def predict_proba(self, X):
        res = np.zeros((X.shape[0], 2))
        res[:, 0] = 1.0 # 100% chance of class 0 (Sufficient)
        return res

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

def run_prompting_identify(model, tokenizer, device, text, n_runs=5):
    prompt = f"""You are an assistant analyzing information gaps in dialogue.
Analyze the following text and determine which of the information elements (who, what, when, where, how) is missing/omitted in the last user utterance, based on the conversational context.

Available choices:
- who: The entity performing the action (e.g. agent/person).
- what: The main object or entity of the action (e.g. food type, train booking).
- when: The date, time, or day of the action/travel.
- where: The location, source, or destination of the travel (e.g. area, departing from, going to).
- how: Quantitative details, price ranges, ratings, or methods (e.g. number of people, cheap/expensive, duration).
- None: No crucial information is missing (the sentence is complete).

Text:
\"\"\"
{text}
\"\"\"

Based on the context, which element is missing in the last User utterance? Answer with exactly one of the options: [who, what, when, where, how, None]. Do not write any other explanation.
"""
    outputs = []
    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    except:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
    for _ in range(n_runs):
        with torch.no_grad():
            gen_out = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.5,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        gen_tokens = gen_out[0][input_ids.shape[1]:]
        resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()
        
        # Match response to valid classes
        matched = "none"
        for c in ALL_CLASSES:
            if c.lower() in resp:
                matched = c.lower()
                break
        matched_proper = "None" if matched == "none" else matched
        outputs.append(matched_proper)
        
    # Majority vote
    counts = {}
    for o in outputs:
        counts[o] = counts.get(o, 0) + 1
    consensus = max(counts, key=counts.get)
    return consensus

def run_prompting_verify(model, tokenizer, device, text, consensus_role):
    if consensus_role == "None":
        return "Sufficient"
        
    prompt = f"""Text:
\"\"\"
{text}
\"\"\"

Check if the semantic role "{consensus_role}" is indeed missing or omitted from the last user utterance in the text.
Answer "Sufficient" if the role is present (filled) or not needed.
Answer "Insufficient" if the role is truly missing or omitted.
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
    parser.add_argument("--eval_size", type=int, default=300, help="Total evaluation size (balanced)")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading cached tensors for layer {args.layer}...")
    train_cache = torch.load(Path(args.cache_dir) / f"{args.prefix}_layer{args.layer}_train.pt", map_location="cpu")
    dev_cache = torch.load(Path(args.cache_dir) / f"{args.prefix}_layer{args.layer}_dev.pt", map_location="cpu")
    
    print("Loading original jsonl text rows...")
    dev_rows = read_jsonl(Path("data/processed/case_grammar/natural_dev.jsonl"))
    
    print("Building dev paired dataset...")
    dev_ds = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs
    
    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
        print(f"Warning: Paired dataset size ({len(dev_pairs)}) does not match cache size ({dev_cache['f_hs'].shape[0]}).")
        
    # Group indices by 5W1H query categories (who, what, when, where, how, None)
    class_groups = {c: [] for c in ALL_CLASSES}
    for i, pair in enumerate(dev_pairs):
        role = pair["case_role"]
        if not role or role not in CASE_ROLES:
            continue
            
        mapped_dir = ROLE_TO_DIR[role]
        class_groups["None"].append((i, "filled"))
        class_groups[mapped_dir].append((i, "missing"))
        
    num_per_class = max(1, args.eval_size // 6)
    print(f"Sampling balanced eval set: ~{num_per_class} rows per class...")
    
    random.seed(42)
    sampled_items = []
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        if len(idxs) < num_per_class:
            print(f"Warning: Class {c} has only {len(idxs)} samples, keeping all.")
            sampled_items.extend(idxs)
        else:
            sampled_items.extend(random.sample(idxs, num_per_class))
            
    print(f"Sampled {len(sampled_items)} items for direct comparison.")
    
    # Gather ground truths and prompt inputs
    eval_texts = []
    y_true_role = []
    y_true_suff = []
    
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        if cond == "filled":
            eval_texts.append(pair["filled_text"])
            y_true_role.append("None")
            y_true_suff.append(0) # Sufficient
        else:
            eval_texts.append(pair["missing_text"])
            y_true_role.append(ROLE_TO_DIR[pair["case_role"]])
            y_true_suff.append(1) # Insufficient
            
    # 2. Train 7 independent binary probes on the remaining training cache
    print("Fitting 7 independent query-token binary classifiers...")
    
    train_f_hs = train_cache["f_hs"].float().numpy() # [N_train, 7, D]
    train_m_hs = train_cache["m_hs"].float().numpy() # [N_train, 7, D]
    train_y_labels = train_cache["y"].numpy()        # [N_train, 7]
    N_train = train_f_hs.shape[0]
    
    probes = []
    for d in range(7):
        print(f"  Training binary probe for slot {d} ({DIRS[d]})...")
        X_f = train_f_hs[:, d, :] # [N_train, D]
        X_m = train_m_hs[:, d, :] # [N_train, D]
        X = np.concatenate([X_f, X_m], axis=0) # [2*N_train, D]
        
        y_f = np.zeros(N_train)
        y_m = train_y_labels[:, d]
        y = np.concatenate([y_f, y_m], axis=0)
        
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
            clf.fit(X, y)
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
            clf.fit(X, y)
        probes.append(clf)
        
    # 3. Predict using Probing on identical eval samples
    print("Evaluating Probing on eval set...")
    probe_pred_role = []
    probe_pred_suff = []
    
    dev_f_hs = dev_cache["f_hs"].float().numpy() # [N_dev, 7, D]
    dev_m_hs = dev_cache["m_hs"].float().numpy() # [N_dev, 7, D]
    
    THRESHOLDS = {
        "who": 0.05,
        "what": 0.05,
        "when": 0.05,
        "where": 0.1,
        "why": 0.05,
        "how": 0.05,
        "which": 0.05
    }

    for idx, cond in sampled_items:
        if cond == "filled":
            hs_7 = dev_f_hs[idx]
        else:
            hs_7 = dev_m_hs[idx]
            
        scores = []
        preds = []
        for d in range(7):
            feat = hs_7[d].reshape(1, -1)
            prob = probes[d].predict_proba(feat)[0, 1]
            slot = DIRS[d]
            thresh = THRESHOLDS.get(slot, 0.5)
            
            pred = 1 if prob >= thresh else 0
            score = prob - thresh
            
            preds.append(pred)
            scores.append(score)
            
        if all(p == 0 for p in preds):
            probe_pred_role.append("None")
            probe_pred_suff.append("Sufficient")
        else:
            # Select the slot with the highest confidence margin (prob - threshold) among those that crossed threshold
            valid_slots = [(scores[d], d) for d in range(7) if preds[d] == 1]
            if valid_slots:
                _, d_best = max(valid_slots, key=lambda x: x[0])
                pred_slot = DIRS[d_best]
            else:
                d_best = np.argmax(scores)
                pred_slot = DIRS[d_best]
            
            if pred_slot in ["why", "which"]:
                probe_pred_role.append("None")
                probe_pred_suff.append("Sufficient")
            else:
                probe_pred_role.append(pred_slot)
                probe_pred_suff.append("Insufficient")
                
    # 4. Load Gemma model and run Prompting Baseline
    print(f"Loading local LLM: {args.model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()
    
    prompt_pred_role = []
    prompt_pred_suff = []
    
    print("Running Prompting Baseline (Identify-then-Verify) on eval rows...")
    for text in tqdm(eval_texts):
        consensus_role = run_prompting_identify(model, tokenizer, device, text, n_runs=5)
        prompt_pred_role.append(consensus_role)
        
        if consensus_role == "None":
            prompt_pred_suff.append("Sufficient")
        else:
            suff = run_prompting_verify(model, tokenizer, device, text, consensus_role)
            prompt_pred_suff.append(suff)
            
    # 5. Compute Metrics and Compare
    y_true_suff_str = ["Insufficient" if s == 1 else "Sufficient" for s in y_true_suff]
    
    # Sufficiency (Binary)
    acc_probe_suff = accuracy_score(y_true_suff_str, probe_pred_suff)
    f1_probe_suff = f1_score(y_true_suff_str, probe_pred_suff, pos_label="Insufficient")
    
    acc_prompt_suff = accuracy_score(y_true_suff_str, prompt_pred_suff)
    f1_prompt_suff = f1_score(y_true_suff_str, prompt_pred_suff, pos_label="Insufficient")
    
    # Identify (6-class: 5 roles + None)
    acc_probe_role = accuracy_score(y_true_role, probe_pred_role)
    f1_probe_role = f1_score(y_true_role, probe_pred_role, average="macro")
    
    acc_prompt_role = accuracy_score(y_true_role, prompt_pred_role)
    f1_prompt_role = f1_score(y_true_role, prompt_pred_role, average="macro")
    
    print("\n" + "="*50)
    print(" HEAD-TO-HEAD ACCURACY COMPARISON (QUERY-TOKEN BASED) ")
    print("="*50)
    print(f"Metric\t\t\tProbing\t\tPrompting")
    print("-"*50)
    print(f"Identify Acc (6-Class):\t{acc_probe_role:.4f}\t\t{acc_prompt_role:.4f}")
    print(f"Identify F1 (Macro):\t{f1_probe_role:.4f}\t\t{f1_prompt_role:.4f}")
    print(f"Verify Acc (Binary):\t{acc_probe_suff:.4f}\t\t{acc_prompt_suff:.4f}")
    print(f"Verify F1 (Omission):\t{f1_probe_suff:.4f}\t\t{f1_prompt_suff:.4f}")
    print("="*50)
    
    # Save raw result logs
    results = {
        "eval_size": len(eval_texts),
        "probing": {
            "identify_accuracy": float(acc_probe_role),
            "identify_f1_macro": float(f1_probe_role),
            "sufficiency_accuracy": float(acc_probe_suff),
            "sufficiency_f1_omission": float(f1_probe_suff)
        },
        "prompting": {
            "identify_accuracy": float(acc_prompt_role),
            "identify_f1_macro": float(f1_prompt_role),
            "sufficiency_accuracy": float(acc_prompt_suff),
            "sufficiency_f1_omission": float(f1_prompt_suff)
        }
    }
    
    with (out_dir / "comparison_report.json").open("w") as f:
        json.dump(results, f, indent=2)
        
    # Generate MD table
    with (out_dir / "results.md").open("w") as f:
        f.write("# Head-to-Head Comparison: Probing vs. Prompting (Query-Token Based)\n\n")
        f.write(f"Evaluated on a balanced set of **{len(eval_texts)} samples** from `natural_dev.jsonl` using `google/gemma-2-2b-it`.\n\n")
        f.write("| Evaluation Stage / Metric | Probing (Implicit Query-Token Probes) | Prompting (Explicit 5W1H Identify-then-Verify) | Gap (Probing - Prompting) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Identify Accuracy** (6-Class) | **{acc_probe_role*100:.2f}%** | {acc_prompt_role*100:.2f}% | **{((acc_probe_role - acc_prompt_role)*100):+.2f}%** |\n")
        f.write(f"| **Identify F1** (Macro) | **{f1_probe_role*100:.2f}%** | {f1_prompt_role*100:.2f}% | **{((f1_probe_role - f1_prompt_role)*100):+.2f}%** |\n")
        f.write(f"| **Verify / Sufficiency Accuracy** | **{acc_probe_suff*100:.2f}%** | {acc_prompt_suff*100:.2f}% | **{((acc_probe_suff - acc_prompt_suff)*100):+.2f}%** |\n")
        f.write(f"| **Verify / Sufficiency F1** (Omitted) | **{f1_probe_suff*100:.2f}%** | {f1_prompt_suff*100:.2f}% | **{((f1_probe_suff - f1_prompt_suff)*100):+.2f}%** |\n\n")
        f.write("### Detailed Analysis\n")
        f.write("- **Identify Comparison**: Checks whether the model knows *what* is missing (mapping to the 5 query categories: who, what, when, where, how or None).\n")
        f.write("- **Verify Comparison**: Checks whether the model correctly assesses context completeness (Sufficient vs Insufficient).\n")

    # Plot comparisons
    labels = ['Identify Acc', 'Identify F1', 'Verify Acc', 'Verify F1']
    probing_scores = [acc_probe_role, f1_probe_role, acc_probe_suff, f1_probe_suff]
    prompting_scores = [acc_prompt_role, f1_prompt_role, acc_prompt_suff, f1_prompt_suff]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    rects1 = ax.bar(x - width/2, probing_scores, width, label='Probing (Implicit Query-Token)', color='#004D40')
    rects2 = ax.bar(x + width/2, prompting_scores, width, label='Prompting (Explicit 5W1H)', color='#E65100')
    
    ax.set_ylabel('Scores')
    ax.set_title('Head-to-Head Comparison: Query-Token Probing vs. Prompting\n(Gemma-2-2b-it, Soft Omissions)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_chart.png")
    print("Comparison reports and charts generated successfully.")

if __name__ == "__main__":
    main()
