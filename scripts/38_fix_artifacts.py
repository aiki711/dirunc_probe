#!/usr/bin/env python3
import json
import torch
import re
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

# Reuse logic from script 36
def load_script_36():
    spec = importlib.util.spec_from_file_location("script_36", "scripts/36_full_generation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

s36 = load_script_36()
GEN_MODEL_ID = s36.GEN_MODEL_ID
generate_natural_missing = s36.generate_natural_missing

DATA_PATH = Path("data/processed/case_grammar/cg_train_natural.jsonl")
TEMP_PATH = Path("data/processed/case_grammar/cg_train_natural_fixed.jsonl")

def get_bad_ids():
    # Use the same regex as script 37 to identify samples to fix
    preps = [r"at", r"on", r"in", r"to", r"from", r"by", r"of", r"with", r"for"]
    patterns = [
        re.compile(r"\b(" + "|".join(preps) + r")\s*[.,!?]", re.IGNORECASE),
        re.compile(r"\b(" + "|".join(preps) + r")\s+(?:at|on|in|to|from|by|of|with|for)\b", re.IGNORECASE)
    ]
    
    bad_ids = set()
    with open(DATA_PATH, "r") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("llm_missing", "")
            if any(p.search(text) for p in patterns):
                bad_ids.add(row["id"])
    return bad_ids

def clean_dangling_prepositions(text: str) -> str:
    # Pattern: preposition + optional space + punctuation mark -> punctuation mark
    preps = ["at", "on", "in", "to", "from", "by", "of", "with", "for"]
    p_regex = re.compile(r"\b(" + "|".join(preps) + r")\s*([.,!?;:])", re.IGNORECASE)
    text = p_regex.sub(r"\2", text)
    
    # Pattern: consecutive prepositions (e.g. "from at") -> first preposition? 
    # Usually better to just keep it or remove both if weird. Let's aim safely:
    # "at on" -> "on" often works better for destination/time.
    double_p = re.compile(r"\b(" + "|".join(preps) + r")\s+(" + "|".join(preps) + r")\b", re.IGNORECASE)
    text = double_p.sub(r"\2", text)

    # Clean double spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def main():
    bad_ids = get_bad_ids()
    print(f"Total problematic samples to fix: {len(bad_ids)}")
    
    # Even if bad_ids is small, we will apply rule-based cleanup to EVERY sample for safety
    fixed_rows = []
    with open(DATA_PATH, "r") as f:
        rows = [json.loads(line) for line in f]

    print("Applying rule-based cleanup and selective re-generation...")
    
    # We only need the model if we still have many bad cases, 
    # but for simplicity let's assume we want to fix remaining bad_ids with LLM first, 
    # then apply rule-based to everything.
    
    # (Skip model loading if not strictly needed or keep for one more pass)
    print(f"Loading Model: {GEN_MODEL_ID} for one final pass on dangling cases...")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    for row in tqdm(rows, desc="Fixing"):
        text = row.get("llm_missing", "")
        
        # 1. LLM fix for known bad IDs
        if row["id"] in bad_ids:
            role = row["metadata"]["case_role"].lower()
            dropped_span = row["metadata"].get("dropped_span", "")
            if "\nUser: " in row["text"]:
                context, filled_text = row["text"].rsplit("\nUser: ", 1)
            else:
                context, filled_text = "None", row["text"]
            text = generate_natural_missing(model, tokenizer, context, filled_text, role, dropped_span)
        
        # 2. Rule-based safety cleanup for EVERYONE
        row["llm_missing"] = clean_dangling_prepositions(text)
        fixed_rows.append(row)

    # Save to temp and then rename
    with open(TEMP_PATH, "w", encoding="utf-8") as out_f:
        # Maintain order
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    
    print(f"Fixed dataset saved to {TEMP_PATH}")

if __name__ == "__main__":
    main()
