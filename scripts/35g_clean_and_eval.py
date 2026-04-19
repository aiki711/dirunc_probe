import json
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
PILOT_RESULTS_PATH = Path("pilot_results_v4.json")
GEN_MODEL_ID = "google/gemma-2-9b-it"

def clean_mech_missing(text):
    # If it's the dialogue format, the last "User: " line is the actual utterance
    # If not found, look for "[Target Verb: ...]" and remove it
    if "User:" in text:
        parts = text.split("User:")
        return parts[-1].strip()
    
    # Remove context-prefix like "[Target Verb: verbname]" or "[Domain: ...]"
    text = re.sub(r"\[.*?\]", "", text)
    return text.strip()

def eval_version(gen_model, gen_tokenizer, filled, candidate, dropped_span, role, context, label):
    # Ensure candidate is just the utterance (already cleaned before calling this)
    context_str = f"\nContext:\n{context}\n" if context and context.strip() != "None" else ""
    
    prompt = f"""Rate the following "{label}" sentence on a scale of 1-5 for Naturalness and Information Omission.
Sentence refers to information "{dropped_span}" (Role: {role}) being removed from the "Filled" version.{context_str}
Filled: {filled}
{label}: {candidate}
Information intended to be removed: {dropped_span}

Scores (1-5):
- Naturalness (Is it fluent English? If there is a Context, does it follow the Context naturally?):
- Omission (Is the "{dropped_span}" information truly GONE/MISSING from the "{label}"? 5 means it is gone, 1 means it is still there.):
- MinimalChange (Did it keep other unrelated info and the core nuance?):

Only output the scores in format: Naturalness: X, Omission: Y, MinimalChange: Z"""

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    
    eval_text = gen_tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return eval_text.strip()

def main():
    # 1. Determine input file
    input_path = Path("pilot_results_v4.json")
    if not input_path.exists():
        input_path = Path("pilot_results_v5.json")
    
    if not input_path.exists():
        print(f"Error: Neither pilot_results_v4.json nor pilot_results_v5.json found.")
        return

    print(f"Loading Pilot Results from {input_path}...")
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Setup Evaluator
    print(f"Loading Evaluator Model: {GEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    # 3. Processing with Resumption
    print("Starting evaluation (skipping already evaluated samples)...")
    for d in tqdm(data):
        # Check if already evaluated in v5
        has_v5_eval = d.get("eval_natural_v5") and len(d["eval_natural_v5"].strip()) > 0
        if has_v5_eval:
            continue

        # Cleaning (if not already done)
        if "mech_missing_clean" not in d:
            d["mech_missing_clean"] = clean_mech_missing(d["mech_missing"])
        
        # Evaluation
        filled = d["filled"].strip()
        mech = d["mech_missing_clean"]
        natural = d["llm_missing"].strip()
        role = d["role"]
        dropped_span = d["dropped_span"]
        context = d.get("context", "None")

        d["eval_filled_v5"] = eval_version(model, tokenizer, filled, filled, dropped_span, role, context, "Filled")
        d["eval_mech_v5"] = eval_version(model, tokenizer, filled, mech, dropped_span, role, context, "Mechanical")
        d["eval_natural_v5"] = eval_version(model, tokenizer, filled, natural, dropped_span, role, context, "Natural")

        # Save immediately (Checkpointing)
        with open("pilot_results_v5.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print("Fair Evaluation (v5) completed. Results saved to pilot_results_v5.json")

if __name__ == "__main__":
    main()
