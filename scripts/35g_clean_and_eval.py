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
    print("Loading Pilot Results (v4)...")
    with PILOT_RESULTS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Cleaning data and loading Evaluator Model: {GEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    results = []
    for d in tqdm(data):
        # 1. CLEANING Step
        d["mech_missing_clean"] = clean_mech_missing(d["mech_missing"])
        
        # 2. EVALUATION Step
        filled = d["filled"].strip()
        mech = d["mech_missing_clean"]
        natural = d["llm_missing"].strip()
        role = d["role"]
        dropped_span = d["dropped_span"]
        context = d.get("context", "None")

        # Fair Evaluation (Iteration 4 / v5)
        d["eval_filled_v5"] = eval_version(model, tokenizer, filled, filled, dropped_span, role, context, "Filled")
        d["eval_mech_v5"] = eval_version(model, tokenizer, filled, mech, dropped_span, role, context, "Mechanical")
        d["eval_natural_v5"] = eval_version(model, tokenizer, filled, natural, dropped_span, role, context, "Natural")

        results.append(d)

    with open("pilot_results_v5.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Fair Evaluation (Iteration 4) completed. Results saved to pilot_results_v5.json")

if __name__ == "__main__":
    main()
