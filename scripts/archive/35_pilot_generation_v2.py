import json
import random
import torch
import re
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# --- Configuration ---
SOURCE_PATH = Path("data/processed/case_grammar/cg_train.jsonl")
GEN_MODEL_ID = "google/gemma-2-9b-it" 
PPL_MODEL_ID = "google/gemma-2-2b-it"
SAMPLE_COUNT = 30  # per dataset, roughly 100 total

FEW_SHOT_EXAMPLES = {
    "who": [
        {"filled": "I want a regular ride to 100 Smith Ranch Road for 4 people.", "omit": "4 people", "missing": "I want a regular ride to 100 Smith Ranch Road."},
        {"filled": "Can you book it for Hobson's House?", "omit": "Hobson's House", "missing": "Can you book it for me?"}
    ],
    "where": [
        {"filled": "I'll pick it up in Portland, OR.", "omit": "Portland, OR", "missing": "I'll pick it up."},
        {"filled": "A landslide happens when a large amount of soil suddenly falls down a slope.", "omit": "down a slope", "missing": "A landslide happens when a large amount of soil suddenly falls."}
    ],
    "when": [
        {"filled": "Im going to cambridge on thursday", "omit": "thursday", "missing": "Im going to cambridge."},
        {"filled": "The event starts at afternoon 1:15", "omit": "afternoon 1:15", "missing": "The event starts."}
    ],
    "how": [
        {"filled": "I would like an expensive restaurant in the north.", "omit": "expensive", "missing": "I would like a restaurant in the north."},
        {"filled": "Some transformers increase the voltage.", "omit": "Some transformers", "missing": "This increases the voltage."}
    ],
    "what": [
         {"filled": "Italian sounds good. Can you give me an address?", "omit": "Italian", "missing": "That sounds good. Can you give me an address?"},
         {"filled": "The bowling ball is made of matter, which is packed.", "omit": "matter", "missing": "The bowling ball is made of something, which is packed."}
    ]
}

def get_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    if inputs['input_ids'].size(1) <= 1: return 0.0
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return torch.exp(outputs.loss).item()

def generate_natural_missing(gen_model, gen_tokenizer, context, filled_text, role, dropped_span):
    # Select few-shot based on role
    examples = FEW_SHOT_EXAMPLES.get(role, FEW_SHOT_EXAMPLES["how"]) # fallback to how
    few_shot_str = ""
    for ex in examples:
        few_shot_str += f"Filled: {ex['filled']}\nOmit information: {ex['omit']}\nMissing: {ex['missing']}\n\n"

    prompt = f"""You are a linguist assisting in Case Grammar research.
Task: Given a Context and a "Filled" sentence, create a "Missing" version where the specifically mentioned information is omitted.

Constraints:
1. The "Missing" version MUST be perfectly grammatical, natural, and fluent English.
2. The "Missing" version MUST NOT contain the information indicated in "Omit information" ("{dropped_span}").
3. If simply removing "{dropped_span}" leaves a dangling preposition or marker (like "on", "at", "to", "by", "of") that makes the sentence ungrammatical, you MUST remove or adjust those markers as well.
4. You may use indefinite pronouns (e.g., "something", "someone", "it", "then") or adjust the phrase structure to maintain natural flow if necessary.
5. The change should be as minimal as possible while obeying the above rules.
6. Output ONLY the resulting sentence, no explanation.

{few_shot_str}Context Information: {context}
Filled Sentence: {filled_text}
Omit information: {dropped_span} (Role: {role})
Missing Sentence:"""

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    # Use sampling to avoid repetitive failure
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.2, top_p=0.9)
    
    generated_text = gen_tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return generated_text.strip().split("\n")[0]

def self_eval(gen_model, gen_tokenizer, context, filled, missing, role, dropped_span):
    prompt = f"""Rate the following "Missing" sentence on a scale of 1-5 for Naturalness and Information Omission.
Sentence refers to information "{dropped_span}" (Role: {role}) being removed from the "Filled" version.

Filled: {filled}
Missing: {missing}
Information removed: {dropped_span}

Scores (1-5):
- Naturalness (Is it fluent English?):
- Omission (Is the "{dropped_span}" information truly gone?):
- MinimalChange (Did it keep other info?):

Only output the scores in format: Naturalness: X, Omission: Y, MinimalChange: Z"""

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    
    eval_text = gen_tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return eval_text.strip()

def main():
    print("Loading datasets...")
    data_by_ds = defaultdict(list)
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["condition"] == "filled":
                data_by_ds[row["dataset"]].append(row)

    samples = []
    random.seed(42)
    for ds, rows in data_by_ds.items():
        count = min(len(rows), SAMPLE_COUNT)
        samples.extend(random.sample(rows, count))

    # Matching with 'missing' counterparts for PPL comparison
    missing_lookup = {}
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["condition"] == "missing":
                missing_lookup[row["id"].replace("::missing", "")] = row

    print(f"Sampled {len(samples)} entries.")

    print(f"Loading Generation Model: {GEN_MODEL_ID}")
    gen_tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    print(f"Loading PPL Model: {PPL_MODEL_ID}")
    ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_ID)
    ppl_model = AutoModelForCausalLM.from_pretrained(
        PPL_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )


    results = []
    for row in tqdm(samples):
        base_id = row["id"].replace("::filled", "")
        mech_missing = missing_lookup.get(base_id)
        if not mech_missing: continue

        role = row["metadata"]["case_role"].lower()
        dropped_span = row["metadata"].get("dropped_span", "")
        
        context = row["text"].split("\nUser:")[0] if "\nUser:" in row["text"] else "None"
        if row["dataset"] == "qasrl": context = "None"
        
        filled_text = row["text"]
        if "\nUser:" in filled_text: filled_text = filled_text.split("\nUser:")[-1]

        # 1. Generate with Retry Loop
        llm_missing = ""
        best_omission = -1
        for attempt in range(3):
            candidate = generate_natural_missing(gen_model, gen_tokenizer, context, filled_text, role, dropped_span)
            
            # Basic validation: did it actually change something?
            if candidate.lower() == filled_text.lower():
                continue
            
            # Heuristic: does it still contain the forbidden word? (Simple case insensitive check)
            if dropped_span.lower() in candidate.lower() and len(dropped_span) > 3:
                continue
            
            llm_missing = candidate
            break
        
        if not llm_missing: # Fallback if all retries failed
            llm_missing = candidate

        # 2. Evaluate PPL
        ppl_mech = get_ppl(ppl_model, ppl_tokenizer, mech_missing["text"])
        ppl_llm  = get_ppl(ppl_model, ppl_tokenizer, llm_missing)
        ppl_filled = get_ppl(ppl_model, ppl_tokenizer, filled_text)

        # 3. Self Eval
        eval_scores = self_eval(gen_model, gen_tokenizer, context, filled_text, llm_missing, role, dropped_span)

        results.append({
            "id": base_id,
            "dataset": row["dataset"],
            "role": role,
            "dropped_span": dropped_span,
            "context": context,
            "filled": filled_text,
            "mech_missing": mech_missing["text"],
            "llm_missing": llm_missing,
            "ppl_filled": ppl_filled,
            "ppl_mech": ppl_mech,
            "ppl_llm": ppl_llm,
            "eval": eval_scores
        })

    with open("pilot_results_v2.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Pilot v2 completed. Results saved to pilot_results_v2.json")

if __name__ == "__main__":
    main()
