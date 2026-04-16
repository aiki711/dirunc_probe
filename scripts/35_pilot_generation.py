import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# --- Configuration ---
SOURCE_PATH = Path("data/processed/case_grammar/cg_train.jsonl")
GEN_MODEL_ID = "google/gemma-2-9b-it" 
PPL_MODEL_ID = "google/gemma-2-2b-it"
SAMPLE_COUNT = 30  # per dataset, roughly 100 total

def get_ppl(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    if inputs['input_ids'].size(1) <= 1: return 0.0
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return torch.exp(outputs.loss).item()

def generate_natural_missing(gen_model, gen_tokenizer, context, filled_text, role):
    prompt = f"""You are a linguist assisting in Case Grammar research.
Task: Given a Context and a "Filled" sentence, create a "Missing" version where the specific Case Role is omitted.

Constraints:
1. The "Missing" version MUST be perfectly grammatical and natural English.
2. The "Missing" version MUST NOT contain the information from the specified Case Role.
3. The change must be minimal, maintaining original wording where possible.
4. Output ONLY the resulting sentence, no explanation.

Context Information: {context}
Filled Sentence: {filled_text}
Case Role to omit: {role}
Missing Sentence:"""

    inputs = gen_tokenizer(prompt, return_tensors="pt").to(gen_model.device)
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=64, do_sample=False)
    
    generated_text = gen_tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return generated_text.strip().split("\n")[0]

def self_eval(gen_model, gen_tokenizer, context, filled, missing, role):
    prompt = f"""Rate the following "Missing" sentence on a scale of 1-5 for Naturalness and Information Omission.
Sentence refers to a Case Role "{role}" being removed from the "Filled" version.

Filled: {filled}
Missing: {missing}
Role removed: {role}

Scores (1-5):
- Naturalness (Is it fluent English?):
- Omission (Is the {role} information truly gone?):
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
        GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Loading PPL Model: {PPL_MODEL_ID}")
    ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_ID)
    ppl_model = AutoModelForCausalLM.from_pretrained(
        PPL_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    results = []
    for row in tqdm(samples):
        base_id = row["id"].replace("::filled", "")
        mech_missing = missing_lookup.get(base_id)
        if not mech_missing: continue

        role = row["metadata"]["case_role"]
        context = row["text"].split("\nUser:")[0] if "\nUser:" in row["text"] else "None"
        if row["dataset"] == "qasrl": context = "None"
        
        filled_text = row["text"]
        if "\nUser:" in filled_text: filled_text = filled_text.split("\nUser:")[-1]

        # 1. Generate
        llm_missing = generate_natural_missing(gen_model, gen_tokenizer, context, filled_text, role)
        
        # 2. Evaluate PPL
        # We evaluate PPL of the WHOLE prompt if we want, or just the sentence.
        # Let's do just the sentence for pure naturalness.
        ppl_mech = get_ppl(ppl_model, ppl_tokenizer, mech_missing["text"])
        ppl_llm  = get_ppl(ppl_model, ppl_tokenizer, llm_missing)
        ppl_filled = get_ppl(ppl_model, ppl_tokenizer, filled_text)

        # 3. Self Eval
        eval_scores = self_eval(gen_model, gen_tokenizer, context, filled_text, llm_missing, role)

        results.append({
            "id": base_id,
            "dataset": row["dataset"],
            "role": role,
            "context": context,
            "filled": filled_text,
            "mech_missing": mech_missing["text"],
            "llm_missing": llm_missing,
            "ppl_filled": ppl_filled,
            "ppl_mech": ppl_mech,
            "ppl_llm": ppl_llm,
            "eval": eval_scores
        })

    with open("pilot_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Pilot completed. Results saved to pilot_results.json")

if __name__ == "__main__":
    main()
