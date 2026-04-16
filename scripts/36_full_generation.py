import json
import torch
import re
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

# --- Configuration ---
SOURCE_PATH = Path("data/processed/case_grammar/cg_train.jsonl")
OUTPUT_PATH = Path("data/processed/case_grammar/cg_train_natural.jsonl")
GEN_MODEL_ID = "google/gemma-2-9b-it"

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

def generate_natural_missing(gen_model, gen_tokenizer, context, filled_text, role, dropped_span):
    examples = FEW_SHOT_EXAMPLES.get(role, FEW_SHOT_EXAMPLES["how"])
    few_shot_str = ""
    for ex in examples:
        few_shot_str += f"Filled: {ex['filled']}\nOmit information: {ex['omit']}\nMissing: {ex['missing']}\n\n"

    # Updated Prompt (Iteration 4 Final)
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
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.1)
    
    generated_text = gen_tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    return generated_text.strip().split("\n")[0]

def main():
    # 1. Check existing progress (Checkpointing)
    finished_ids = set()
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    finished_ids.add(json.loads(line)["id"])
                except:
                    continue
    print(f"Resuming from {len(finished_ids)} already generated samples.")

    # 2. Load Missing candidates from train set
    target_tasks = []
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["condition"] == "missing" and row["id"] not in finished_ids:
                target_tasks.append(row)
    
    print(f"Total tasks remaining: {len(target_tasks)}")
    if not target_tasks:
        print("Everything is finished!")
        return

    # 3. Load Model
    print(f"Loading Model: {GEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )

    # 4. Sequential Generation with disk streaming
    with OUTPUT_PATH.open("a", encoding="utf-8") as out_f:
        for row in tqdm(target_tasks, desc="Generating"):
            role = row["metadata"]["case_role"].lower()
            dropped_span = row["metadata"].get("dropped_span", "")
            
            # Simple context extraction from unified text
            # Format: [Context]\nUser: filled_text
            if "\nUser: " in row["text"]:
                context, filled_text = row["text"].rsplit("\nUser: ", 1)
            else:
                context, filled_text = "None", row["text"]

            # Speed-priority: 1 generation, 1 retry if failed
            llm_missing = generate_natural_missing(model, tokenizer, context, filled_text, role, dropped_span)
            
            # Basic validation check
            if (dropped_span.lower() in llm_missing.lower() and len(dropped_span) > 3) or (llm_missing.strip() == filled_text.strip()):
                # One retry
                llm_missing = generate_natural_missing(model, tokenizer, context, filled_text, role, dropped_span)

            # Store result
            row["llm_missing"] = llm_missing
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush() # Ensure it's on disk

    print(f"Production generation completed successfully. Results at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
