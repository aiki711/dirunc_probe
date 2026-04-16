import torch
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "google/gemma-2-2b-it"
    pairs_path = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
    ).to(device)
    
    with open(pairs_path, "r") as f:
        pairs = json.load(f)
        
    print(f"\nSearching for high-contrast pairs (Base A != Base B)...")
    
    for i in range(min(40, len(pairs))):
        pair = pairs[i]
        text_A = pair["text_A"]
        text_B = pair["text_B"]
        label = pair["direction"]
        
        # Apply Chat Template
        chat_A = [{"role": "user", "content": text_A}]
        chat_B = [{"role": "user", "content": text_B}]
        prompt_A = tokenizer.apply_chat_template(chat_A, add_generation_prompt=True, tokenize=False)
        prompt_B = tokenizer.apply_chat_template(chat_B, add_generation_prompt=True, tokenize=False)

        # Generate for A
        inputs_A = tokenizer(prompt_A, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids_A = model.generate(**inputs_A, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_A = tokenizer.decode(out_ids_A[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Generate for B
        inputs_B = tokenizer(prompt_B, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids_B = model.generate(**inputs_B, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        gen_B = tokenizer.decode(out_ids_B[0][inputs_B["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Check if responses are significantly different
        # Often the model starts with "Sure! ..." so we check a bit deeper
        if gen_A[:30] != gen_B[:30]:
            print(f"\n[FOUND] Pair ID: {i} (Label: {label})")
            print(f"  Gen A : {gen_A.replace(chr(10), ' ')}")
            print(f"  Gen B : {gen_B.replace(chr(10), ' ')}")
            print(f"  Diff Check : Pass")
        else:
            pass

if __name__ == "__main__":
    main()
