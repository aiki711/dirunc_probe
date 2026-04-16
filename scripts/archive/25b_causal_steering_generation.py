import torch
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
    ).to(device)
    
    # ---------------------------------------------------------
    # Selecting a high-contrast pair manually based on dataset
    # ---------------------------------------------------------
    # Context A: Restaurant name is missing
    text_A = "User: I want to find a diner in San Carlos. Assistant: I found 3 options. Maybe you'd like to go. User: That sounds nice. Let's make a reservation please."
    # Context B: Restaurant name is present
    text_B = "User: I want to find a diner in San Carlos. Assistant: I found 3 options. Maybe you'd like to go to Piacere Restaurant. User: That sounds nice. Let's make a reservation please."
    
    chat_A = [{"role": "user", "content": text_A}]
    chat_B = [{"role": "user", "content": text_B}]
    
    prompt_A = tokenizer.apply_chat_template(chat_A, add_generation_prompt=True, tokenize=False)
    prompt_B = tokenizer.apply_chat_template(chat_B, add_generation_prompt=True, tokenize=False)
    
    print(f"\nPrompt A (Missing):\n{prompt_A}")
    print(f"\nPrompt B (Filled):\n{prompt_B}")
    
    # 1. Extract hidden states
    inputs_A = tokenizer(prompt_A, return_tensors="pt").to(device)
    inputs_B = tokenizer(prompt_B, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out_A = model.model(inputs_A["input_ids"], attention_mask=inputs_A["attention_mask"], output_hidden_states=True)
        out_B = model.model(inputs_B["input_ids"], attention_mask=inputs_B["attention_mask"], output_hidden_states=True)
        
        # Layer output is index layer_idx + 1
        h_A = out_A.hidden_states[layer_idx + 1][:, -1, :].clone()
        h_B = out_B.hidden_states[layer_idx + 1][:, -1, :].clone()
        
    steering_vec = h_B - h_A
    print(f"Steering vector norm: {torch.norm(steering_vec).item():.4f}")

    # 2. Define Steering Function
    def generate_steered(alpha):
        has_patched = False
        def hook_fn(module, input, output):
            nonlocal has_patched
            if not has_patched:
                hs = output[0]
                # Steering: A + alpha * (B - A)
                hs[:, -1, :] = h_A + alpha * steering_vec
                has_patched = True
                return (hs,) + output[1:]
            return output

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        try:
            out_ids = model.generate(
                **inputs_A,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        finally:
            handle.remove()
        
        return tokenizer.decode(out_ids[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # 3. Base Generations
    print("\n[Base A] (No Steering):")
    with torch.no_grad():
        base_A_ids = model.generate(**inputs_A, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(base_A_ids[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        
    print("\n[Base B] (Full Filled Context):")
    with torch.no_grad():
        base_B_ids = model.generate(**inputs_B, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(base_B_ids[0][inputs_B["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    # 4. Steered Generations
    alphas = [1.0, 2.0, 4.0]
    for alpha in alphas:
        print(f"\n[Steered A] (alpha={alpha}):")
        result = generate_steered(alpha)
        print(result)

if __name__ == "__main__":
    main()
