import torch
import json
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "google/gemma-2-2b-it"
    # Steering at multiple layers to make the "Information Presence" robust
    layers_to_steer = [12, 14, 16, 18, 20]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
    ).to(device)
    
    # ---------------------------------------------------------
    # High-contrast pair (Restaurant Name)
    # ---------------------------------------------------------
    text_A = "User: I want to find a diner in San Carlos. Assistant: I found 3 options. Maybe you'd like to go. User: That sounds nice. Let's make a reservation please."
    text_B = "User: I want to find a diner in San Carlos. Assistant: I found 3 options. Maybe you'd like to go to Piacere Restaurant. User: That sounds nice. Let's make a reservation please."
    
    chat_A = [{"role": "user", "content": text_A}]
    chat_B = [{"role": "user", "content": text_B}]
    prompt_A = tokenizer.apply_chat_template(chat_A, add_generation_prompt=True, tokenize=False)
    prompt_B = tokenizer.apply_chat_template(chat_B, add_generation_prompt=True, tokenize=False)
    
    inputs_A = tokenizer(prompt_A, return_tensors="pt").to(device)
    inputs_B = tokenizer(prompt_B, return_tensors="pt").to(device)
    
    # 1. Extract hidden states for all target layers
    steering_vectors = {}
    with torch.no_grad():
        out_A = model.model(inputs_A["input_ids"], attention_mask=inputs_A["attention_mask"], output_hidden_states=True)
        out_B = model.model(inputs_B["input_ids"], attention_mask=inputs_B["attention_mask"], output_hidden_states=True)
        
        for l in layers_to_steer:
            h_A = out_A.hidden_states[l + 1][:, -1, :].clone()
            h_B = out_B.hidden_states[l + 1][:, -1, :].clone()
            steering_vectors[l] = h_B - h_A
            print(f"Layer {l} Steering vec norm: {torch.norm(steering_vectors[l]).item():.4f}")

    # 2. Define Multi-Layer Steering Function
    def generate_multi_steer(alpha):
        handles = []
        # We need to track patching for each layer separately during the prompt processing
        patched_count = {l: 0 for l in layers_to_steer}
        
        def create_hook(l):
            def hook_fn(module, input, output):
                if patched_count[l] == 0: # Only patch during the first forward (prompt passing)
                    hs = output[0]
                    # Steering: h_A + alpha * (h_B - h_A)
                    # Note: input h_A is already in the stream, so we just add alpha*diff
                    hs[:, -1, :] = hs[:, -1, :] + alpha * steering_vectors[l]
                    patched_count[l] += 1
                    return (hs,) + output[1:]
                return output
            return hook_fn

        for l in layers_to_steer:
            h = model.model.layers[l].register_forward_hook(create_hook(l))
            handles.append(h)
            
        try:
            out_ids = model.generate(
                **inputs_A,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        finally:
            for h in handles:
                h.remove()
        
        return tokenizer.decode(out_ids[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # 3. Generations
    print("\n[Base A] (No Steering):")
    with torch.no_grad():
        base_A_ids = model.generate(**inputs_A, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(base_A_ids[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True).strip())
        
    print("\n[Base B] (Filled):")
    with torch.no_grad():
        base_B_ids = model.generate(**inputs_B, max_new_tokens=40, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(base_B_ids[0][inputs_B["input_ids"].shape[1]:], skip_special_tokens=True).strip())

    alphas = [2.0, 5.0, 10.0]
    for alpha in alphas:
        print(f"\n[Multi-Steered A] (alpha={alpha}):")
        result = generate_multi_steer(alpha)
        print(result)

if __name__ == "__main__":
    main()
