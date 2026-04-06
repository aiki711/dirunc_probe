import torch
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to sys.path
sys.path.append(os.getcwd())
from scripts.common import DIRS

def get_final_token_position(attention_mask):
    valid_indices = torch.nonzero(attention_mask[0]).squeeze(-1)
    return valid_indices[-1].item()

def generate_with_patch(model, tokenizer, prompt_text, patch_layer, patch_vector, device, max_new_tokens=20):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    # Hook to apply patch only once during the first forward pass (prompt processing)
    has_patched = False
    
    def hook_fn(module, input, output):
        nonlocal has_patched
        if not has_patched:
            # output is a CausalLMOutputWithPast or similar
            # Hidden states are usually in output.hidden_states if output_hidden_states=True
            # BUT in a standard forward, it's just the last layer output.
            # We need to hook the specific layer.
            
            # The output of a layer is usually (hidden_states, ...)
            hs = output[0] # [B, L, H]
            # Replace final token of prompt
            if patch_vector is not None:
                hs[:, -1, :] = patch_vector.to(hs.dtype)
            has_patched = True
            return (hs,) + output[1:]
        return output

    # Register hook on the specific layer
    # Gemma2-2b-it has layers in model.model.layers
    handle = model.model.layers[patch_layer].register_forward_hook(hook_fn)
    
    try:
        # Generate
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    finally:
        handle.remove()
        
    generated_text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return generated_text

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 16
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
        
    # Select a good example for 'who' (slot for number of people)
    target_label = "who"
    example_pair = None
    for p in pairs:
        if p["label"] == target_label:
            example_pair = p
            break
            
    if not example_pair:
        print(f"Error: No example for {target_label} found.")
        return

    text_A = example_pair["A"] # "I want to reserve a table."
    text_B = example_pair["B"] # "I want to reserve a table for two people."
    
    print(f"\nTarget Slot: {target_label}")
    print(f"Text A (Missing): {text_A}")
    print(f"Text B (Filled):  {text_B}")
    
    # 1. Extract hidden state from B (Filled)
    print("\nTracing Text B to extract Layer 16 state...")
    inputs_B = tokenizer(text_B, return_tensors="pt").to(device)
    with torch.no_grad():
        out_B = model.model(inputs_B["input_ids"], attention_mask=inputs_B["attention_mask"], output_hidden_states=True)
        # Hidden states: 0 is embedding, 1-N are layer outputs
        hs_B = out_B.hidden_states[layer_idx + 1] 
        pos_B = get_final_token_position(inputs_B["attention_mask"])
        patch_vector = hs_B[:, pos_B, :].clone()
        
    # 2. Base Generation A
    print("\n[Run 1] Base Generation (A - Missing):")
    # Note: passing None to generate_with_patch logic would need a check, 
    # but I'll just use standard model.generate for base.
    inputs_A = tokenizer(text_A, return_tensors="pt").to(device)
    out_ids_A = model.generate(**inputs_A, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_base_A = tokenizer.decode(out_ids_A[0][inputs_A["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Result: \"{gen_base_A.strip()}\"")
    
    # 3. Base Generation B
    print("\n[Run 2] Base Generation (B - Filled):")
    inputs_B = tokenizer(text_B, return_tensors="pt").to(device)
    out_ids_B = model.generate(**inputs_B, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    gen_base_B = tokenizer.decode(out_ids_B[0][inputs_B["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Result: \"{gen_base_B.strip()}\"")
    
    # 4. Patched Generation (A patched with B's state)
    print(f"\n[Run 3] Patched Generation (A -> B state at Layer {layer_idx}):")
    gen_patched = generate_with_patch(model, tokenizer, text_A, layer_idx, patch_vector, device)
    print(f"  Result: \"{gen_patched.strip()}\"")

    print("\nAnalysis:")
    if gen_patched.strip() != gen_base_A.strip():
        print("SUCCESS: Patching changed the generation behavior!")
        if any(word in gen_patched.lower() for word in ["restaurant", "time", "where", "what", "day"]):
             print("OBSERVATION: The model seems to have moved on to other slots, behaving as if 'who' is filled.")
    else:
        print("FAILURE: Patching did not change the generation.")

if __name__ == "__main__":
    main()
