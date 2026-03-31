import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import sys
import os
import tqdm

sys.path.append(os.getcwd())
from scripts.common import DIRS

def get_final_token_position(attention_mask):
    valid_indices = torch.nonzero(attention_mask[0]).squeeze(-1)
    return valid_indices[-1].item()

def run_probe_final_token(hs, pos, W, b):
    vec = hs[0, pos].to(torch.float32)
    H = vec.unsqueeze(0).expand(len(DIRS), -1)
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

# Keywords and their perturbations for testing
PERTURBATIONS = {
    "tonight": ["sometime", "a certain time"],
    "airport": ["the place", "destination"],
    "two people": ["some people", "guests"],
    "free wifi": ["internet", "connection"],
    "shared ride": ["transport", "option"],
    "Uber": ["the provider", "service"],
    "cheap": ["affordable", "budget"]
}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_final_token/lodo_query_layer8_sgd_Events_1.pt"
    pairs_path = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device).to(torch.bfloat16)
    b = weights["b"].to(device).to(torch.bfloat16)

    with open(pairs_path, "r") as f:
        pairs = json.load(f)

    print("=== Keyword Invariance Baseline Test ===")
    print(f"{'Label':<7} | {'Original B':<10} | {'Perturbed B':<10} | {'Change':<10}")
    print("-" * 50)

    for pair in pairs:
        label = pair["label"]
        orig_text = pair["B"]
        
        # Find keyword to perturb
        p_text = orig_text
        for kw, subs in PERTURBATIONS.items():
            if kw in orig_text:
                p_text = orig_text.replace(kw, subs[0])
                break
        
        if p_text == orig_text:
            continue

        with torch.no_grad():
            # Original B
            enc_o = tokenizer(orig_text, return_tensors="pt").to(device)
            out_o = lm(**enc_o, output_hidden_states=True)
            pos_o = get_final_token_position(enc_o["attention_mask"])
            prob_o = run_probe_final_token(out_o.hidden_states[layer_idx+1], pos_o, W, b)[label]
            
            # Perturbed B
            enc_p = tokenizer(p_text, return_tensors="pt").to(device)
            out_p = lm(**enc_p, output_hidden_states=True)
            pos_p = get_final_token_position(enc_p["attention_mask"])
            prob_p = run_probe_final_token(out_p.hidden_states[layer_idx+1], pos_p, W, b)[label]
            
            delta = prob_p - prob_o
            print(f"{label:<7} | {prob_o:>9.2%} | {prob_p:>9.2%} | {delta:>+9.2%}")

    print("==========================================")

if __name__ == "__main__":
    main()
