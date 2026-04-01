import torch
import json
import numpy as np
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

PERTURBATIONS = {
    "tonight": ["sometime"],
    "airport": ["the place"],
    "two people": ["some people"],
    "free wifi": ["internet"],
    "shared ride": ["transport"],
    "Uber": ["the provider"],
    "cheap": ["affordable"]
}

def evaluate_model(lm, tokenizer, W, b, layer_idx, pairs, device):
    results = []
    for pair in pairs:
        label = pair["label"]
        res = {"label": label, "pa": 0, "pb": 0, "pb_pert": 0}
        
        with torch.no_grad():
            # A: Missing
            enc_a = tokenizer(pair["A"], return_tensors="pt").to(device)
            out_a = lm(**enc_a, output_hidden_states=True)
            pos_a = get_final_token_position(enc_a["attention_mask"])
            res["pa"] = run_probe_final_token(out_a.hidden_states[layer_idx+1], pos_a, W, b)[label]
            
            # B: Filled
            enc_b = tokenizer(pair["B"], return_tensors="pt").to(device)
            out_b = lm(**enc_b, output_hidden_states=True)
            pos_b = get_final_token_position(enc_b["attention_mask"])
            res["pb"] = run_probe_final_token(out_b.hidden_states[layer_idx+1], pos_b, W, b)[label]
            
            # B_pert: Perturbed (Keyword removed)
            p_text = pair["B"]
            for kw, subs in PERTURBATIONS.items():
                if kw in p_text:
                    p_text = p_text.replace(kw, subs[0])
                    break
            
            if p_text != pair["B"]:
                enc_p = tokenizer(p_text, return_tensors="pt").to(device)
                out_p = lm(**enc_p, output_hidden_states=True)
                pos_p = get_final_token_position(enc_p["attention_mask"])
                res["pb_pert"] = run_probe_final_token(out_p.hidden_states[layer_idx+1], pos_p, W, b)[label]
            else:
                res["pb_pert"] = res["pb"]
                
        results.append(res)
    return results

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    old_path = "runs/balanced/experiment6_final_token/lodo_query_layer8_sgd_Events_1.pt"
    new_path = "runs/improved/improved_lodo_layer8_sgd_Events_1.pt"
    pairs_path = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    with open(pairs_path, "r") as f:
        pairs = json.load(f)

    # 1. Evaluate Old
    print(f"Evaluating OLD model: {old_path}")
    w_old = torch.load(old_path, map_location="cpu")
    res_old = evaluate_model(lm, tokenizer, w_old["W"].to(device), w_old["b"].to(device), layer_idx, pairs, device)
    
    # 2. Evaluate New
    if not Path(new_path).exists():
        print(f"NEW model not found yet at {new_path}. Baseline Evaluation only.")
        res_new = None
    else:
        print(f"Evaluating NEW model: {new_path}")
        w_new = torch.load(new_path, map_location="cpu")
        res_new = evaluate_model(lm, tokenizer, w_new["W"].to(device), w_new["b"].to(device), layer_idx, pairs, device)

    def print_stats(res_list, title):
        total = len(res_list)
        a_hit = sum(1 for r in res_list if r["pa"] > 0.5)
        b_hit = sum(1 for r in res_list if r["pb"] < 0.1)
        # Strict Invariance: B_pert stays low
        inv_hit = sum(1 for r in res_list if r["pb_pert"] < 0.1)
        pair_hit = sum(1 for r in res_list if r["pa"] > 0.5 and r["pb"] < 0.1)
        strict_hit = sum(1 for r in res_list if r["pa"] > 0.8 and r["pb"] < 0.1 and r["pb_pert"] < 0.1)
        
        print(f"\n[{title}]")
        print(f"  A-Score (P(A)>0.5): {a_hit}/{total} ({a_hit/total:.1%})")
        print(f"  B-Score (P(B)<0.1): {b_hit}/{total} ({b_hit/total:.1%})")
        print(f"  Inv-Score (P(B_p)<0.1): {inv_hit}/{total} ({inv_hit/total:.1%})")
        print(f"  Pair Accuracy (A & B): {pair_hit}/{total} ({pair_hit/total:.1%})")
        print(f"  Strict Accuracy: {strict_hit}/{total} ({strict_hit/total:.1%})")

    print_stats(res_old, "OLD BASELINE")
    if res_new:
        print_stats(res_new, "NEW IMPROVED")
        
        print("\n--- Detailed Comparison ---")
        print(f"{'Label':<7} | {'Old (A->B)':<15} | {'New (A->B)':<15} | {'N-Inv.':<7}")
        for ro, rn in zip(res_old, res_new):
            label = ro["label"]
            # Formatting prob as %
            fmt_old = f"{ro['pa']:>4.0%}->{ro['pb']:>4.0%}"
            fmt_new = f"{rn['pa']:>4.0%}->{rn['pb']:>4.0%}"
            inv_n = "OK" if rn["pb_pert"] < 0.1 else f"NG({rn['pb_pert']:>4.0%})"
            print(f"{label:<7} | {fmt_old:<15} | {fmt_new:<15} | {inv_n:<7}")

if __name__ == "__main__":
    main()
