import json
import torch
from transformers import AutoTokenizer, AutoModel
import sys
import os
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.getcwd())
from scripts.common import DIRS

def run_probe_mean_pooling(hidden_states, attention_mask, W, b):
    # hidden_states: [B, L, H]
    # attention_mask: [B, L]
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.dtype)
    sum_hs = torch.sum(hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_hs = sum_hs / sum_mask # [B, H]
    
    H = mean_hs.unsqueeze(1).expand(-1, len(DIRS), -1) # [B, N, H]
    logits = (H * W.to(hidden_states.dtype)).sum(dim=2) + b.to(hidden_states.dtype)
    probs = torch.sigmoid(logits)
    return probs # [B, N]

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/contrastive/contrastive_mean_layer8.pt"
    pairs_path = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    with open(pairs_path, "r") as f:
        pairs = json.load(f)

    results = []
    print(f"Evaluating Contrastive Mean Pooling Probe on {len(pairs)} pairs...")

    for pair in tqdm(pairs):
        label = pair["label"]
        l_idx = list(DIRS).index(label)
        
        with torch.no_grad():
            # A: Missing
            enc_a = tokenizer(pair["A"], return_tensors="pt").to(device)
            out_a = lm(**enc_a, output_hidden_states=True)
            prob_a = run_probe_mean_pooling(out_a.hidden_states[layer_idx+1], enc_a["attention_mask"], W, b)[0, l_idx]
            
            # B: Filled
            enc_b = tokenizer(pair["B"], return_tensors="pt").to(device)
            out_b = lm(**enc_b, output_hidden_states=True)
            prob_b = run_probe_mean_pooling(out_b.hidden_states[layer_idx+1], enc_b["attention_mask"], W, b)[0, l_idx]
            
            results.append({
                "label": label,
                "pa": float(prob_a.cpu().numpy()),
                "pb": float(prob_b.cpu().numpy())
            })

    # Summary Stats
    total = len(results)
    a_hit = sum(1 for r in results if r["pa"] > 0.5)
    b_hit = sum(1 for r in results if r["pb"] < 0.1)
    pair_hit = sum(1 for r in results if r["pa"] > 0.5 and r["pb"] < 0.1)
    
    print("\n[CONTRASTIVE MEAN POOLING RESULTS]")
    print(f"  A-Score (P(A)>0.5): {a_hit}/{total} ({a_hit/total:.1%})")
    print(f"  B-Score (P(B)<0.1): {b_hit}/{total} ({b_hit/total:.1%})")
    print(f"  Pair Accuracy (A & B): {pair_hit}/{total} ({pair_hit/total:.1%})")
    
    print("\nDetailed Comparison (A -> B)")
    for r in results:
        print(f"{r['label']:<7} | {r['pa']:>4.0%} -> {r['pb']:>4.0%} | {'OK' if r['pa'] > 0.5 and r['pb'] < 0.1 else 'NG'}")

if __name__ == "__main__":
    main()
