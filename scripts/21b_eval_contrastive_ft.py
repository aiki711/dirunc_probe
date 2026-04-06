import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
import tqdm

sys.path.append(os.getcwd())
from scripts.common import DIRS, strip_query_tokens

def get_final_token_position(attention_mask):
    valid_indices = torch.nonzero(attention_mask[0]).squeeze(-1)
    if valid_indices.numel() == 0:
        return -1
    return valid_indices[-1].item()

def run_probe_final_token(hs, pos, W, b):
    if pos < 0:
        return torch.zeros(len(DIRS))
    vec = hs[0, pos].to(torch.float32)
    H = vec.unsqueeze(0).expand(len(DIRS), -1)
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--probe_path", type=str, default="runs/contrastive/contrastive_ft_layer8.pt")
    parser.add_argument("--test_json", type=str, default="runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json")
    parser.add_argument("--layer_idx", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lm = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print(f"Loading probe from {args.probe_path}...")
    weights = torch.load(args.probe_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    with open(args.test_json, "r") as f:
        test_pairs = json.load(f)

    print(f"Evaluating {len(test_pairs)} pairs...")
    
    n_correct = 0
    results = []

    for pair in tqdm.tqdm(test_pairs):
        # Strip query tokens for FT evaluation
        text_a = strip_query_tokens(pair["A"]).strip()
        text_b = strip_query_tokens(pair["B"]).strip()
        label = pair["label"]
        l_idx = DIRS.index(label)

        with torch.no_grad():
            # Process A
            enc_a = tokenizer([text_a], return_tensors="pt").to(device)
            out_a = lm(**enc_a, output_hidden_states=True)
            hs_a = out_a.hidden_states[args.layer_idx + 1]
            pos_a = get_final_token_position(enc_a["attention_mask"])
            probs_a = run_probe_final_token(hs_a, pos_a, W, b)
            prob_a = float(probs_a[l_idx].cpu().numpy())

            # Process B
            enc_b = tokenizer([text_b], return_tensors="pt").to(device)
            out_b = lm(**enc_b, output_hidden_states=True)
            hs_b = out_b.hidden_states[args.layer_idx + 1]
            pos_b = get_final_token_position(enc_b["attention_mask"])
            probs_b = run_probe_final_token(hs_b, pos_b, W, b)
            prob_b = float(probs_b[l_idx].cpu().numpy())

        # Success criteria
        # A should be Missing (>0.5), B should be Filled (<0.1)
        is_correct = (prob_a > 0.5) and (prob_b < 0.1)
        if is_correct:
            n_correct += 1
        
        results.append({
            "label": label,
            "prob_A": prob_a,
            "prob_B": prob_b,
            "correct": is_correct
        })

    print("\n=== FT Contrastive Probe Evaluation Results ===")
    print(f"Pair Accuracy: {n_correct}/{len(test_pairs)} ({100*n_correct/len(test_pairs):.2f}%)")
    print("-" * 50)
    print(f"{'Label':<10} | {'P(A)':<8} | {'P(B)':<8} | {'Status'}")
    for res in results:
        status = "OK" if res["correct"] else "NG"
        print(f"{res['label']:<10} | {res['prob_A']:>7.1%} | {res['prob_B']:>7.1%} | {status}")

if __name__ == "__main__":
    main()
