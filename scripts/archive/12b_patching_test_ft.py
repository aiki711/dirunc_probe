import torch
import json
from transformers import AutoTokenizer, AutoModel
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
from scripts.common import DIRS

def get_final_token_position(attention_mask):
    valid_indices = torch.nonzero(attention_mask[0]).squeeze(-1)
    return valid_indices[-1].item()

def run_probe_final_token(hs, pos, W, b, patch_spec=None):
    vec = hs[0, pos].clone().to(torch.float32)
    if patch_spec:
        label, n_idx, val = patch_spec
        vec[n_idx] = val
    H = vec.unsqueeze(0).expand(len(DIRS), -1)
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_final_token/lodo_query_layer8_sgd_Events_1.pt"
    pairs_path = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"
    neurons_path = "runs/balanced/experiment7_neurons_ft/neurons_report.json"

    with open(neurons_path, "r") as f:
        neuron_report = json.load(f)

    # Top 2 neurons per label for patching
    targets = []
    for label in DIRS:
        top_ns = neuron_report.get(label, [])
        for n in top_ns[:2]:
            targets.append((label, n["index"], f"FT top neuron for {label} (n{n['index']})"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    with open(pairs_path, "r") as f:
        raw_pairs = json.load(f)
        pairs_data = {p["label"]: p for p in raw_pairs}
    
    output_dir = Path("runs/balanced/experiment8_causal_ft")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("\n" + "="*80)
    print("EXPERIMENT 8B (FT): CAUSAL PATCHING (STATE TRANSPLANTATION)")
    print("="*80)

    for target_label, neuron_idx, desc in targets:
        if target_label not in pairs_data:
            continue
        
        pair = pairs_data[target_label]
        text_A = pair["A"]
        text_B = pair["B"]
        
        print(f"\n>>> Target: {desc}")
        
        with torch.no_grad():
            # 1. Forward pass [B] Filled -> extract patch value
            enc_B = tokenizer(text_B, return_tensors="pt").to(device)
            out_B = lm(**enc_B, output_hidden_states=True)
            hs_B = out_B.hidden_states[layer_idx + 1]
            pos_B = get_final_token_position(enc_B["attention_mask"])
            patch_value = float(hs_B[0, pos_B, neuron_idx].to(torch.float32).cpu().numpy())
            print(f"    Extracted Patch Value from [B]: {patch_value:.3f}")
            
            # 2. Forward pass [A] Missing -> base prediction + patched prediction
            enc_A = tokenizer(text_A, return_tensors="pt").to(device)
            out_A = lm(**enc_A, output_hidden_states=True)
            hs_A = out_A.hidden_states[layer_idx + 1]
            pos_A = get_final_token_position(enc_A["attention_mask"])
            
            base_probs_A = run_probe_final_token(hs_A, pos_A, W, b)
            patch_spec = (target_label, neuron_idx, patch_value)
            patched_probs_A = run_probe_final_token(hs_A, pos_A, W, b, patch_spec)

            print(f"{'Label':<10} | {'Base [A] Prob (%)':<18} | {'Patched [A] Prob (%)':<20} | {'Delta':<10}")
            print("-" * 75)
            for d in DIRS:
                b_p = base_probs_A[d] * 100
                p_p = patched_probs_A[d] * 100
                diff = p_p - b_p
                marker = "<-- TARGET" if d == target_label else ""
                alert = "[!] Crosstalk" if d != target_label and abs(diff) > 5.0 else ""
                print(f"{d:<10} | {b_p:>8.2f}%          | {p_p:>8.2f}%              | {diff:>6.2f}% {marker} {alert}")

            results.append({
                "target_label": target_label,
                "neuron_idx": neuron_idx,
                "description": desc,
                "patch_value": patch_value,
                "base_probs_A": base_probs_A,
                "patched_probs_A": patched_probs_A
            })

    with open(output_dir / "patching_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved patching results to {output_dir / 'patching_results.json'}")

if __name__ == "__main__":
    main()
