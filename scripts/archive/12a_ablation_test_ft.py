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

def run_probe_final_token(hs, pos, W, b, ablation_spec=None):
    """
    Runs the Final Token probe. A single vector from `pos` is fed to all 7 heads.
    If ablation_spec=(target_label, neuron_idx, ablate_value), that neuron is zeroed out.
    """
    vec = hs[0, pos].clone().to(torch.float32)
    
    if ablation_spec:
        label, n_idx, val = ablation_spec
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
    
    ablation_value = -10.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print("Loading Final Token probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)

    # Load top neurons from experiment 7 FT results
    with open(neurons_path, "r") as f:
        neuron_report = json.load(f)

    # Build ablation targets from the FT neuron report (top 2 per label)
    targets = []
    for label in DIRS:
        top_ns = neuron_report.get(label, [])
        for n in top_ns[:2]:
            targets.append((label, n["index"], f"FT top neuron for {label} (n{n['index']}, avg_w={n['avg_weight']:.3f})"))

    with open(pairs_path, "r") as f:
        raw_pairs = json.load(f)
        pairs_data = {p["label"]: p for p in raw_pairs}
    
    output_dir = Path("runs/balanced/experiment8_causal_ft")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("\n" + "="*80)
    print("EXPERIMENT 8A (FT): CAUSAL ABLATION (KNOCK-OUT)")
    print("="*80)

    for target_label, neuron_idx, desc in targets:
        if target_label not in pairs_data:
            continue
        
        pair = pairs_data[target_label]
        text_A = pair["A"]
        
        print(f"\n>>> Target: {desc}")
        print(f"    Ablation Value: {ablation_value}")
        
        with torch.no_grad():
            enc = tokenizer(text_A, return_tensors="pt").to(device)
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1]
            pos = get_final_token_position(enc["attention_mask"])
            
            base_probs = run_probe_final_token(hs, pos, W, b)
            ablation_spec = (target_label, neuron_idx, ablation_value)
            ablated_probs = run_probe_final_token(hs, pos, W, b, ablation_spec)

            print(f"{'Label':<10} | {'Base Prob (%)':<15} | {'Ablated Prob (%)':<20} | {'Delta':<10}")
            print("-" * 65)
            for d in DIRS:
                b_p = base_probs[d] * 100
                a_p = ablated_probs[d] * 100
                diff = a_p - b_p
                marker = "<-- TARGET" if d == target_label else ""
                alert = "[!] Crosstalk" if d != target_label and abs(diff) > 5.0 else ""
                print(f"{d:<10} | {b_p:>8.2f}%       | {a_p:>8.2f}%           | {diff:>6.2f}% {marker} {alert}")

            results.append({
                "target_label": target_label,
                "neuron_idx": neuron_idx,
                "description": desc,
                "base_probs": base_probs,
                "ablated_probs": ablated_probs
            })

    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved ablation results to {output_dir / 'ablation_results.json'}")

if __name__ == "__main__":
    main()
