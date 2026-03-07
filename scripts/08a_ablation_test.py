import torch
import json
from transformers import AutoTokenizer, AutoModel
import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.getcwd())
from scripts.common import DIRS

def get_all_query_positions(input_ids):
    full_ids = input_ids[0].tolist()
    tid_end = 57528 # ']' part of '?]'
    
    all_pos = [i for i, tid in enumerate(full_ids) if tid == tid_end]
    if len(all_pos) < 7:
        return None
    
    last_7 = all_pos[-7:]
    return {d: pos for d, pos in zip(DIRS, last_7)}

def run_probe_prediction(hs, pos_map, W, b, ablation_spec=None):
    """
    Runs the linear probe over the hidden states.
    If ablation_spec is provided as (target_label, neuron_idx, ablate_value),
    it forcibly overwrites that specific activation before prediction.
    """
    dir_vecs = []
    for d in DIRS:
        # Clone to avoid modifying the original hs in place
        vec = hs[0, pos_map[d]].clone().to(torch.float32)
        
        if ablation_spec and d == ablation_spec[0]:
            label, n_idx, val = ablation_spec
            vec[n_idx] = val
            
        dir_vecs.append(vec)
        
    H = torch.stack(dir_vecs, dim=0) # [7, hidden_size]
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_sgd_Events_1.pt"
    pairs_path = "runs/balanced/experiment7_neurons/shift_pairs.json"
    
    ablation_value = -10.0 # Force severe suppression
    
    # Define targets based on Experiment 7 statistical results:
    # (Label to apply ablation on, Neuron Index, Description)
    targets = [
        ("where", 100, "Specialized WHERE neuron"),
        ("who", 742, "Specialized WHO neuron"),
        ("when", 1725, "General Attribute neuron (Found in WHEN/WHAT)"),
        ("where", 731, "General Spatial/Temporal neuron (Found in WHERE/WHEN)")
    ]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print("Loading proxy probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    with open(pairs_path, "r") as f:
        raw_pairs = json.load(f)
        # Convert list to dict mapping label -> pair
        pairs_data = {p["label"]: p for p in raw_pairs}
    
    output_dir = Path("runs/balanced/experiment8_causal")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print("\n" + "="*80)
    print("EXPERIMENT 8A: CAUSAL ABLATION (KNOCK-OUT)")
    print("="*80)

    for target_label, neuron_idx, desc in targets:
        if target_label not in pairs_data: continue
        
        pair = pairs_data[target_label]
        text_A = pair["A"]
        
        print(f"\n>>> Target: {desc}")
        print(f"    Label: {target_label.upper()}, Neuron: n{neuron_idx}")
        print(f"    Ablation Value: {ablation_value}")
        
        with torch.no_grad():
            enc = tokenizer(text_A, return_tensors="pt").to(device)
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1]
            
            pos_map = get_all_query_positions(enc["input_ids"])
            if not pos_map:
                print("Error: Could not find query positions.")
                continue
            
            # 1. Base Prediction (Original behavior on Missing text)
            base_probs = run_probe_prediction(hs, pos_map, W, b)
            
            # 2. Ablated Prediction (Forcing one neuron to silence)
            ablation_spec = (target_label, neuron_idx, ablation_value)
            ablated_probs = run_probe_prediction(hs, pos_map, W, b, ablation_spec)

            print(f"{'Label':<10} | {'Base Prob (%)':<15} | {'Ablated Prob (%)':<20} | {'Delta':<10}")
            print("-" * 65)
            for d in DIRS:
                b_p = base_probs[d] * 100
                a_p = ablated_probs[d] * 100
                diff = a_p - b_p
                
                marker = "<-- TARGET" if d == target_label else ""
                # Highlight if non-target labels are affected (Crosstalk)
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
