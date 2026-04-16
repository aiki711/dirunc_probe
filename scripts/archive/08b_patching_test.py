import torch
import json
from transformers import AutoTokenizer, AutoModel
import sys
import os
from pathlib import Path

# Add the project root to sys.path
sys.path.append(os.getcwd())
from scripts.common import DIRS

def get_all_query_positions(input_ids, tokenizer):
    full_ids = input_ids[0].tolist()
    tid_end = 57528 # ']'
    
    all_pos = [i for i, tid in enumerate(full_ids) if tid == tid_end]
    if len(all_pos) < 7:
        return None
    
    last_7 = all_pos[-7:]
    return {d: pos for d, pos in zip(DIRS, last_7)}

def run_probe_prediction(hs, pos_map, W, b, patch_spec=None):
    """
    Runs the linear probe over the hidden states.
    If patch_spec is (target_label, neuron_idx, patch_value),
    it forces the specific activation to the patch_value before prediction.
    """
    dir_vecs = []
    for d in DIRS:
        vec = hs[0, pos_map[d]].clone().to(torch.float32)
        
        if patch_spec and d == patch_spec[0]:
            label, n_idx, val = patch_spec
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
    
    # Define targets based on Experiment 7 statistical results:
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
    print("EXPERIMENT 8B: CAUSAL PATCHING (STATE TRANSPLANTATION)")
    print("="*80)

    for target_label, neuron_idx, desc in targets:
        if target_label not in pairs_data: continue
        
        pair = pairs_data[target_label]
        text_A = pair["A"]
        text_B = pair["B"]
        
        print(f"\n>>> Target: {desc}")
        print(f"    Label: {target_label.upper()}, Neuron: n{neuron_idx}")
        
        with torch.no_grad():
            # 1. Forward pass [B] Filled to get the "Suppressed" activation value
            enc_B = tokenizer(text_B, return_tensors="pt").to(device)
            out_B = lm(**enc_B, output_hidden_states=True)
            hs_B = out_B.hidden_states[layer_idx + 1]
            pos_map_B = get_all_query_positions(enc_B["input_ids"], tokenizer)
            if not pos_map_B: continue
            
            # Extract the target activation from [B]
            patch_value = float(hs_B[0, pos_map_B[target_label], neuron_idx].to(torch.float32).cpu().numpy())
            print(f"    Extracted Patch Value from [B]: {patch_value:.3f}")
            
            # 2. Forward pass [A] Missing
            enc_A = tokenizer(text_A, return_tensors="pt").to(device)
            out_A = lm(**enc_A, output_hidden_states=True)
            hs_A = out_A.hidden_states[layer_idx + 1]
            pos_map_A = get_all_query_positions(enc_A["input_ids"], tokenizer)
            if not pos_map_A: continue
            
            # Run Base Prediction on [A] (Original state)
            base_probs_A = run_probe_prediction(hs_A, pos_map_A, W, b)
            
            # Run Patched Prediction on [A] (Transplanting B's activation into A's context)
            patch_spec = (target_label, neuron_idx, patch_value)
            patched_probs_A = run_probe_prediction(hs_A, pos_map_A, W, b, patch_spec)

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
