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

def run_probe_prediction(hs, pos_map, W, b, injection_spec=None):
    """
    Runs the linear probe over the hidden states.
    If injection_spec is (target_label, neuron_idx, inject_value),
    it forces the specific activation to that value before prediction.
    """
    dir_vecs = []
    for d in DIRS:
        vec = hs[0, pos_map[d]].clone().to(torch.float32)
        
        if injection_spec and d == injection_spec[0]:
            label, n_idx, val = injection_spec
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
    
    # Target Neuron for HOW (from Experiment 7 statistics)
    how_neuron_idx = 625 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print("Loading proxy probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device) # [7, hidden_size]
    b = weights["b"].to(device)
    
    # --- Approach 2: Weight Sign Check ---
    print("\n" + "="*80)
    print("APPROACH 2: PROBE WEIGHT SIGN CHECK")
    print("="*80)
    how_idx = DIRS.index("how")
    w_val = float(W[how_idx, how_neuron_idx].to(torch.float32).cpu().numpy())
    print(f"Probe Weight W[how, n{how_neuron_idx}]: {w_val:.6f}")
    if w_val < 0:
        print(">> RESULT: Weight is NEGATIVE. Firing (Activation increase) decreases missing probability.")
        print("           This perfectly explains why HOW neurons 'fire' when information is provided.")
    else:
        print(">> RESULT: Weight is POSITIVE. Direct correlation with missing state.")

    with open(pairs_path, "r") as f:
        raw_pairs = json.load(f)
        pairs_data = {p["label"]: p for p in raw_pairs}
    
    if "how" not in pairs_data:
        print("Error: 'how' label data not found in shift_pairs.json")
        return

    output_dir = Path("runs/balanced/experiment8_causal")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair = pairs_data["how"]
    text_A = pair["A"] # Missing
    text_B = pair["B"] # Filled

    # --- Approach 1: Token-level Tracing ---
    print("\n" + "="*80)
    print("APPROACH 1: TOKEN-LEVEL ACTIVATION TRACING (Text B: Filled)")
    print("="*80)
    with torch.no_grad():
        enc = tokenizer(text_B, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        out = lm(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1] # [1, seq_len, hidden_size]
        
        activations = hs[0, :, how_neuron_idx].to(torch.float32).cpu().numpy()
        
        print(f"{'Token':<20} | {'Activation (n625)':<20}")
        print("-" * 45)
        for t, val in zip(tokens, activations):
            bar = "#" * int(max(0, val) * 2)
            print(f"{t:<20} | {val:>8.3f} {bar}")

    # --- Approach 3: Activation Injection ---
    print("\n" + "="*80)
    print("APPROACH 3: ACTIVATION INJECTION (Text A: Missing)")
    print("="*80)
    injection_value = 5.0
    print(f"Target: Injecting n{how_neuron_idx} = {injection_value} into 'Missing' state.")
    
    with torch.no_grad():
        enc_A = tokenizer(text_A, return_tensors="pt").to(device)
        out_A = lm(**enc_A, output_hidden_states=True)
        hs_A = out_A.hidden_states[layer_idx + 1]
        pos_map_A = get_all_query_positions(enc_A["input_ids"], tokenizer)
        
        # Base Prediction on Missing text
        base_probs = run_probe_prediction(hs_A, pos_map_A, W, b)
        
        # Injected Prediction (Forcing high activation)
        inject_spec = ("how", how_neuron_idx, injection_value)
        injected_probs = run_probe_prediction(hs_A, pos_map_A, W, b, inject_spec)

        print(f"{'Label':<10} | {'Base Prob (%)':<15} | {'Injected Prob (%)':<20} | {'Delta':<10}")
        print("-" * 65)
        for d in DIRS:
            b_p = base_probs[d] * 100
            i_p = injected_probs[d] * 100
            diff = i_p - b_p
            marker = "<-- TARGET" if d == "how" else ""
            print(f"{d:<10} | {b_p:>8.2f}%       | {i_p:>8.2f}%           | {diff:>6.2f}% {marker}")

        results_how = {
            "weight_analysis": {
                "neuron_idx": how_neuron_idx,
                "weight_value": w_val
            },
            "tracing_data": {
                "tokens": tokens,
                "activations": activations.tolist()
            },
            "injection_results": {
                "injection_value": injection_value,
                "base_probs": base_probs,
                "injected_probs": injected_probs
            }
        }
        with open(output_dir / "how_mechanism_results.json", "w") as f:
            json.dump(results_how, f, indent=2)
        print(f"\nSaved HOW mechanism results to {output_dir / 'how_mechanism_results.json'}")

if __name__ == "__main__":
    main()
