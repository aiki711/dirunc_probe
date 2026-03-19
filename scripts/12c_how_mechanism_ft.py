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

def run_probe_final_token(hs, pos, W, b, injection_spec=None):
    vec = hs[0, pos].clone().to(torch.float32)
    if injection_spec:
        label, n_idx, val = injection_spec
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

    # Use top HOW neuron from FT analysis
    how_neurons = neuron_report.get("how", [])
    if not how_neurons:
        print("Error: No 'how' neurons found in FT neuron report. Run 11a first.")
        return
    how_neuron_idx = how_neurons[0]["index"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)

    # --- Weight Sign Check ---
    print("\n" + "="*80)
    print("APPROACH 2: PROBE WEIGHT SIGN CHECK (HOW neuron from FT)")
    print("="*80)
    how_idx_dir = DIRS.index("how")
    w_val = float(W[how_idx_dir, how_neuron_idx].to(torch.float32).cpu().numpy())
    print(f"Probe Weight W[how, n{how_neuron_idx}]: {w_val:.6f}")
    if w_val < 0:
        print(">> RESULT: Weight is NEGATIVE. Firing decreases missing probability.")
    else:
        print(">> RESULT: Weight is POSITIVE. Direct correlation with missing state.")

    with open(pairs_path, "r") as f:
        raw_pairs = json.load(f)
        pairs_data = {p["label"]: p for p in raw_pairs}
    
    if "how" not in pairs_data:
        print("Error: 'how' label data not found in shift_pairs_ft.json")
        return

    output_dir = Path("runs/balanced/experiment8_causal_ft")
    output_dir.mkdir(parents=True, exist_ok=True)

    pair = pairs_data["how"]
    text_A = pair["A"]
    text_B = pair["B"]

    # --- Token-level Tracing on Text B (Filled) ---
    print("\n" + "="*80)
    print("APPROACH 1: TOKEN-LEVEL ACTIVATION TRACING (Text B: Filled)")
    print("="*80)
    with torch.no_grad():
        enc = tokenizer(text_B, return_tensors="pt").to(device)
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        out = lm(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1]
        activations = hs[0, :, how_neuron_idx].to(torch.float32).cpu().numpy()
        
        print(f"{'Token':<20} | {'Activation (n' + str(how_neuron_idx) + ')':<20}")
        print("-" * 45)
        for t, val in zip(tokens, activations):
            bar = "#" * int(max(0, val) * 2)
            print(f"{t:<20} | {val:>8.3f} {bar}")

    # --- Activation Injection on Text A (Missing) ---
    print("\n" + "="*80)
    print("APPROACH 3: ACTIVATION INJECTION (Text A: Missing)")
    print("="*80)
    injection_value = 5.0
    print(f"Target: Injecting n{how_neuron_idx} = {injection_value} into 'Missing' state.")
    
    with torch.no_grad():
        enc_A = tokenizer(text_A, return_tensors="pt").to(device)
        out_A = lm(**enc_A, output_hidden_states=True)
        hs_A = out_A.hidden_states[layer_idx + 1]
        pos_A = get_final_token_position(enc_A["attention_mask"])
        
        base_probs = run_probe_final_token(hs_A, pos_A, W, b)
        inject_spec = ("how", how_neuron_idx, injection_value)
        injected_probs = run_probe_final_token(hs_A, pos_A, W, b, inject_spec)

        print(f"{'Label':<10} | {'Base Prob (%)':<15} | {'Injected Prob (%)':<20} | {'Delta':<10}")
        print("-" * 65)
        for d in DIRS:
            b_p = base_probs[d] * 100
            i_p = injected_probs[d] * 100
            diff = i_p - b_p
            marker = "<-- TARGET" if d == "how" else ""
            print(f"{d:<10} | {b_p:>8.2f}%       | {i_p:>8.2f}%           | {diff:>6.2f}% {marker}")

        results_how = {
            "weight_analysis": {"neuron_idx": how_neuron_idx, "weight_value": w_val},
            "tracing_data": {"tokens": tokens, "activations": activations.tolist()},
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
