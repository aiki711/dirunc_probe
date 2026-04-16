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
    """Returns the index of the last valid (non-padding) token."""
    valid_indices = torch.nonzero(attention_mask[0]).squeeze(-1)
    return valid_indices[-1].item()

def run_probe_final_token(hs, pos, W, b):
    """
    Runs the Final Token probe:
    - extracts a SINGLE vector at position `pos` (the final token)
    - feeds the SAME vector to all 7 direction heads
    """
    vec = hs[0, pos].to(torch.float32)  # [hidden_size]
    H = vec.unsqueeze(0).expand(len(DIRS), -1)  # [7, hidden_size]
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    # Use sgd_Events_1 as a representative proxy (same domain as 07c)
    model_path = "runs/balanced/experiment6_final_token/lodo_query_layer8_sgd_Events_1.pt"
    pairs_path = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"
    neurons_path = "runs/balanced/experiment7_neurons_ft/neurons_report.json"
    out_path = "runs/balanced/experiment7_neurons_ft/activation_shifts.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print(f"Loading Final Token probe weights from {model_path}...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device).to(torch.bfloat16)
    b = weights["b"].to(device).to(torch.bfloat16)

    with open(pairs_path, "r") as f:
        pairs = json.load(f)
    with open(neurons_path, "r") as f:
        neuron_report = json.load(f)

    results = []
    
    for pair in tqdm.tqdm(pairs, desc="Processing pairs"):
        target_label = pair["label"]
        top_neurons = [n["index"] for n in neuron_report[target_label][:10]]
        
        pair_res = {
            "label": target_label,
            "A_text": pair["A_raw"],
            "B_text": pair["B_raw"],
            "neuron_indices": top_neurons,
            "A_activations": [],
            "B_activations": [],
            "A_prob": 0.0,
            "B_prob": 0.0
        }

        with torch.no_grad():
            for mode in ["A", "B"]:
                text = pair[mode]
                enc = tokenizer(text, return_tensors="pt").to(device)
                out = lm(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer_idx + 1]  # [1, seq_len, hidden_size]
                
                # Final Token: use the last valid token position
                pos = get_final_token_position(enc["attention_mask"])
                vec = hs[0, pos].to(torch.float32)
                acts = [float(vec[idx].cpu().numpy()) for idx in top_neurons]
                
                probs = run_probe_final_token(hs, pos, W, b)
                prob = probs[target_label]

                if mode == "A":
                    pair_res["A_activations"] = acts
                    pair_res["A_prob"] = prob
                else:
                    pair_res["B_activations"] = acts
                    pair_res["B_prob"] = prob
        
        results.append(pair_res)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nActivation shifts (Final Token) saved to {out_path}")
    print("\n--- Results Summary ---")
    for r in results:
        delta = r["B_prob"] - r["A_prob"]
        print(f"  {r['label']:<7}: A_prob={r['A_prob']:.3f}, B_prob={r['B_prob']:.3f}, delta={delta:+.3f}")

if __name__ == "__main__":
    main()
