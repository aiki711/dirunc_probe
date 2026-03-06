import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import sys
import os
import tqdm

# Add the project root to sys.path
sys.path.append(os.getcwd())
from scripts.common import DIRS, QUERY_LABEL_STR

def get_all_query_positions(input_ids, tokenizer):
    full_ids = input_ids[0].tolist()
    tid_end = 57528 # ']' part of '?]' or similar. Verified ID for '?]' in Gemma-2 is 57528.
    
    # Extract positions of all tokens ending in ']'
    all_pos = [i for i, tid in enumerate(full_ids) if tid == tid_end]
    
    # We expect the last 7 to be our query tokens [WHO, WHAT, WHEN, WHERE, WHY, HOW, WHICH]
    if len(all_pos) < 7:
        return None
    
    last_7 = all_pos[-7:]
    return {d: pos for d, pos in zip(DIRS, last_7)}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_sgd_Events_1.pt"
    pairs_path = "runs/balanced/experiment7_neurons/shift_pairs.json"
    neurons_path = "runs/balanced/experiment7_neurons/neurons_report.json"
    out_path = "runs/balanced/experiment7_neurons/activation_shifts.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print(f"Loading probe weights and reports...")
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
                # Hidden States index: 0 is embedding, 1 is layer 0, ..., layer_idx+1
                hs = out.hidden_states[layer_idx + 1] # [1, seq_len, hidden_size]
                
                # Get positions of all query tokens
                pos_map = get_all_query_positions(enc["input_ids"], tokenizer)
                if not pos_map:
                    print(f"Warning: Could not find all 7 query tokens in {mode}")
                    # Fallback or error
                    continue
                
                pos = pos_map[target_label]
                
                # Extract activations for top neurons at this position
                vec = hs[0, pos].to(torch.float32) # [hidden_size]
                acts = [float(vec[idx].cpu().numpy()) for idx in top_neurons]
                
                # Full probe prediction for this label
                dir_vecs = []
                for d in DIRS:
                    dir_vecs.append(hs[0, pos_map[d]])
                
                H = torch.stack(dir_vecs, dim=0).to(torch.float32) # [7, hidden_size]
                logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
                probs = torch.sigmoid(logits)
                prob = float(probs[DIRS.index(target_label)].cpu().numpy())

                if mode == "A":
                    pair_res["A_activations"] = acts
                    pair_res["A_prob"] = prob
                else:
                    pair_res["B_activations"] = acts
                    pair_res["B_prob"] = prob
        
        results.append(pair_res)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Activation shifts saved to {out_path}")

if __name__ == "__main__":
    main()
