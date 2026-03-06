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
    
    if len(all_pos) < 7:
        return None
    
    last_7 = all_pos[-7:]
    return {d: pos for d, pos in zip(DIRS, last_7)}

def process_pairs(pairs, target_label, top_neurons, tokenizer, lm, layer_idx, W, b, device):
    results = []
    
    for pair in tqdm.tqdm(pairs, desc=f"Processing {target_label} pairs"):
        pair_res = {
            "dialogue_id": pair["dialogue_id"],
            "turn_a": pair["turn_a"],
            "turn_b": pair["turn_b"],
            "confounding_diff": pair["confounding_diff"],
            "A_activations": [],
            "B_activations": [],
            "A_prob": 0.0,
            "B_prob": 0.0,
            "valid": False
        }

        with torch.no_grad():
            valid_pair = True
            for mode, text_key in [("A", "text_a"), ("B", "text_b")]:
                text = pair[text_key]
                enc = tokenizer(text, return_tensors="pt").to(device)
                out = lm(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer_idx + 1] # [1, seq_len, hidden_size]
                
                pos_map = get_all_query_positions(enc["input_ids"], tokenizer)
                if not pos_map:
                    valid_pair = False
                    break
                
                pos = pos_map[target_label]
                
                # Extract activations
                vec = hs[0, pos].to(torch.float32)
                acts = [float(vec[idx].cpu().numpy()) for idx in top_neurons]
                
                # Full probe prediction
                dir_vecs = []
                for d in DIRS:
                    dir_vecs.append(hs[0, pos_map[d]])
                
                H = torch.stack(dir_vecs, dim=0).to(torch.float32)
                logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
                probs = torch.sigmoid(logits)
                prob = float(probs[DIRS.index(target_label)].cpu().numpy())

                if mode == "A":
                    pair_res["A_activations"] = acts
                    pair_res["A_prob"] = prob
                else:
                    pair_res["B_activations"] = acts
                    pair_res["B_prob"] = prob
            
            if valid_pair:
                pair_res["valid"] = True
                results.append(pair_res)

    return results

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    # Use sgd_Events_1 as a representative average proxy
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_sgd_Events_1.pt"
    pairs_dir = Path("runs/balanced/experiment7_neurons/corpus_pairs")
    neurons_path = "runs/balanced/experiment7_neurons/neurons_report.json"
    out_path = "runs/balanced/experiment7_neurons/corpus_shift_data.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print(f"Loading proxy probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device).to(torch.bfloat16)
    b = weights["b"].to(device).to(torch.bfloat16)

    with open(neurons_path, "r") as f:
        neuron_report = json.load(f)

    all_results = {}
    
    for target_label in DIRS:
        pair_file = pairs_dir / f"corpus_pairs_{target_label}.jsonl"
        if not pair_file.exists():
            continue
            
        pairs = []
        with open(pair_file, "r") as f:
            for line in f:
                pairs.append(json.loads(line))
                
        # Limit to 50 pairs for faster statistical validation
        pairs = pairs[:50]
        
        if not pairs:
            continue
            
        print(f"\n--- Processing {len(pairs)} pairs for {target_label} ---")
        # Take top 3 neurons for statistical tracking to avoid massive files
        top_neurons = [n["index"] for n in neuron_report[target_label][:3]]
        
        label_results = process_pairs(pairs, target_label, top_neurons, tokenizer, lm, layer_idx, W, b, device)
        
        all_results[target_label] = {
            "neuron_indices": top_neurons,
            "pairs": label_results
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBatch tracking complete. Data saved to {out_path}")

if __name__ == "__main__":
    main()
