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
    vec = hs[0, pos].to(torch.float32)
    H = vec.unsqueeze(0).expand(len(DIRS), -1)
    logits = (H * W.to(torch.float32)).sum(dim=1) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

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
            for mode, text_key in [("A", "text_a"), ("B", "text_b")]:
                text = pair[text_key]
                enc = tokenizer(text, return_tensors="pt").to(device)
                out = lm(**enc, output_hidden_states=True)
                hs = out.hidden_states[layer_idx + 1]

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
            
            pair_res["valid"] = True
            results.append(pair_res)

    return results

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_final_token/lodo_query_layer8_sgd_Events_1.pt"
    pairs_dir = Path("runs/balanced/experiment7_neurons/corpus_pairs")
    neurons_path = "runs/balanced/experiment7_neurons_ft/neurons_report.json"
    out_path = "runs/balanced/experiment7_neurons_ft/corpus_shift_data.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print(f"Loading Final Token probe weights from {model_path}...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device).to(torch.bfloat16)
    b = weights["b"].to(device).to(torch.bfloat16)

    with open(neurons_path, "r") as f:
        neuron_report = json.load(f)

    all_results = {}
    
    for target_label in DIRS:
        pair_file = pairs_dir / f"corpus_pairs_{target_label}.jsonl"
        if not pair_file.exists():
            print(f"  Skipping {target_label}: no corpus pairs found.")
            continue
            
        pairs = []
        with open(pair_file, "r") as f:
            for line in f:
                pairs.append(json.loads(line))
        
        if not pairs:
            continue
            
        print(f"\n--- Processing {len(pairs)} pairs for {target_label} ---")
        top_neurons = [n["index"] for n in neuron_report[target_label][:3]]
        
        label_results = process_pairs(pairs, target_label, top_neurons, tokenizer, lm, layer_idx, W, b, device)
        
        all_results[target_label] = {
            "neuron_indices": top_neurons,
            "pairs": label_results
        }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nCorpus tracking (Final Token) complete. Data saved to {out_path}")

if __name__ == "__main__":
    main()
