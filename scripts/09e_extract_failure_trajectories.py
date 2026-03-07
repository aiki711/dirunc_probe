import torch
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

# Add the project root to sys.path
sys.path.append(os.getcwd())
from scripts.common import DIRS

def run_probe_prediction(hs_token, W, b):
    logits = (hs_token.to(torch.float32) @ W.T.to(torch.float32)) + b.to(torch.float32)
    probs = torch.sigmoid(logits)
    return {d: float(probs[i].cpu().numpy()) for i, d in enumerate(DIRS)}

def fold_trajectory(tokens, hs, W, b):
    new_trajectory = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if '[' in token and i + 1 < len(tokens):
            j = i
            found_end = -1
            block_tokens = []
            while j < min(i + 6, len(tokens)):
                block_tokens.append(tokens[j])
                if ']' in tokens[j]:
                    found_end = j
                    break
                j += 1
            
            if found_end != -1:
                # Build accumulated string to find the label
                full_block_str = "".join(block_tokens).upper()
                rep_idx = found_end
                rep_hs = hs[rep_idx]
                probs = run_probe_prediction(rep_hs, W, b)
                
                detected_label = "??"
                for d in DIRS:
                    if d.upper() in full_block_str:
                        detected_label = d.upper()
                        break
                
                new_trajectory.append({
                    "token_idx": found_end,
                    "token_str": f"[{detected_label}?]",
                    "probs": probs
                })
                i = found_end + 1
                continue
        
        token_clean = token.replace(" ", "") if token.startswith(" ") else token
        probs = run_probe_prediction(hs[i], W, b)
        new_trajectory.append({
            "token_idx": i,
            "token_str": token_clean,
            "probs": probs
        })
        i += 1
    return new_trajectory

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_sgd_Events_1.pt"
    failure_path = Path("runs/balanced/experiment9_visuals/failure_candidates.json")
    
    if not failure_path.exists():
        print(f"Error: {failure_path} not found.")
        return

    with open(failure_path, "r") as f:
        targets = json.load(f)
    
    output_dir = Path("runs/balanced/experiment9_visuals")
    out_file = output_dir / "failure_trajectory_data.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print("Loading probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    results = {}
    
    with torch.no_grad():
        for target in targets[:5]:
            print(f"Processing Failure: {target['id']}")
            text = target["text_B"] 
            
            enc = tokenizer(text, return_tensors="pt").to(device)
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1][0]
            
            trajectory = fold_trajectory(tokens, hs, W, b)
                
            results[target["id"]] = {
                "text": text,
                "label": target["label"],
                "expected": 0, 
                "trajectory": trajectory
            }
            
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved (folded) failure trajectory data to {out_file}")

if __name__ == "__main__":
    main()
