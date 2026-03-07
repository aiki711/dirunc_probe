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
    """
    Folds related tokens (like [ WHO ? ]) into single units for cleaner visualization.
    Only takes the representation of the ']' token as the representative for the group.
    """
    new_trajectory = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # Detect start of a special query token block
        if '[' in token and i + 1 < len(tokens):
            # Look ahead for ']'
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
        
        # Default: normal token
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
    
    # Selected dramatic samples
    targets = [
        {"id": "exp9_how_shared_ride", "text": "Could you book me a shared ride?"},
        {"id": "exp9_where_airport", "text": "I need to book a taxi to the airport."},
        {"id": "exp9_time_tonight", "text": "I want to find a restaurant for tonight."}
    ]
    
    output_dir = Path("runs/balanced/experiment9_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "trajectory_data.json"
    
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
        for target in targets:
            print(f"Processing: {target['id']}")
            text = target["text"]
            full_text = text + " [WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?] [WHICH?]"
            
            enc = tokenizer(full_text, return_tensors="pt").to(device)
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1][0] 
            
            # Use folding logic
            trajectory = fold_trajectory(tokens, hs, W, b)
                
            results[target["id"]] = {
                "text": text,
                "trajectory": trajectory
            }
            
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved (folded) trajectory data to {out_file}")

if __name__ == "__main__":
    main()
