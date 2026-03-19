import torch
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.getcwd())
from scripts.common import DIRS

def run_probe_final_token(hs_token, W, b):
    """
    Final Token variant: feed a single hidden state vector to all 7 label heads.
    hs_token: [hidden_size] tensor
    W: [num_labels, hidden_size], b: [num_labels]
    """
    vec = hs_token.to(torch.float32)          # [hidden]
    logits = (W.to(torch.float32) @ vec) + b.to(torch.float32)  # [num_labels]
    probs = torch.sigmoid(logits)             # [num_labels]
    return {d: probs[i].item() for i, d in enumerate(DIRS)}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_final_token/lodo_query_layer8_sgd_Events_1.pt"
    
    # Selected dramatic samples (same as standard version for comparison)
    targets = [
        {"id": "exp9_how_shared_ride", "text": "Could you book me a shared ride?"},
        {"id": "exp9_where_airport",   "text": "I need to book a taxi to the airport."},
        {"id": "exp9_time_tonight",    "text": "I want to find a restaurant for tonight."}
    ]
    
    output_dir = Path("runs/balanced/experiment9_visuals_ft")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "trajectory_data.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print(f"Loading Final Token probe weights from {model_path}...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    results = {}
    
    with torch.no_grad():
        for target in targets:
            print(f"Processing: {target['id']}")
            text = target["text"]
            
            # Tokenize the full text (NO QUERY TOKENS appended)
            enc = tokenizer(text, return_tensors="pt").to(device)
            input_ids = enc["input_ids"][0]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # Build trajectory by progressively extending the input (1 to N tokens)
            # Each step: feed the first i+1 tokens, and check the FINAL token position
            trajectory = []
            for i in range(1, len(input_ids) + 1):
                partial_ids = input_ids[:i].unsqueeze(0)
                partial_mask = torch.ones_like(partial_ids)
                out = lm(input_ids=partial_ids, attention_mask=partial_mask, output_hidden_states=True)
                hs = out.hidden_states[layer_idx + 1][0]  # [seq, hidden]
                
                # Use the last token's hidden state (the current "frontier")
                final_hs = hs[-1]  # [hidden]
                probs = run_probe_final_token(final_hs, W, b)
                
                token_clean = tokens[i-1].replace("▁", " ").strip()
                trajectory.append({
                    "token_idx": i - 1,
                    "token_str": token_clean if token_clean else tokens[i-1],
                    "probs": probs
                })
            
            results[target["id"]] = {
                "text": text,
                "trajectory": trajectory
            }
            print(f"  -> {len(trajectory)} trajectory steps computed.")
            
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved (Final Token) trajectory data to {out_file}")

if __name__ == "__main__":
    main()
