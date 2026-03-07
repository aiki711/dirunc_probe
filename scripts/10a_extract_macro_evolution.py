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

def extract_special_token_probs(tokens, hs, W, b):
    """
    Extracts probabilities ONLY at the special token positions by finding blocks like [WHO?].
    Returns a dictionary of {label: probability} for the specific label at its token.
    """
    probs_dict = {}
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
                
                for d in DIRS:
                    if d.upper() in full_block_str:
                        probs_dict[d] = probs[d]
                        break
                
                i = found_end + 1
                continue
        i += 1
    return probs_dict

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_sgd_Events_1.pt"
    
    # Define a multi-turn scenario (Booking a train)
    scenario_name = "train_booking"
    turns = [
        {"turn_id": 1, "speaker": "User", "text": "I need to book a train ticket."},
        {"turn_id": 2, "speaker": "Agent", "text": "Sure, where are you leaving from and where are you going?"},
        {"turn_id": 3, "speaker": "User", "text": "I will be leaving from London Liverpool Street and going to Cambridge."},
        {"turn_id": 4, "speaker": "Agent", "text": "When would you like to travel?"},
        {"turn_id": 5, "speaker": "User", "text": "I want to leave on Friday after 15:15."},
        {"turn_id": 6, "speaker": "Agent", "text": "How many people are traveling?"},
        {"turn_id": 7, "speaker": "User", "text": "Just 2 adults."}
    ]
    
    output_dir = Path("runs/balanced/experiment10_macro")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "macro_trajectory.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    print("Loading probe weights...")
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device)
    b = weights["b"].to(device)
    
    results = {
        "scenario": scenario_name,
        "turns": turns,
        "macro_evolution": []
    }
    
    current_context = ""
    
    with torch.no_grad():
        for turn in turns:
            print(f"Processing Turn {turn['turn_id']}...")
            
            # Append this turn's text to the context
            if current_context:
                current_context += "\n"
            current_context += f"{turn['speaker']}: {turn['text']}"
            
            # Append queries to simulate the full state at the END of this turn
            probe_text = current_context + "\n[WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?] [WHICH?]"
            
            enc = tokenizer(probe_text, return_tensors="pt").to(device)
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            
            out = lm(**enc, output_hidden_states=True)
            hs = out.hidden_states[layer_idx + 1][0] 
            
            # Extract probabilities ONLY at the special tokens
            turn_probs = extract_special_token_probs(tokens, hs, W, b)
            
            # If a token wasn't found (fallback), assign 1.0 (Missing)
            for d in DIRS:
                if d not in turn_probs:
                    turn_probs[d] = 1.0
                
            results["macro_evolution"].append({
                "turn_id": turn['turn_id'],
                "speaker": turn['speaker'],
                "added_text": turn['text'],
                "probs": turn_probs
            })
            
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved macro trajectory data to {out_file}")

if __name__ == "__main__":
    main()
