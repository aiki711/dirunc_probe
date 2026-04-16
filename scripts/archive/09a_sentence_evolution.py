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
    Faithfully reproduces the ACTUAL Experiment 6 methodology.
    Our tests show that Experiment 6 probes were effectively trained on the sequence end (hs[-1])
    due to token splitting fallback. This position provides the complete checklist state.
    """
    # Probing all labels at the absolute end of the special token block sequence.
    # This is where the model has finished summarizing all slot missing/filled states.
    probs = run_probe_prediction(hs[-1], W, b)
    return {d: probs[d] for d in DIRS}

def main():
    model_name = "google/gemma-2-2b-it"
    layer_idx = 8
    model_path = "runs/balanced/experiment6_lodo/lodo_query_layer8_multiwoz_restaurant.pt"
    
    # Restaurant booking scenario: WHEN is clearly missing until T3
    scenario_name = "restaurant_booking_multiwoz_style"
    turns = [
        {"turn_id": 1, "speaker": "User", "text": "I need to book a table at a restaurant."},
        {"turn_id": 2, "speaker": "User", "text": "I am looking for a restaurant in the centre of Cambridge."},
        {"turn_id": 3, "speaker": "User", "text": "I need a table for Saturday at 18:30."},
        {"turn_id": 4, "speaker": "User", "text": "There will be 4 of us."}
    ]
    
    output_dir = Path("runs/balanced/experiment9_sentence")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sentence_trajectory.json"
    
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
            
            # Append this turn's text to the context as a single continuous paragraph
            if current_context:
                current_context += " "
            current_context += turn['text']
            
            # Append queries to simulate the full state at the END of this turn
            probe_text = current_context + " [WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?] [WHICH?]"
            
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
                "current_context": current_context,
                "probs": turn_probs
            })
            
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved sentence trajectory data to {out_file}")

if __name__ == "__main__":
    main()
