import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Ensure common.py is accessible
sys.path.append(os.getcwd())
from scripts.common import strip_query_tokens

def get_final_token_positions(attention_mask):
    # attention_mask: (B, L)
    return attention_mask.sum(dim=1) - 1

def main():
    model_name = "google/gemma-2-2b-it"
    input_file = "data/processed/sgd/dirunc_balanced/semantic_minimal_pairs_v3.jsonl"
    output_dir = Path("runs/latent_analysis")
    output_file = output_dir / "latent_states.pt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers_to_extract = [8, 16, 24]
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    lm = AutoModel.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(device)
    lm.eval()
    
    pairs = []
    with open(input_file, "r") as f:
        for line in f:
            pairs.append(json.loads(line))
            
    print(f"Extracting states for {len(pairs)} pairs (Batched size={batch_size})...")
    
    data = {
        layer: {
            "A": [],
            "B": [],
            "label": [],
            "domain": []
        } for layer in layers_to_extract
    }
    
    # Batch processing
    for i in tqdm(range(0, len(pairs), batch_size)):
        batch_pairs = pairs[i : i + batch_size]
        
        texts_a = [strip_query_tokens(p["text_A"]).strip() for p in batch_pairs]
        texts_b = [strip_query_tokens(p["text_B"]).strip() for p in batch_pairs]
        labels = [p["direction"] for p in batch_pairs]
        domains = [p.get("domain", "unknown") for p in batch_pairs]
        
        with torch.no_grad():
            # Process A
            enc_a = tokenizer(texts_a, return_tensors="pt", padding=True).to(device)
            out_a = lm(**enc_a, output_hidden_states=True)
            pos_a = get_final_token_positions(enc_a["attention_mask"])
            
            # Process B
            enc_b = tokenizer(texts_b, return_tensors="pt", padding=True).to(device)
            out_b = lm(**enc_b, output_hidden_states=True)
            pos_b = get_final_token_positions(enc_b["attention_mask"])
            
            for layer in layers_to_extract:
                # Extract for each sample in batch
                for idx in range(len(batch_pairs)):
                    h_a = out_a.hidden_states[layer + 1][idx, pos_a[idx]].cpu().to(torch.float32)
                    h_b = out_b.hidden_states[layer + 1][idx, pos_b[idx]].cpu().to(torch.float32)
                    
                    data[layer]["A"].append(h_a)
                    data[layer]["B"].append(h_b)
                    data[layer]["label"].append(labels[idx])
                    data[layer]["domain"].append(domains[idx])
                
    # Stack tensors
    for layer in layers_to_extract:
        data[layer]["A"] = torch.stack(data[layer]["A"])
        data[layer]["B"] = torch.stack(data[layer]["B"])
        
    print(f"Saving combined latent states to {output_file}...")
    torch.save(data, output_file)
    print("Done!")

if __name__ == "__main__":
    main()
