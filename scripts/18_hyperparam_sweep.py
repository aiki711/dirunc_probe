import os
import subprocess
import json
import csv
import torch
from pathlib import Path
from itertools import product
from tqdm import tqdm

# Hyperparameter Grid
GRID = {
    "lr": [5e-4, 1e-3],
    "mask": [0.5, 0.7],
    "pos_w": [1.0, 4.0],
    "epochs": [3, 10]
}

DATA_JSONL = "data/processed/sgd/dirunc_balanced/train.jsonl"
TEST_DOMAIN = "sgd_Events_1"
LAYER_IDX = 8
SAVE_DIR = "runs/sweep"
RESULTS_CSV = "runs/sweep_results.csv"
PYTHON_BIN = "/home/s2550009/anaconda3/envs/dirunc_probe/bin/python"

# Minimal Pais for Evaluation
PAIRS_PATH = "runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json"

def run_training(lr, mask, pos_w, epochs, model_tag):
    # Note: Using the newly added --pos_weight_mult flag
    cmd = [
        PYTHON_BIN, "scripts/16_train_probe_improved.py",
        "--data_jsonl", DATA_JSONL,
        "--test_domain", TEST_DOMAIN,
        "--save_dir", SAVE_DIR,
        "--layer_idx", str(LAYER_IDX),
        "--batch_size", "32",
        "--learning_rate", str(lr),
        "--mask_prob", str(mask),
        "--pos_weight_mult", str(pos_w),
        "--num_epochs", str(epochs),
        "--seed", "42"
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    return Path(SAVE_DIR) / f"improved_lodo_layer{LAYER_IDX}_{TEST_DOMAIN}.pt"

def evaluate_strict(model_path):
    # This calls 17 or uses its logic to return a single 'Strict Accuracy' score
    import sys
    sys.path.append(os.getcwd())
    from scripts.common import DIRS
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    # Load model and tokenizer once if possible, but for sweep we keep it simple
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    lm = AutoModel.from_pretrained("google/gemma-2-2b-it", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    
    weights = torch.load(model_path, map_location="cpu")
    W = weights["W"].to(device).to(torch.bfloat16)
    b = weights["b"].to(device).to(torch.bfloat16)
    
    with open(PAIRS_PATH, "r") as f:
        pairs = json.load(f)
        
    import importlib.util
    spec = importlib.util.spec_from_file_location("eval_mod", "scripts/17_compare_models.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    evaluate_model = mod.evaluate_model
    
    results = evaluate_model(lm, tokenizer, W, b, LAYER_IDX, pairs, device)
    
    total = len(results)
    # Strict Success: P(A) > 0.8 AND P(B) < 0.1 AND P(B_pert) < 0.1
    strict_count = 0
    for r in results:
        if r["pa"] > 0.8 and r["pb"] < 0.1 and r["pb_pert"] < 0.1:
            strict_count += 1
            
    return strict_count / total

def main():
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    keys = GRID.keys()
    combinations = list(product(*GRID.values()))
    
    print(f"Starting Hyperparameter Sweep: {len(combinations)} combinations")
    
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lr", "mask", "pos_w", "epochs", "strict_acc"])
        
        for lr, mask, pos_w, epochs in tqdm(combinations):
            print(f"\nTraining: LR={lr}, Mask={mask}, Epochs={epochs}")
            try:
                model_path = run_training(lr, mask, pos_w, epochs, f"lr{lr}_m{mask}_e{epochs}")
                score = evaluate_strict(model_path)
                print(f"-> Strict Accuracy: {score:.1%}")
                writer.writerow([lr, mask, pos_w, epochs, score])
                f.flush()
            except Exception as e:
                print(f"Error in combination: {e}")

if __name__ == "__main__":
    main()
