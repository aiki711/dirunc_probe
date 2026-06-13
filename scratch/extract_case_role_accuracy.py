import json
from pathlib import Path
import numpy as np

# CONFIGURATIONS
RUNS_DIR = Path("runs")
CONFIGS = {
    "Query (Aligned)": "layer_sweep_gemini_nq_aligned",
    "Final (Aligned)": "layer_sweep_gemini_final_token_aligned"
}
OMISSIONS = ["soft", "strong"]
LAYERS = [0, 4, 8, 12, 16, 20, 24, 26]
CASE_ROLES = ["Agent", "Goal", "Location", "Manner", "Source", "Theme", "Time"]

def get_best_layer_data(config_dir_name: str, omission: str):
    best_layer = -1
    best_f1 = -1.0
    best_epoch_data = None

    config_path = RUNS_DIR / config_dir_name
    for layer in LAYERS:
        log_file = config_path / f"{omission}_layer_{layer}" / "log.jsonl"
        if not log_file.exists():
            continue
        
        # Read all epochs
        epochs = []
        with log_file.open("r") as f:
            for line in f:
                if line.strip():
                    epochs.append(json.loads(line))
        
        if not epochs:
            continue
        
        # Find best epoch for this layer based on macro_f1
        best_ep_idx = np.argmax([ep["macro_f1"] for ep in epochs])
        best_ep_data = epochs[best_ep_idx]
        
        if best_ep_data["macro_f1"] > best_f1:
            best_f1 = best_ep_data["macro_f1"]
            best_layer = layer
            best_epoch_data = best_ep_data
            
    return best_layer, best_epoch_data

def main():
    results = {}
    
    # Extract data
    for omission in OMISSIONS:
        results[omission] = {}
        for config_name, dir_name in CONFIGS.items():
            best_layer, best_data = get_best_layer_data(dir_name, omission)
            results[omission][config_name] = {
                "layer": best_layer,
                "data": best_data
            }
            
    # Print summary of best layers
    print("# Best Layers Identified")
    for omission in OMISSIONS:
        print(f"## Omission: {omission.upper()}")
        for config_name in CONFIGS.keys():
            layer = results[omission][config_name]["layer"]
            f1 = results[omission][config_name]["data"]["macro_f1"]
            print(f"- {config_name}: Best Layer = {layer} (Macro F1 = {f1:.4f})")
    print("\n" + "="*80 + "\n")

    # Generate Markdown Table for Strict Pair Accuracy
    print("# Case Role: Strict Pair Accuracy (2x2 Comparison)")
    print("| Case Role | Soft / Query Aligned | Soft / Final Aligned | Strong / Query Aligned | Strong / Final Aligned |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    
    # We also want to know 'n' for each role (should be same across configs, but check Soft/Strong differences)
    for role in CASE_ROLES:
        row_str = f"| {role} "
        
        # Fetch n and accuracies
        cells = []
        for omission in OMISSIONS:
            for config_name in CONFIGS.keys():
                config_res = results[omission][config_name]
                layer = config_res["layer"]
                data = config_res["data"]
                
                role_data = data.get("by_case_role", {}).get(role, {})
                n = role_data.get("n", 0)
                strict_acc = role_data.get("pair_acc_strict", 0.0)
                std_acc = role_data.get("pair_acc_standard", 0.0)
                
                cells.append(f"{std_acc:.3f} / {strict_acc:.3f} (n={n})")
        
        row_str += "| " + " | ".join(cells) + " |"
        print(row_str)
        
    print("\nNote: Table cells show 'Standard Pair Acc / Strict Pair Acc (n=sample size)' at the best performing layer for each configuration.")

if __name__ == "__main__":
    main()
