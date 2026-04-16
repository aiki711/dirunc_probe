import json
import numpy as np
from pathlib import Path

def evaluate_minimal_pairs(json_path: str):
    if not Path(json_path).exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    drops = 0
    meaningful_drops = 0 # P(B) < 0.5 * P(A)
    resolved_success = 0 # P(B) < 0.1 (completely resolved)
    
    results = []

    print(f"=== Minimal Pair Baseline Evaluation ({total} pairs) ===")
    print(f"{'Label':<10} | {'Prob A':<10} | {'Prob B':<10} | {'Status':<15}")
    print("-" * 55)

    for item in data:
        pa = item["A_prob"]
        pb = item["B_prob"]
        label = item["label"]
        
        is_drop = pb < pa
        is_meaningful = pb < (pa * 0.5)
        is_resolved = pb < 0.1
        
        if is_drop: drops += 1
        if is_meaningful: meaningful_drops += 1
        if is_resolved: resolved_success += 1
        
        status = "SUCCESS" if is_resolved else ("DROPPED" if is_drop else "FAILED")
        print(f"{label:<10} | {pa:>9.2%} | {pb:>9.2%} | {status:<15}")

    print("-" * 55)
    print(f"Total Pairs: {total}")
    print(f"Drop Rate (P(B) < P(A)): {drops}/{total} ({drops/total:.1%})")
    print(f"Meaningful Drop Rate (P(B) < 0.5*P(A)): {meaningful_drops}/{total} ({meaningful_drops/total:.1%})")
    print(f"Resolved Success Rate (P(B) < 10%): {resolved_success}/{total} ({resolved_success/total:.1%})")
    print("==============================================")

if __name__ == "__main__":
    evaluate_minimal_pairs("runs/balanced/experiment7_neurons_ft/activation_shifts.json")
