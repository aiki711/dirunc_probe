import json
import math
from pathlib import Path
from scipy import stats
import numpy as np

def draw_distribution_bar(mean_val, std_val, max_scale=3.0, length=30):
    if math.isnan(mean_val): return "[No Data]"
    
    # Position of 0
    zero_pos = length // 2
    
    # Scale to positions
    mean_pos = int((mean_val / max_scale) * (length//2)) + zero_pos
    mean_pos = max(0, min(length, mean_pos))
    
    std_width = int((std_val / max_scale) * (length//2))
    std_min = max(0, mean_pos - std_width)
    std_max = min(length, mean_pos + std_width)
    
    chars = [" "] * (length + 1)
    chars[zero_pos] = "|"
    
    # Draw std dev range
    for i in range(std_min, std_max + 1):
        if chars[i] == " ": chars[i] = "-"
        
    # Draw mean
    chars[mean_pos] = "O" if mean_val <= 0 else "X"
    
    return "".join(chars)

def main():
    data_path = Path("runs/balanced/experiment7_neurons/corpus_shift_data.json")
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    print("="*90)
    print(f"{'Label (Neuron)':<15} | {'N':<4} | {'Mean ΔAct':<10} | {'p-value':<10} | {'Suppr %':<8} | {'Distribution (-3.0 to +3.0)':<30}")
    print("-" * 90)

    for label, data in all_results.items():
        pairs = data.get("pairs", [])
        top_neurons = data.get("neuron_indices", [])
        
        valid_pairs = [p for p in pairs if p.get("valid")]
        n_samples = len(valid_pairs)
        
        if n_samples == 0:
            continue
            
        print(f"--- {label.upper()} ({n_samples} valid pairs) ---")
        
        # Analyze each tracked neuron
        for i, neuron_idx in enumerate(top_neurons):
            acts_a = [p["A_activations"][i] for p in valid_pairs]
            acts_b = [p["B_activations"][i] for p in valid_pairs]
            
            # Calculates deltas (B - A). Negative means suppression.
            deltas = [b - a for a, b in zip(acts_a, acts_b)]
            
            mean_delta = np.mean(deltas)
            std_delta = np.std(deltas)
            
            # Suppression Rate (% of negative deltas)
            suppr_count = sum(1 for d in deltas if d < -0.05)
            suppr_rate = (suppr_count / n_samples) * 100
            
            # Paired t-test
            t_stat, p_val = stats.ttest_rel(acts_b, acts_a)
            
            # Format p-value and significance star
            p_str = f"{p_val:.1e}"
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            p_display = f"{p_str}{sig}"
            
            bar = draw_distribution_bar(mean_delta, std_delta)
            
            row = f"n{neuron_idx:<13} | {n_samples:<4} | {mean_delta:>8.3f}   | {p_display:<10} | {suppr_rate:>5.1f}%   | [{bar}]"
            print(row)
            
        print("-" * 90)
        
    print("Legend: O = Mean (Negative/Suppressed), X = Mean (Positive/Increased), - = 1 Std Dev")
    print("        * p<0.05, ** p<0.01, *** p<0.001 (Paired t-test)")
    print("="*90)

if __name__ == "__main__":
    main()
