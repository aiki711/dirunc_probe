import json
from pathlib import Path

def draw_bar(val, max_val=5.0, length=20):
    if val == 0: return "|" + " " * length + "|"
    # Normalize
    scaled = int(abs(val) / max_val * length)
    scaled = min(scaled, length)
    bar = "=" * scaled + " " * (length - scaled)
    sign = "+" if val > 0 else "-"
    return f"|{bar}| ({sign}{abs(val):.2f})"

def main():
    results_path = "runs/balanced/experiment7_neurons/activation_shifts.json"
    if not Path(results_path).exists():
        print("Results file not found.")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    print("="*80)
    print(f"{'Label':<10} | {'Mode':<8} | {'Prob':<8} | {'Top Neuron Activations (ASCII)':<40}")
    print("-"*80)

    for item in data:
        label = item["label"]
        top_neurons = item["neuron_indices"]
        
        # We'll just show the top 3 neurons for brevity in ASCII
        for i in range(min(3, len(top_neurons))):
            idx = top_neurons[i]
            act_a = item["A_activations"][i]
            act_b = item["B_activations"][i]
            shift = act_b - act_a
            
            row_label = f"{label} n{idx}" if i == 0 else f"      n{idx}"
            
            # Line for A (Missing)
            p_a = f"{item['A_prob']:.2%}" if i == 0 else ""
            print(f"{row_label:<10} | {'Missing':<8} | {p_a:<8} | {draw_bar(act_a)}")
            
            # Line for B (Filled)
            p_b = f"{item['B_prob']:.2%}" if i == 0 else ""
            print(f"{'':<10} | {'Filled':<8} | {p_b:<8} | {draw_bar(act_b)}")
            
            # Shift indicator
            status = "SUPPRESSED (鎮火)" if shift < -0.05 else ("INCREASED" if shift > 0.05 else "STABLE")
            print(f"{'':<10} | {'SHIFT':<8} | {'':<8} | Δ: {shift:+.3f} [{status}]")
            print("-" * 10 + " + " + "-" * 8 + " + " + "-" * 8 + " + " + "-" * 40)

    print("="*80)

if __name__ == "__main__":
    main()
