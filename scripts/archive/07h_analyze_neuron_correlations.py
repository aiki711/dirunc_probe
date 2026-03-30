import json
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    data_path = Path("runs/balanced/experiment7_neurons/corpus_shift_data.json")
    if not data_path.exists():
        print(f"Data not found at {data_path}")
        return
        
    with open(data_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    print("="*100)
    print("EXPERIMENT 7 ENRICHED ANALYSIS: CORRELATIONS & CROSS-LABEL INSIGHTS")
    print("="*100)

    # 1. Neuraon Co-occurrence / Sharing
    neuron_to_labels = {}
    for label, data in all_results.items():
        indices = data.get("neuron_indices", [])
        for idx in indices:
            if idx not in neuron_to_labels:
                neuron_to_labels[idx] = []
            neuron_to_labels[idx].append(label)

    print("\n[PART 1] GENERALIST NEURONS (Neurons active across multiple labels)")
    print("-" * 60)
    shared_neurons = {k: v for k, v in neuron_to_labels.items() if len(v) > 1}
    sorted_shared = sorted(shared_neurons.items(), key=lambda x: len(x[1]), reverse=True)
    
    print(f"{'Neuron ID':<10} | {'Count':<5} | {'Labels'}")
    for nid, labels in sorted_shared:
        print(f"n{nid:<9} | {len(labels):<5} | {', '.join(labels)}")

    # 2. Delta Probability vs Delta Activation Correlation
    print("\n[PART 2] CAUSAL CORRELATION (ΔProb vs ΔActivation)")
    print("-" * 80)
    print(f"{'Label':<10} | {'Neuron':<8} | {'Pearson R':<10} | {'p-value':<10} | {'Significance'}")
    
    for label, data in all_results.items():
        pairs = [p for p in data.get("pairs", []) if p.get("valid")]
        top_neurons = data.get("neuron_indices", [])
        
        if not pairs: continue
        
        # Calculate Delta Prob (B_prob - A_prob)
        # Note: B_prob is Filled (usually low), A_prob is Missing (usually high)
        # So ΔProb is usually negative.
        d_probs = [p["B_prob"] - p["A_prob"] for p in pairs]
        
        for i, neuron_idx in enumerate(top_neurons):
            acts_a = [p["A_activations"][i] for p in pairs]
            acts_b = [p["B_activations"][i] for p in pairs]
            d_acts = [b - a for a, b in zip(acts_a, acts_b)]
            
            r, p_val = stats.pearsonr(d_acts, d_probs)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
            
            # For HOW, we expect r > 0 (both increase or both decrease)
            # For others, we expect r > 0 if Suppression (both decrease)
            print(f"{label:<10} | n{neuron_idx:<7} | {r:>10.3f} | {p_val:>10.1e} | {sig}")

    # 3. High Impact Samples (The "Smoking Guns")
    print("\n[PART 3] HIGH IMPACT SAMPLES (Strongest Suppression + Prob Drop)")
    print("-" * 100)
    
    for label in ["when", "where", "who", "what"]:
        if label not in all_results: continue
        data = all_results[label]
        pairs = [p for p in data.get("pairs", []) if p.get("valid")]
        if not pairs: continue
        
        # Use Tip 1 neuron for correlation
        # Sorting by (ΔProb * ΔAct) to find cases where both dropped significantly
        # Since both are negative, product is positive.
        def impact_score(p):
            # Focus on the first neuron
            d_act = p["B_activations"][0] - p["A_activations"][0]
            d_prob = p["B_prob"] - p["A_prob"]
            if d_act < 0 and d_prob < 0:
                return d_act * d_prob
            return 0

        sorted_pairs = sorted(pairs, key=impact_score, reverse=True)
        
        print(f"\nTop Impact Samples for {label.upper()}:")
        print(f"{'Dialogue ID':<15} | {'Turns':<8} | {'ΔAct (n1)':<10} | {'ΔProb':<10}")
        for p in sorted_pairs[:3]:
            d_act = p["B_activations"][0] - p["A_activations"][0]
            d_prob = p["B_prob"] - p["A_prob"]
            print(f"{p['dialogue_id']:<15} | {p['turn_a']:>2}->{p['turn_b']:<4} | {d_act:>10.3f} | {d_prob:>10.3f}")

if __name__ == "__main__":
    main()
