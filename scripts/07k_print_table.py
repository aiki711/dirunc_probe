import json
import numpy as np
from scipy import stats

def get_stats(path):
    with open(path) as f:
        data = json.load(f)
    stats_dict = {}
    for lbl, d in data.items():
        valid_pairs = [p for p in d.get("pairs",[]) if p.get("valid")]
        top = d.get("neuron_indices",[])
        if not valid_pairs or not top: continue
        acts_a = [p["A_activations"][0] for p in valid_pairs]
        acts_b = [p["B_activations"][0] for p in valid_pairs]
        deltas = [b-a for a,b in zip(acts_a, acts_b)]
        mean = np.mean(deltas)
        _, p = stats.ttest_rel(acts_b, acts_a)
        stats_dict[lbl.upper()] = (top[0], mean, p)
    return stats_dict

try:
    d_lodo = get_stats("runs/balanced/experiment7_neurons/corpus_shift_data.json")
    d_ft = get_stats("runs/balanced/experiment7_neurons_ft/corpus_shift_data.json")

    print("| Label | Standard (LODO) Top Neuron | ΔAct (LODO) | p-value (LODO) | Final Token (FT) Top Neuron | ΔAct (FT) | p-value (FT) |")
    print("|---|---|---|---|---|---|---|")

    all_lbls = sorted(list(set(d_lodo.keys()) | set(d_ft.keys())))
    for lbl in all_lbls:
        lodo = d_lodo.get(lbl)
        ft = d_ft.get(lbl)
        
        lodo_str = f"n{lodo[0]} | {lodo[1]:+.3f} | {lodo[2]:.2e}" if lodo else "- | - | -"
        ft_str = f"n{ft[0]} | {ft[1]:+.3f} | {ft[2]:.2e}" if ft else "- | - | -"
        
        print(f"| **{lbl}** | {lodo_str} | {ft_str} |")
except Exception as e:
    print(f"Error: {e}")
