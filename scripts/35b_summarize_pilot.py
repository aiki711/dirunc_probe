import json
from collections import defaultdict

def main():
    with open("pilot_results_v2.json", "r", encoding="utf-8") as f:

        data = json.load(f)

    print(f"Total processed: {len(data)}")
    
    avg_ppl_filled = sum(d["ppl_filled"] for d in data) / len(data)
    avg_ppl_mech = sum(d["ppl_mech"] for d in data) / len(data)
    avg_ppl_llm = sum(d["ppl_llm"] for d in data) / len(data)

    print(f"Average PPL (Filled): {avg_ppl_filled:.2f}")
    print(f"Average PPL (Mechanical Ablation): {avg_ppl_mech:.2f}")
    print(f"Average PPL (LLM Natural Rewriting): {avg_ppl_llm:.2f}")

    # Dataset specific PPL
    ds_stats = defaultdict(lambda: {"filled": 0, "mech": 0, "llm": 0, "count": 0})
    for d in data:
        ds = d["dataset"]
        ds_stats[ds]["filled"] += d["ppl_filled"]
        ds_stats[ds]["mech"] += d["ppl_mech"]
        ds_stats[ds]["llm"] += d["ppl_llm"]
        ds_stats[ds]["count"] += 1

    for ds, stats in ds_stats.items():
        n = stats["count"]
        print(f"\n[{ds}] (n={n})")
        print(f"  PPL Filled: {stats['filled']/n:.2f}")
        print(f"  PPL Mech:   {stats['mech']/n:.2f}")
        print(f"  PPL LLM:    {stats['llm']/n:.2f}")

    # Success rate (PPL improvement)
    improved = sum(1 for d in data if d["ppl_llm"] < d["ppl_mech"])
    print(f"\nPPL Improvement Rate (LLM < Mech): {improved/len(data)*100:.1f}%")

if __name__ == "__main__":
    main()
