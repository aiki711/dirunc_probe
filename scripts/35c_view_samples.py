import json
import re

def main():
    with open("pilot_results_v2.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("# Pilot Results Summary\n")
    
    # 1. Qualitative Evaluation (Samples)
    datasets = ["qasrl", "multiwoz", "sgd"]
    for ds in datasets:
        print(f"## Dataset: {ds}")
        ds_samples = [d for d in data if d["dataset"] == ds][:3]
        for i, d in enumerate(ds_samples):
            print(f"### Sample {i+1} (Role: {d['role']})")
            print(f"**Filled**: {d['filled']}")
            print(f"**Mechanical Missing**: {d['mech_missing']}")
            print(f"**LLM Natural Missing**: {d['llm_missing']}")
            print(f"**LLM Self-Eval**: {d['eval']}")
            print(f"**PPL (Filled/Mech/LLM)**: {d['ppl_filled']:.1f} / {d['ppl_mech']:.1f} / {d['ppl_llm']:.1f}")
            print("\n---\n")

    # 2. Statistics
    print("## Quantitative Scores")
    scores = {"nat": [], "om": [], "min": []}
    for d in data:
        n = re.search(r"Naturalness: (\d)", d["eval"])
        o = re.search(r"Omission: (\d)", d["eval"])
        m = re.search(r"MinimalChange: (\d)", d["eval"])
        if n: scores["nat"].append(int(n.group(1)))
        if o: scores["om"].append(int(o.group(1)))
        if m: scores["min"].append(int(m.group(1)))

    if scores["nat"]:
        print(f"Average Naturalness: {sum(scores['nat'])/len(scores['nat']):.2f} / 5.0")
        print(f"Average Omission:    {sum(scores['om'])/len(scores['om']):.2f} / 5.0")
        print(f"Average Minimal:     {sum(scores['min'])/len(scores['min']):.2f} / 5.0")

if __name__ == "__main__":
    main()
