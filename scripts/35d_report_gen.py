import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

def generate_report():
    with open("pilot_results_v5.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    ds_stats = defaultdict(lambda: {
        "ppl_filled":[], "ppl_mech":[], "ppl_llm":[], 
        "nat_llm":[], "om_llm":[], "min_llm":[],
        "nat_filled":[], "om_filled":[], "min_filled":[],
        "nat_mech":[], "om_mech":[], "min_mech":[]
    })
    samples = defaultdict(list)

    for d in data:
        ds = d["dataset"]
        ds_stats[ds]["ppl_filled"].append(d["ppl_filled"])
        ds_stats[ds]["ppl_mech"].append(d["ppl_mech"])
        ds_stats[ds]["ppl_llm"].append(d["ppl_llm"])
        
        # Fair Evaluation Score (Iteration 4 / v5)
        def parse_scores(eval_text):
            n = re.search(r"Naturalness: (\d)", eval_text)
            o = re.search(r"Omission: (\d)", eval_text)
            m = re.search(r"MinimalChange: (\d)", eval_text)
            return (int(n.group(1)) if n else 0, 
                    int(o.group(1)) if o else 0, 
                    int(m.group(1)) if m else 0)

        n_l, o_l, m_l = parse_scores(d["eval_natural_v5"])
        if n_l: ds_stats[ds]["nat_llm"].append(n_l)
        if o_l: ds_stats[ds]["om_llm"].append(o_l)
        if m_l: ds_stats[ds]["min_llm"].append(m_l)

        n_f, o_f, m_f = parse_scores(d["eval_filled_v5"])
        if n_f: ds_stats[ds]["nat_filled"].append(n_f)
        if o_f: ds_stats[ds]["om_filled"].append(o_f)
        if m_f: ds_stats[ds]["min_filled"].append(m_f)

        n_m, o_m, m_m = parse_scores(d["eval_mech_v5"])
        if n_m: ds_stats[ds]["nat_mech"].append(n_m)
        if o_m: ds_stats[ds]["om_mech"].append(o_m)
        if m_m: ds_stats[ds]["min_mech"].append(m_m)
        
        if len(samples[ds]) < 5:
            samples[ds].append(d)

    report_path = Path("experiment_report_pilot_v5.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Case Grammar Pilot Generation (Final Fair Comparison v5)\n\n")
        f.write("This report provides a truly fair comparison by cleaning the Mechanical version strings to match the length of Filled and LLM versions, and providing ALL to the evaluator with the same separate Context field.\n\n")

        f.write("## 1. Quantitative Analysis by Dataset\n\n")
        f.write("| Dataset | Metric | Filled (Baseline) | Mechanical Ablation | **LLM (Natural)** |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")

        for ds in sorted(ds_stats.keys()):
            s = ds_stats[ds]
            f.write(f"| {ds} | Average PPL | {statistics.mean(s['ppl_filled']):.2f} | {statistics.mean(s['ppl_mech']):.2f} | {statistics.mean(s['ppl_llm']):.2f} |\n")
            f.write(f"| | Naturalness | {statistics.mean(s['nat_filled']):.2f} | {statistics.mean(s['nat_mech']):.2f} | **{statistics.mean(s['nat_llm']):.2f}** |\n")
            f.write(f"| | Omission | {statistics.mean(s['om_filled']):.2f} | {statistics.mean(s['om_mech']):.2f} | **{statistics.mean(s['om_llm']):.2f}** |\n")
            f.write(f"| | MinimalChange | {statistics.mean(s['min_filled']):.2f} | {statistics.mean(s['min_mech']):.2f} | **{statistics.mean(s['min_llm']):.2f}** |\n")

        f.write("\n## 2. Qualitative Analysis (Samples)\n\n")
        for ds in sorted(samples.keys()):
            f.write(f"### {ds.upper()} Samples\n\n")
            for i, d in enumerate(samples[ds]):
                ctx = d.get('context', 'None')
                f_eval = d['eval_filled_v5'].replace('\n', ' ')
                m_eval = d['eval_mech_v5'].replace('\n', ' ')
                l_eval = d['eval_natural_v5'].replace('\n', ' ')
                f.write(f"#### Sample {i+1} (Role: {d['role']})\n")
                f.write(f"- **Context**: `{ctx}`\n")
                f.write(f"- **Filled**: `{d['filled'].strip()}` (N/O/M: {f_eval})\n")
                f.write(f"- **Mech Missing**: `{d['mech_missing_clean']}` (N/O/M: {m_eval})\n")
                f.write(f"- **LLM Missing**: `{d['llm_missing'].strip()}` (N/O/M: {l_eval})\n")
                f.write(f"- **PPL (F/M/L)**: {d['ppl_filled']:.1f} / {d['ppl_mech']:.1f} / {d['ppl_llm']:.1f}\n\n")
            f.write("---\n\n")



    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    generate_report()
