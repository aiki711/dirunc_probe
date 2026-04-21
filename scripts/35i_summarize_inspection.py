import json
from pathlib import Path
from collections import defaultdict

def main():
    inp_path = Path("analysis_results/inspection_results.jsonl")
    out_path = Path("analysis_results/prediction_inspection_report.md")
    
    if not inp_path.exists():
        print("Input file not found.")
        return

    data = []
    with inp_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("dataset") == "qasrl":
                continue
            data.append(d)

    # 1. Dataset-wise & Overall Statistics
    ds_stats = defaultdict(lambda: {"total": 0, "correct_strict": 0, "correct_pair": 0})
    overall_total = 0
    overall_correct_strict = 0
    overall_correct_pair = 0
    
    # 2. Case-role-wise Statistics & Sampling
    roles_stats = defaultdict(lambda: {"total": 0, "correct_strict": 0, "correct_pair": 0})
    roles = defaultdict(lambda: {"TP": [], "FN": []})
    
    for d in data:
        ds = d["dataset"]
        role = d["case_role"]
        
        is_pair_correct = d["prob_missing"] > d["prob_filled"]
        
        # Overall
        overall_total += 1
        ds_stats[ds]["total"] += 1
        roles_stats[role]["total"] += 1
        
        if is_pair_correct:
            overall_correct_pair += 1
            ds_stats[ds]["correct_pair"] += 1
            roles_stats[role]["correct_pair"] += 1

        if d["is_correct"]: # Strict criteria (threshold based)
            overall_correct_strict += 1
            ds_stats[ds]["correct_strict"] += 1
            roles_stats[role]["correct_strict"] += 1
            roles[role]["TP"].append(d)
        else:
            roles[role]["FN"].append(d)

    # Sort to prioritize "Confident" samples
    for role in roles:
        roles[role]["TP"].sort(key=lambda x: x["confidence_diff"], reverse=True)
        roles[role]["FN"].sort(key=lambda x: x["confidence_diff"])

    # 3. Generate Markdown Report
    content = ["# プローブ予測結果の詳細分析 (Qualitative Analysis Report)\n"]
    content.append("学習済みの対照プローブ（Layer 16）による、自然な対照文（Natural Minimal Pairs）の予測結果を定性的に分析しました。\n")
    content.append("> [!NOTE]\n> 本分析では、ノイズの多い QA-SRL データセットを除外し、MultiWOZ および SGD データセットのみを対象としています。\n")
    
    content.append("## 1. 精度統計")
    content.append("- **Strict Accuracy**: `prob_missing >= threshold` かつ `prob_filled < threshold` (実運用を想定)")
    content.append("- **Pair Accuracy**: `prob_missing > prob_filled` (モデルの内部分離能力を評価)\n")
    
    content.append("### 1.1 全体およびデータセット別")
    content.append("| Dataset | Total Pairs | Strict Acc | Pair Acc |")
    content.append("| :--- | :--- | :--- | :--- |")
    for ds in sorted(ds_stats.keys()):
        s = ds_stats[ds]
        acc_s = s["correct_strict"] / s["total"] if s["total"] > 0 else 0
        acc_p = s["correct_pair"] / s["total"] if s["total"] > 0 else 0
        content.append(f"| {ds} | {s['total']} | {acc_s:.2%} | {acc_p:.2%} |")
    
    overall_acc_s = overall_correct_strict / overall_total if overall_total > 0 else 0
    overall_acc_p = overall_correct_pair / overall_total if overall_total > 0 else 0
    content.append(f"| **OVERALL** | **{overall_total}** | **{overall_acc_s:.2%}** | **{overall_acc_p:.2%}** |")
    content.append("\n")

    content.append("### 1.2 格役割別精度")
    content.append("| Case Role | Total Pairs | Strict Acc | Pair Acc |")
    content.append("| :--- | :--- | :--- | :--- |")
    for role in sorted(roles_stats.keys()):
        if not role: continue
        s = roles_stats[role]
        acc_s = s["correct_strict"] / s["total"] if s["total"] > 0 else 0
        acc_p = s["correct_pair"] / s["total"] if s["total"] > 0 else 0
        content.append(f"| {role} | {s['total']} | {acc_s:.2%} | {acc_p:.2%} |")
    content.append("\n")

    content.append("## 2. 格役割別の具体例分析")
    content.append("各格役割において、モデルが自信を持って正解した例（TP）と、自信を持って間違えた（情報の欠落を見逃した）例（FN）を2例ずつ抽出しました。\n")

    for role in sorted(roles.keys()):
        if not role: continue
        content.append(f"### 格役割: {role}")
        
        # TP Examples
        content.append("#### ✅ 正解例 (True Positive) - 自信度の高い順")
        tp_samples = roles[role]["TP"][:2]
        if not tp_samples:
            content.append("*該当なし*")
        for i, s in enumerate(tp_samples):
            content.append(f"**例 {i+1} (Source: {s['dataset']})**")
            content.append(f"- **Filled**: {s['filled_text']}")
            content.append(f"- **Missing**: {s['missing_text']}")
            content.append(f"- **Prediction**: Missing Prob={s['prob_missing']:.3f}, Filled Prob={s['prob_filled']:.3f} (Diff={s['confidence_diff']:.3f})")
            content.append("")

        # FN Examples
        content.append("#### ❌ 不正解例 (False Negative) - 見逃し・自信のある誤判定")
        fn_samples = roles[role]["FN"][:2]
        if not fn_samples:
            content.append("*該当なし*")
        for i, s in enumerate(fn_samples):
            content.append(f"**例 {i+1} (Source: {s['dataset']})**")
            content.append(f"- **Filled**: {s['filled_text']}")
            content.append(f"- **Missing**: {s['missing_text']}")
            content.append(f"- **Prediction**: Missing Prob={s['prob_missing']:.3f}, Filled Prob={s['prob_filled']:.3f} (Diff={s['confidence_diff']:.3f})")
            content.append("")
        
        content.append("---\n")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"Report generated at {out_path}")

if __name__ == "__main__":
    main()
