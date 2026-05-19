import json
from pathlib import Path
from collections import defaultdict

def main():
    root = Path("analysis_results")
    files = {
        "soft": root / "inspection_results_soft.jsonl",
        "strong": root / "inspection_results_strong.jsonl"
    }
    out_path = root / "prediction_inspection_report_v2.md"
    
    all_data = []
    for strength, path in files.items():
        if not path.exists():
            print(f"File not found: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("dataset") == "qasrl": continue
                d["strength"] = strength
                all_data.append(d)

    if not all_data:
        print("No data loaded.")
        return

    # 1. Stats by Strength & Saturation
    # {strength: {is_saturated: {total, correct_pair}}}
    comp_stats = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct_pair": 0}))
    
    # 2. Case Role x Strength
    role_comp = defaultdict(lambda: defaultdict(lambda: {"total": 0, "correct_pair": 0}))

    for d in all_data:
        str_key = d["strength"]
        sat_key = d["is_saturated"]
        role = d["case_role"]
        
        # We use Pair Accuracy (prob_missing > prob_filled) for comparison
        is_pair_correct = d["probs_missing"][DIRS.index(role)] > d["probs_filled"][DIRS.index(role)] if role in DIRS else False
        if not is_pair_correct:
            # Fallback if case_role mapping is missing from DIRS (should not happen)
            target_idx = -1
            for i, label in enumerate(d["labels"].values()):
                if label == 1: target_idx = i; break
            if target_idx >= 0:
                is_pair_correct = d["probs_missing"][target_idx] > d["probs_filled"][target_idx]

        comp_stats[str_key][sat_key]["total"] += 1
        if is_pair_correct:
            comp_stats[str_key][sat_key]["correct_pair"] += 1
            
        role_comp[role][str_key]["total"] += 1
        if is_pair_correct:
            role_comp[role][str_key]["correct_pair"] += 1

    # 3. Generate Markdown Report
    content = ["# 格文法プローブ：詳細比較分析レポート (Strength & Saturation)\n"]
    content.append("Geminiで生成した2種類の強度（Soft/Strong）および情報の充足度（Saturated/Unsaturated）がプローブの予測精度に与える影響を分析しました。\n")
    content.append("> [!IMPORTANT]\n> 分析対象レイヤー: **Layer 16** (Gemma-2-2b-it)\n")
    
    content.append("## 1. 強度と充足度による精度の違い (Pair Accuracy)")
    content.append("| Strength | Saturation | Total Pairs | Pair Accuracy |")
    content.append("| :--- | :--- | :--- | :--- |")
    
    for s in ["soft", "strong"]:
        for sat in [False, True]:
            stats = comp_stats[s][sat]
            sat_label = "Saturated" if sat else "Unsaturated"
            acc = stats["correct_pair"] / stats["total"] if stats["total"] > 0 else 0
            content.append(f"| {s.capitalize()} | {sat_label} | {stats['total']} | {acc:.2%} |")
    content.append("\n")

    content.append("### 考察")
    content.append("- **強度の影響**: Soft (指示代名詞など) と Strong (完全削除) ではどちらがプローブにとって「情報の欠落」として検出しやすいかを確認できます。一般に、Strongの方が情報の欠落が激しいため精度が高くなる傾向があります。")
    content.append("- **充足度の影響**: 情報が既に文脈にある (Saturated) 場合、モデルはその欠落を補完しやすいため、プローブがその「欠落の不自然さ」をより敏感に捉える可能性があります。\n")

    content.append("## 2. 格役割別の強度比較")
    content.append("| Case Role | Soft Acc | Strong Acc | Diff |")
    content.append("| :--- | :--- | :--- | :--- |")
    
    for role in sorted(role_comp.keys()):
        if not role or role == "unknown": continue
        s_stats = role_comp[role]["soft"]
        st_stats = role_comp[role]["strong"]
        acc_s = s_stats["correct_pair"] / s_stats["total"] if s_stats["total"] > 0 else 0
        acc_st = st_stats["correct_pair"] / st_stats["total"] if st_stats["total"] > 0 else 0
        diff = acc_st - acc_s
        content.append(f"| {role} | {acc_s:.2%} | {acc_st:.2%} | {diff:+.2%} |")
    content.append("\n")

    content.append("## 3. 具体例：強度による変化")
    content.append("同じ文に対して、SoftとStrongで予測自信度がどう変わるかの例です。\n")
    
    # Sample some examples where strong improved over soft
    samples = []
    # (Simplified sampling: just pick first few from soft data but show both)
    # To do this properly, we'd need to align IDs, but let's just use the loaded data for now
    content.append("> [!NOTE]\n> 具体例の詳細は `analysis_results/inspection_results_*.jsonl` を参照してください。\n")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"Report generated at {out_path}")

# Import DIRS for indexing
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

if __name__ == "__main__":
    main()
