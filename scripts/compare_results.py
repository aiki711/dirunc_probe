#!/usr/bin/env python3
"""
実験結果を比較して、改善効果をレポートするスクリプト
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def load_summary(dir_path: Path) -> Dict[str, Any]:
    """サマリJSONファイルを読み込む"""
    summary_file = dir_path / "summary.json"
    if not summary_file.exists():
        # log.jsonlから最良結果を抽出
        log_file = dir_path / "log.jsonl"
        if not log_file.exists():
            return {}
        
        best = None
        with open(log_file, "r") as f:
            for line in f:
                record = json.loads(line)
                if best is None or record.get("macro_f1_posonly", 0) > best.get("macro_f1_posonly", 0):
                    best = record
        return best or {}
    
    with open(summary_file, "r") as f:
        return json.load(f)


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    """サマリから主要なメトリクスを抽出"""
    metrics = {}
    
    # Macro F1 (positive only)
    metrics["macro_f1_posonly"] = summary.get("macro_f1_posonly", 0.0)
    
    # Micro F1
    metrics["micro_f1"] = summary.get("micro_f1", 0.0)
    
    # Per-class F1
    per_class_tuned = summary.get("per_class_tuned", {})
    f1_dict = per_class_tuned.get("f1_dict", {})
    
    if f1_dict:
        metrics.update(f1_dict)
    elif "per_label_f1" in summary:
        # Fallback
        dirs = ["who", "what", "when", "where", "why", "how", "which"]
        per_label_f1 = summary["per_label_f1"]
        for i, d in enumerate(dirs):
            if i < len(per_label_f1):
                metrics[d] = per_label_f1[i]
    
    return metrics


def calculate_improvement(baseline: float, improved: float) -> float:
    """改善率を計算（パーセンテージ）"""
    if baseline == 0:
        return 0.0
    return ((improved - baseline) / baseline) * 100


def generate_comparison_markdown(
    baseline_metrics: Dict[str, float],
    perclass_metrics: Dict[str, float],
    multilayer_metrics: Dict[str, float] = None,
) -> str:
    """比較レポートをMarkdown形式で生成"""
    
    dirs = ["who", "what", "when", "where", "why", "how", "which"]
    
    md = "# プローブモデル精度改善の比較レポート\n\n"
    
    md += "## 概要\n\n"
    md += "このレポートは、方向性不確定性プローブモデルの精度改善結果をまとめたものです。\n\n"
    
    md += "## 全体的なメトリクス\n\n"
    md += "| メトリクス | ベースライン | クラスごと閾値 | 改善率 |\n"
    md += "|-----------|------------|--------------|--------|\n"
    
    # Macro F1
    baseline_macro = baseline_metrics.get("macro_f1_posonly", 0.0)
    perclass_macro = perclass_metrics.get("macro_f1_posonly", 0.0)
    improvement = calculate_improvement(baseline_macro, perclass_macro)
    md += f"| Macro F1 (pos only) | {baseline_macro:.4f} | {perclass_macro:.4f} | {improvement:+.2f}% |\n"
    
    # Micro F1
    baseline_micro = baseline_metrics.get("micro_f1", 0.0)
    perclass_micro = perclass_metrics.get("micro_f1", 0.0)
    improvement_micro = calculate_improvement(baseline_micro, perclass_micro)
    md += f"| Micro F1 | {baseline_micro:.4f} | {perclass_micro:.4f} | {improvement_micro:+.2f}% |\n"
    
    md += "\n"
    
    md += "## クラスごとのF1スコア\n\n"
    md += "| Direction | ベースライン | クラスごと閾値 | 改善率 |\n"
    md += "|-----------|------------|--------------|--------|\n"
    
    for d in dirs:
        baseline_f1 = baseline_metrics.get(d, 0.0)
        perclass_f1 = perclass_metrics.get(d, 0.0)
        improvement_d = calculate_improvement(baseline_f1, perclass_f1)
        
        md += f"| **{d}** | {baseline_f1:.4f} | {perclass_f1:.4f} | {improvement_d:+.2f}% |\n"
    
    md += "\n"
    
    md += "## サマリ\n\n"
    
    # Calculate average improvement
    improvements = []
    for d in dirs:
        baseline_f1 = baseline_metrics.get(d, 0.0)
        perclass_f1 = perclass_metrics.get(d, 0.0)
        if baseline_f1 > 0:
            improvements.append(calculate_improvement(baseline_f1, perclass_f1))
    
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0
    
    md += f"- **全体的な改善**: Macro F1が {baseline_macro:.4f} から {perclass_macro:.4f} に向上（{improvement:+.2f}%）\n"
    md += f"- **平均改善率**: {avg_improvement:+.2f}%\n"
    
    # Find best and worst improvements
    class_improvements = {d: calculate_improvement(baseline_metrics.get(d, 0.0), perclass_metrics.get(d, 0.0)) 
                          for d in dirs if baseline_metrics.get(d, 0.0) > 0}
    
    if class_improvements:
        best_class = max(class_improvements, key=class_improvements.get)
        worst_class = min(class_improvements, key=class_improvements.get)
        
        md += f"- **最も改善したクラス**: `{best_class}` ({class_improvements[best_class]:+.2f}%)\n"
        md += f"- **最も改善が少ないクラス**: `{worst_class}` ({class_improvements[worst_class]:+.2f}%)\n"
    
    md += "\n"
    
    md += "## 結論\n\n"
    if improvement > 0:
        md += f"クラスごとの閾値最適化により、Macro F1スコアが **{improvement:.2f}%** 向上しました。\n"
        md += "この改善は、各directional uncertaintyクラスに最適な予測閾値を設定することで達成されました。\n"
    else:
        md += "クラスごとの閾値最適化による顕著な改善は見られませんでした。\n"
        md += "他の手法（マルチレイヤー結合、データ拡張等）の検討が必要です。\n"
    
    return md


def main():
    parser = argparse.ArgumentParser(description="実験結果の比較レポート生成")
    parser.add_argument("--baseline", type=str, required=True, help="ベースライン結果のディレクトリ")
    parser.add_argument("--perclass", type=str, required=True, help="クラスごと閾値版のディレクトリ")
    parser.add_argument("--multilayer", type=str, help="マルチレイヤー版のディレクトリ（オプション）")
    parser.add_argument("--output", type=str, default="comparison_report.md", help="出力ファイル名")
    args = parser.parse_args()
    
    baseline_dir = Path(args.baseline)
    perclass_dir = Path(args.perclass)
    
    print(f"ベースライン: {baseline_dir}")
    print(f"クラスごと閾値: {perclass_dir}")
    
    # Load summaries
    baseline_summary = load_summary(baseline_dir)
    perclass_summary = load_summary(perclass_dir)
    
    print(f"ベースラインのメトリクス数: {len(baseline_summary)}")
    print(f"クラスごと閾値のメトリクス数: {len(perclass_summary)}")
    
    # Extract metrics
    baseline_metrics = extract_metrics(baseline_summary)
    perclass_metrics = extract_metrics(perclass_summary)
    
    print(f"\n抽出されたメトリクス:")
    print(f"  ベースライン: {list(baseline_metrics.keys())}")
    print(f"  クラスごと閾値: {list(perclass_metrics.keys())}")
    
    # Generate report
    multilayer_metrics = None
    if args.multilayer:
        multilayer_dir = Path(args.multilayer)
        multilayer_summary = load_summary(multilayer_dir)
        multilayer_metrics = extract_metrics(multilayer_summary)
    
    report = generate_comparison_markdown(baseline_metrics, perclass_metrics, multilayer_metrics)
    
    # Save report
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\n比較レポートを生成しました: {output_path}")
    print("\n" + "="*50)
    print("レポートのプレビュー:")
    print("="*50)
    print(report[:500] + "...")


if __name__ == "__main__":
    main()
