# scripts/02d_merge_datasets.py
# 3ソース（SGD / QA-SRL / MultiWOZ）の Natural Missing データを統合するスクリプト。
#
# 処理内容:
#   1. 各ソースの {train,dev,test}.jsonl を読み込み
#   2. QA-SRL が過大な場合は --max_qasrl でダウンサンプリング
#   3. ラベルの偏りを抑えるため、最小数ラベルに合わせてダウンサンプリング (Measure 1)
#   4. シャッフルして出力
#
# 使用例:
#   python scripts/02d_merge_datasets.py \
#     --sgd_dir      data/processed/sgd/natural \
#     --qasrl_dir    data/processed/qasrl/natural \
#     --multiwoz_dir data/processed/multiwoz/natural \
#     --out_dir      data/processed/mixed \
#     --max_qasrl    10000 \
#     --balance      # ラベルバランスを有効化
#     --seed         42

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from common import DIRS, write_jsonl

# ---------- IO ユーティリティ ----------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"  [warn] not found: {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# ---------- 統計表示 ----------

def print_stats(title: str, rows: List[Dict[str, Any]]) -> None:
    dir_counts: Counter = Counter()
    source_counts: Counter = Counter()
    for r in rows:
        source_counts[r.get("source", "unknown")] += 1
        for d, v in r.get("labels", {}).items():
            if v:
                dir_counts[d] += 1

    total = len(rows)
    print(f"\n--- {title} ---")
    print(f"  Total records: {total}")
    print("  Source distribution:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:>12}: {cnt:>8}  ({cnt / max(total, 1) * 100:5.1f}%)")
    print("  DIR distribution (positive labels):")
    for d in DIRS:
        c = dir_counts.get(d, 0)
        print(f"    {d:>6}: {c:>8}  ({c / max(total, 1) * 100:5.1f}%)")


# ---------- バランス調整 (Measure 1) ----------

def balance_dataset(rows: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """各ラベルの分布が均衡（最小数に合わせる）になるようにダウンサンプリング。"""
    if not rows:
        return []
    
    rng = random.Random(seed)
    
    # 各ラベルごとのインデックスを収集
    label_to_idxs = {d: [] for d in DIRS}
    for i, row in enumerate(rows):
        for d in DIRS:
            if row["labels"].get(d, 0) == 1:
                label_to_idxs[d].append(i)
    
    # 最小のラベル数を確認（0のものは除外）
    counts = {d: len(idxs) for d, idxs in label_to_idxs.items()}
    non_zero_counts = [c for c in counts.values() if c > 0]
    
    if not non_zero_counts:
        return rows # バランス不能

    min_count = min(non_zero_counts)
    print(f"  [balance] Target count per label: {min_count}")
    
    selected_idxs = set()
    for d in DIRS:
        idxs = label_to_idxs[d]
        if not idxs: continue
        
        already_selected = [idx for idx in idxs if idx in selected_idxs]
        if len(already_selected) >= min_count:
            continue
        
        needed = min_count - len(already_selected)
        candidates = [idx for idx in idxs if idx not in selected_idxs]
        rng.shuffle(candidates)
        selected_idxs.update(candidates[:needed])
        
    return [rows[i] for i in sorted(list(selected_idxs))]


# ---------- メイン ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="3ソースのデータを統合し、ラベルバランスを調整する。")
    parser.add_argument("--sgd_dir", type=str, default="data/processed/sgd/natural")
    parser.add_argument("--qasrl_dir", type=str, default="data/processed/qasrl/natural")
    parser.add_argument("--multiwoz_dir", type=str, default="data/processed/multiwoz/natural")
    parser.add_argument("--out_dir", type=str, default="data/processed/mixed")
    parser.add_argument("--max_qasrl", type=int, default=10000, help="QA-SRLの上限レコード数")
    parser.add_argument("--balance", action="store_true", help="ラベルバランスを有効化するか")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sgd_dir = Path(args.sgd_dir)
    qasrl_dir = Path(args.qasrl_dir)
    multiwoz_dir = Path(args.multiwoz_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    for split in ["train", "dev", "test"]:
        print(f"\n========== {split} ==========")
        sgd_rows = read_jsonl(sgd_dir / f"{split}.jsonl")
        qasrl_rows = read_jsonl(qasrl_dir / f"{split}.jsonl")
        multiwoz_rows = read_jsonl(multiwoz_dir / f"{split}.jsonl")

        print(f"  Raw: SGD={len(sgd_rows)}, QA-SRL={len(qasrl_rows)}, MultiWOZ={len(multiwoz_rows)}")

        if split == "train" and args.max_qasrl > 0 and len(qasrl_rows) > args.max_qasrl:
            random.Random(args.seed).shuffle(qasrl_rows)
            qasrl_rows = qasrl_rows[:args.max_qasrl]
            print(f"  QA-SRL (after max_qasrl): {len(qasrl_rows)}")

        all_rows = sgd_rows + qasrl_rows + multiwoz_rows
        if not all_rows: continue

        if args.balance:
            all_rows = balance_dataset(all_rows, seed=args.seed)
            print(f"  After balancing: {len(all_rows)}")

        rng.shuffle(all_rows)
        out_path = out_dir / f"{split}.jsonl"
        write_jsonl(out_path, all_rows)
        print_stats(f"mixed/{split}.jsonl", all_rows)
        print(f"  -> Saved to {out_path}")

if __name__ == "__main__":
    main()
