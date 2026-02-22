# scripts/02d_merge_datasets.py
# 3ソース（SGD / QA-SRL / MultiWOZ）の Natural Missing データを統合するスクリプト。
#
# 処理内容:
#   1. 各ソースの {train,dev}.jsonl を読み込み
#   2. QA-SRL が過大な場合は --max_qasrl でダウンサンプリング
#   3. シャッフルして出力
#      - train -> data/processed/mixed/train.jsonl
#      - dev   -> data/processed/mixed/dev.jsonl
#
# 使用例:
#   python scripts/02d_merge_datasets.py \
#     --sgd_dir      data/processed/sgd/natural \
#     --qasrl_dir    data/processed/qasrl/natural \
#     --multiwoz_dir data/processed/multiwoz/natural \
#     --out_dir      data/processed/mixed \
#     --max_qasrl    10000 \
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
    """JSONL ファイルを読み込んでリストで返す。ファイルが存在しない場合は空リスト。"""
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


# ---------- メイン ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="3ソースの Natural Missing データを統合する。")
    parser.add_argument(
        "--sgd_dir",
        type=str,
        default="data/processed/sgd/natural",
        help="SGD Natural データディレクトリ",
    )
    parser.add_argument(
        "--qasrl_dir",
        type=str,
        default="data/processed/qasrl/natural",
        help="QA-SRL Natural データディレクトリ",
    )
    parser.add_argument(
        "--multiwoz_dir",
        type=str,
        default="data/processed/multiwoz/natural",
        help="MultiWOZ Natural データディレクトリ",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/mixed",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--max_qasrl",
        type=int,
        default=0,
        help="train における QA-SRL の上限レコード数（0=無制限）",
    )
    parser.add_argument("--seed", type=int, default=42, help="シャッフル用乱数シード")
    args = parser.parse_args()

    sgd_dir = Path(args.sgd_dir)
    qasrl_dir = Path(args.qasrl_dir)
    multiwoz_dir = Path(args.multiwoz_dir)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    for split in ["train", "dev", "test"]:
        print(f"\n========== {split} ==========")

        # --- 各ソースの読み込み ---
        sgd_rows = read_jsonl(sgd_dir / f"{split}.jsonl")
        qasrl_rows = read_jsonl(qasrl_dir / f"{split}.jsonl")
        multiwoz_rows = read_jsonl(multiwoz_dir / f"{split}.jsonl")

        print(f"  SGD:      {len(sgd_rows):>8} records")
        print(f"  QA-SRL:   {len(qasrl_rows):>8} records")
        print(f"  MultiWOZ: {len(multiwoz_rows):>8} records")

        # --- QA-SRL のダウンサンプリング（train のみ） ---
        if split == "train" and args.max_qasrl > 0 and len(qasrl_rows) > args.max_qasrl:
            rng_sample = random.Random(args.seed)
            qasrl_rows = rng_sample.sample(qasrl_rows, args.max_qasrl)
            print(f"  QA-SRL (after downsample): {len(qasrl_rows):>8} records")

        # --- 結合 ---
        all_rows = sgd_rows + qasrl_rows + multiwoz_rows

        if not all_rows:
            print(f"  [warn] No records for split={split}, skipping.")
            continue

        # --- シャッフル ---
        rng.shuffle(all_rows)

        # --- 出力 ---
        out_path = out_dir / f"{split}.jsonl"
        write_jsonl(out_path, all_rows)

        print_stats(f"mixed/{split}.jsonl", all_rows)
        print(f"\n  -> Saved {len(all_rows)} records to {out_path}")


if __name__ == "__main__":
    main()
