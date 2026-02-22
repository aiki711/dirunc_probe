import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any
from common import DIRS, write_jsonl

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

def balance_dataset(rows: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """
    マルチラベルデータセットの各ラベル分布が均衡になるようにダウンサンプリングする。
    """
    if not rows:
        return []
    
    rng = random.Random(seed)
    
    # 各ラベルごとのインデックスを収集
    label_to_idxs = {d: [] for d in DIRS}
    for i, row in enumerate(rows):
        for d in DIRS:
            if row["labels"].get(d, 0) == 1:
                label_to_idxs[d].append(i)
    
    # 最小のラベル数を確認（ただし0のものは除外）
    counts = {d: len(idxs) for d, idxs in label_to_idxs.items()}
    non_zero_counts = [c for c in counts.values() if c > 0]
    
    # ターゲット数を決定
    min_count = min(non_zero_counts) if non_zero_counts else 0
    
    print(f"  [balance] Original counts: {counts}")
    print(f"  [balance] Target count per label (min of non-zero): {min_count}")
    
    selected_idxs = set()
    
    # 各ラベルについて、ターゲット数分をサンプリング
    # マルチレベルなので既に選ばれたインデックスは再利用しつつ、足りない分を補う
    for d in DIRS:
        idxs = label_to_idxs[d]
        already_selected = [idx for idx in idxs if idx in selected_idxs]
        
        if len(already_selected) >= min_count:
            continue
        
        needed = min_count - len(already_selected)
        candidates = [idx for idx in idxs if idx not in selected_idxs]
        rng.shuffle(candidates)
        selected_idxs.update(candidates[:needed])
        
    balanced_rows = [rows[i] for i in sorted(list(selected_idxs))]
    
    # 再確認
    new_counts = {d: 0 for d in DIRS}
    for r in balanced_rows:
        for d in DIRS:
            if r["labels"].get(d, 0) == 1:
                new_counts[d] += 1
    print(f"  [balance] Balanced counts: {new_counts}")
    
    return balanced_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/processed/mixed")
    parser.add_argument("--output_dir", type=str, default="data/processed/mixed/dirunc")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        input_file = in_dir / f"{split}.jsonl"
        if not input_file.exists():
            print(f"Skipping {split} (file not found: {input_file})")
            continue

        print(f"Processing {split}...")
        rows = list(read_jsonl(input_file))
        print(f"  Original records: {len(rows)}")
        
        balanced_rows = balance_dataset(rows, seed=args.seed)
        
        out_path = out_dir / f"{split}.jsonl"
        write_jsonl(out_path, balanced_rows)
        print(f"  Saved {len(balanced_rows)} records to {out_path}")

if __name__ == "__main__":
    main()
