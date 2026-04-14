#!/usr/bin/env python3
# scripts/34_split_cg_pairs.py
"""
Merges individual dataset JSONL files and splits them into train/dev
while ensuring that 'filled' and 'missing' pairs remain in the same split.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    root = Path("data/processed/case_grammar")
    files = [
        root / "cg_v1_sgd.jsonl",
        root / "cg_v1_multiwoz.jsonl",
        root / "cg_v1_qasrl.jsonl",
    ]
    
    pairs = defaultdict(dict)
    
    print("Loading datasets...")
    for fp in files:
        if not fp.exists():
            print(f"  Warning: {fp} not found. Skipping.")
            continue
            
        with fp.open("r", encoding="utf-8") as f:
            count = 0
            for line in f:
                data = json.loads(line)
                # ID format: dataset::file::turn::slot::condition
                # We split by the LAST '::' to get the pair ID
                full_id = data["id"]
                if "::" not in full_id:
                    continue
                    
                pid, cond = full_id.rsplit("::", 1)
                pairs[pid][cond] = data
                count += 1
            print(f"  Loaded {count} rows from {fp.name}")

    # Filter only complete pairs
    valid_pids = [pid for pid, content in pairs.items() if "filled" in content and "missing" in content]
    print(f"Total complete pairs found: {len(valid_pids)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(valid_pids)
    
    # Split 80/20
    split_idx = int(0.8 * len(valid_pids))
    train_pids = valid_pids[:split_idx]
    dev_pids = valid_pids[split_idx:]
    
    def save_split(pids, out_path):
        with out_path.open("w", encoding="utf-8") as f:
            for pid in pids:
                f.write(json.dumps(pairs[pid]["filled"], ensure_ascii=False) + "\n")
                f.write(json.dumps(pairs[pid]["missing"], ensure_ascii=False) + "\n")
        print(f"Saved {len(pids)} pairs ({len(pids)*2} rows) to {out_path}")

    save_split(train_pids, root / "cg_train.jsonl")
    save_split(dev_pids, root / "cg_dev.jsonl")

    # Double check dataset distribution
    print("\nDataset distribution in DEV set:")
    dist = defaultdict(int)
    for pid in dev_pids:
        ds_name = pid.split("::")[0]
        dist[ds_name] += 1
    for ds, count in dist.items():
        print(f"  {ds:10s}: {count} pairs")

if __name__ == "__main__":
    main()
