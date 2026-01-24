# scripts/05_downsample_dataset.py
import argparse
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def downsample(rows, cap, seed=42):
    random.seed(seed)
    
    # Group by service
    by_service = defaultdict(list)
    for r in rows:
        svc = str(r.get("service", "unknown"))
        by_service[svc].append(r)
    
    # Select
    out = []
    print(f"Downsampling with cap={cap}...")
    for svc, items in sorted(by_service.items()):
        if len(items) > cap:
            print(f"  {svc}: {len(items)} -> {cap}")
            selected = random.sample(items, cap)
        else:
            print(f"  {svc}: {len(items)} (kept all)")
            selected = items
        out.extend(selected)
    
    # Shuffle final result
    random.shuffle(out)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process train
    print("Processing train.jsonl...")
    train_rows = list(read_jsonl(in_dir / "train.jsonl"))
    print(f"Original train size: {len(train_rows)}")
    train_down = downsample(train_rows, args.cap, args.seed)
    print(f"New train size: {len(train_down)}")
    
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_down:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Process dev
    print("Processing dev.jsonl...")
    dev_rows = list(read_jsonl(in_dir / "dev.jsonl"))
    print(f"Original dev size: {len(dev_rows)}")
    dev_down = downsample(dev_rows, args.cap, args.seed) 
    print(f"New dev size: {len(dev_down)}")

    with (out_dir / "dev.jsonl").open("w", encoding="utf-8") as f:
        for r in dev_down:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Done. Saved to {out_dir}")

if __name__ == "__main__":
    main()
