import json
from collections import Counter
from pathlib import Path

def analyze_jsonl(path: Path):
    print(f"Analyzing {path}...")
    counts = Counter()
    total = 0
    resolved_count = 0
    unresolved_count = 0
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            row = json.loads(line)
            labels = row.get("labels", {})
            meta = row.get("meta", {})
            type_ = meta.get("type")
            
            if type_ == "resolved":
                resolved_count += 1
            else:
                unresolved_count += 1
                # Count positive labels
                for k, v in labels.items():
                    if v == 1:
                        counts[k] += 1

    print(f"Total Rows: {total}")
    print(f"Resolved (Negative): {resolved_count}")
    print(f"Unresolved (Positive): {unresolved_count}")
    print("Positive Counts by Class:", counts)

if __name__ == "__main__":
    analyze_jsonl(Path("data/processed/mixed/dirunc/train.jsonl"))
