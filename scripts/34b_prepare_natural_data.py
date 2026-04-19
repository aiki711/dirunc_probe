#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import defaultdict

def main():
    root = Path("data/processed/case_grammar")
    source_filled = root / "cg_train.jsonl"
    source_natural_missing = root / "cg_train_natural.jsonl"
    
    out_train = root / "natural_train.jsonl"
    out_dev = root / "natural_dev.jsonl"

    print("Loading original 'filled' rows...")
    filled_rows = {}
    with source_filled.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["condition"] == "filled":
                pid = row["id"].rsplit("::", 1)[0]
                filled_rows[pid] = row

    print(f"Loaded {len(filled_rows)} filled rows.")

    print("Merging with LLM-generated natural missing rows...")
    paired_data = []
    with source_natural_missing.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pid = row["id"].rsplit("::", 1)[0]
            
            if pid in filled_rows:
                # Update text with LLM-generated natural missing version
                if "llm_missing" in row and row["llm_missing"]:
                    # Keep original context if it exists in 'text', but replace the utterance
                    # Format is usually [Context]\nUser: filled_text
                    if "\nUser: " in row["text"]:
                        context_prefix = row["text"].rsplit("\nUser: ", 1)[0] + "\nUser: "
                        row["text"] = context_prefix + row["llm_missing"]
                    else:
                        row["text"] = row["llm_missing"]
                    
                    paired_data.append((filled_rows[pid], row))

    print(f"Total pairs created: {len(paired_data)}")

    # Shuffle and Split
    random.seed(42)
    random.shuffle(paired_data)
    
    split_idx = int(0.8 * len(paired_data))
    train_pairs = paired_data[:split_idx]
    dev_pairs = paired_data[split_idx:]

    def save_jsonl(pairs, path):
        with path.open("w", encoding="utf-8") as f:
            for filled, missing in pairs:
                f.write(json.dumps(filled, ensure_ascii=False) + "\n")
                f.write(json.dumps(missing, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs)} pairs to {path}")

    save_jsonl(train_pairs, out_train)
    save_jsonl(dev_pairs, out_dev)

    print("\nPreparation complete.")

if __name__ == "__main__":
    main()
