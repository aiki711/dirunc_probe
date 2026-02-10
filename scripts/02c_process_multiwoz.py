# scripts/02c_process_multiwoz.py
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from common import DIRS, QUERY_TOKENS_STR, map_slot_to_dir, PLACEHOLDER_BY_DIR, normalize_text, replace_values_in_text

def read_list_file(path: Path) -> Set[str]:
    """Read line-separated list of filenames."""
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return set(lines)

def process_multiwoz(
    data_path: Path,
    val_list_path: Path,
    test_list_path: Path,
    out_dir: Path,
    seed: int = 42,
    limit: int = 0
) -> None:
    
    print(f"Loading data from {data_path}...")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        
    val_files = read_list_file(val_list_path)
    test_files = read_list_file(test_list_path)
    
    # Identify train files
    train_files = set(data.keys()) - val_files - test_files
    
    rng = random.Random(seed)
    
    splits = {
        "train": train_files,
        "dev": val_files,
        "test": test_files
    }
    
    for split_name, filenames in splits.items():
        print(f"Processing split: {split_name} ({len(filenames)} dialogues)...")
        rows_out = []
        dir_counts = Counter()
        slot_counts = Counter()
        
        count = 0
        for fname in filenames:
            if limit and count >= limit:
                break
            
            if fname not in data:
                continue
                
            dialogue = data[fname]
            dialogue_id = fname.replace(".json", "")
            log = dialogue.get("log", [])
            
            # Iterate turns
            # MultiWOZ user turns are at even indices: 0, 2, 4...
            # System turns at odd: 1, 3, 5...
            # Metadata in system turn i (odd) reflects state after user turn i-1 (even).
            # So for user turn at idx (even), we look at metadata at idx+1.
            
            for i in range(0, len(log), 2):
                user_turn = log[i]
                if i + 1 >= len(log):
                    break # Last user turn without system response/state update?
                    
                system_turn = log[i+1]
                metadata = system_turn.get("metadata", {})
                
                user_text = normalize_text(user_turn.get("text", ""))
                if not user_text:
                    continue
                
                # Extract active slots present in text
                present_slots = []
                
                # Metadata structure: domain -> semi -> slot: value
                # (Ignore 'book' for now as it's often yes/no or abstract, stick to 'semi' aka informational slots)
                
                for domain, domain_data in metadata.items():
                    semi = domain_data.get("semi", {})
                    for slot, value in semi.items():
                        if not value or value in ["", "not mentioned", "dontcare", "none"]:
                            continue
                        
                        # Check if value is in user text
                        # Simple case-insensitive exact match
                        # Normalize value and text
                        val_norm = normalize_text(value)
                        if not val_norm:
                            continue
                            
                        # Use regex for word boundary? Or just substring?
                        # Value "cheap" matches "cheaply"? Maybe.
                        # "centre" matches "centre"? Yes.
                        # "6" matches "6"? Yes.
                        # Let's use simple substring for robustness with typos in user text
                        if val_norm.lower() in user_text.lower():
                            slot_name = f"{domain}-{slot}"
                            present_slots.append({
                                "slot": slot_name,
                                "value": val_norm,
                                "domain": domain,
                                "clean_slot": slot # just the slot name part for mapping
                            })
                
                if not present_slots:
                    continue
                    
                # For each present slot, create examples
                # 1. Resolved (Target slot present) -> Label 0
                # 2. Unresolved (Target slot removed) -> Label 1 for that DIR
                
                # Optimization: We can share the Resolved example across slots if we want, 
                # but "required_slots" concept in DirUnc usually means "what is needed".
                # Here, we treat "present slots" as "required". 
                # (If user said it, it's important).
                
                # Base labels: All 0 (Assume everything present is satisfied).
                labels_base = {d: 0 for d in DIRS}
                
                # Generate ONE resolved example per turn (containing all info)
                # Or one per slot?
                # Let's generate one per turn to be efficient, but we need to know what "Required" means for the prompt?
                # In DirUnc, the "Query" is implicit. The model predicts missing info.
                # If we provide full text, model should predict 0 for all.
                
                turn_uid = f"{dialogue_id}::t{i}"
                rows_out.append({
                    "id": f"{turn_uid}::resolved",
                    "text": user_text + QUERY_TOKENS_STR,
                    "labels": labels_base,
                    "split": split_name,
                    "meta": {"type": "resolved", "slots": [s["slot"] for s in present_slots]}
                })
                
                # Generate Unresolved examples
                for ps in present_slots:
                    slot_full = ps["slot"]
                    slot_clean = ps["clean_slot"]
                    val = ps["value"]
                    
                    # Map to DIR
                    d = map_slot_to_dir(slot_full, "") # No description available
                    
                    # Generate perturbed text
                    # Level 0: Delete
                    # Level 1: Placeholder
                    
                    for level in [0, 1]:
                        if level == 0:
                            text_mod = replace_values_in_text(user_text, [val], mode="delete")
                        else:
                            ph = PLACEHOLDER_BY_DIR.get(d, "something")
                            text_mod = replace_values_in_text(user_text, [val], mode="placeholder", placeholder=ph)
                        
                        if normalize_text(text_mod) == normalize_text(user_text):
                            continue
                        
                        # Labels: target val removed -> dir d is missing (1)
                        labels = {k: 0 for k in DIRS}
                        labels[d] = 1
                        
                        rows_out.append({
                            "id": f"{turn_uid}::{slot_full}::lvl{level}",
                            "text": text_mod + QUERY_TOKENS_STR,
                            "labels": labels,
                            "split": split_name,
                            "meta": {
                                "type": "unresolved",
                                "level": level,
                                "target_slot": slot_full,
                                "target_dir": d,
                                "removed_value": val
                            }
                        })
                        
                        if level == 0: # count stats once
                            dir_counts[d] += 1
                            slot_counts[slot_full] += 1
                
            count += 1
        
        # Save
        out_path = out_dir / f"{split_name}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                
        print(f"Saved {len(rows_out)} examples to {out_path}")
        print("DIR Stats:", dir_counts)
        print("Top Slots:", slot_counts.most_common(10))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw/multiwoz", help="Input directory containing data.json")
    parser.add_argument("--out_dir", type=str, default="data/processed/multiwoz/dirunc", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit dialogues per split")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    
    process_multiwoz(
        data_path=data_dir / "data.json",
        val_list_path=data_dir / "valListFile.json",
        test_list_path=data_dir / "testListFile.json",
        out_dir=out_dir,
        limit=args.limit
    )

if __name__ == "__main__":
    main()
