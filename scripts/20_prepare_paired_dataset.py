import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    # Existing data source
    input_path = Path("data/processed/sgd/dirunc_balanced/train.jsonl")
    output_path = Path("data/processed/sgd/dirunc_balanced/paired_train.jsonl")
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} samples.")

    # Group by (dialogue_id, turn_idx, target_slot) if metadata exists
    # If not, we have to rely on 'id' patterns or re-generate.
    # Looking at scripts/02, sample IDs are like "sgd::dialogue_id::turn_idx::level::..."
    # For contrastive, we need same (dialogue_id, turn_idx) but different State A vs B.
    
    # Let's use a simpler heuristic for now: 
    # In 'contrastive' generation mode (scripts/02), the ID might contain the slot info.
    # Actually, we can just generate them FRESH to be sure they are perfect pairs.
    
    # [Refinement] Let's create a script that uses common.py utilities to 
    # "Fill" or "Empty" slots in the same sentence to create a synthetic paired dataset.
    import sys
    import os
    sys.path.append(os.getcwd())
    from scripts.common import DIRS, WHO_KWS, WHEN_KWS, WHERE_KWS, HOW_KWS, WHICH_KWS
    KW_MAP = {"who": WHO_KWS, "when": WHEN_KWS, "where": WHERE_KWS, "how": HOW_KWS, "which": WHICH_KWS}

    paired_data = []
    
    # For each direction, we find sentences that have these keywords, 
    # and create a version where keywords are present (B) and removed (A).
    # This ensures a 1:1 correspondence.
    
    for row in tqdm(rows, desc="Pairing samples"):
        text = row["text"]
        labels = row["labels"]
        
        for d, kws in KW_MAP.items():
            if labels.get(d, 0) == 0: # It's currently 'Filled' in original data
                # Find which keyword is present
                found_kw = None
                for kw in sorted(kws, key=len, reverse=True):
                    if kw in text.lower():
                        found_kw = kw
                        break
                
                if found_kw:
                    # Create Missing version (A)
                    text_A = text.replace(found_kw, "[SLOT]") # Use a generic placeholder
                    labels_A = labels.copy()
                    labels_A[d] = 1 # Mark as Missing
                    
                    # Store as a pair
                    paired_data.append({
                        "pair_id": row.get("id", "unk") + f"_{d}",
                        "text_A": text_A,
                        "text_B": text,
                        "label_idx": list(DIRS).index(d),
                        "direction": d
                    })
    
    print(f"Generated {len(paired_data)} high-quality pairs.")
    write_jsonl(output_path, paired_data)

if __name__ == "__main__":
    main()
