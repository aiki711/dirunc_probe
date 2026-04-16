import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
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
    input_path = Path("data/processed/sgd/dirunc_balanced/train.jsonl")
    output_path = Path("data/processed/sgd/dirunc_balanced/paired_train.jsonl")
    
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    rows = read_jsonl(input_path)
    print(f"Loaded {len(rows)} samples.")

    # Group by dialogue_id
    dialogues = defaultdict(list)
    for r in rows:
        dialogues[r["dialogue_id"]].append(r)

    paired_data = []
    dirs = ["who", "when", "where", "how", "which", "what"]

    for d_id, turns in tqdm(dialogues.items(), desc="Extracting real transitions"):
        # Sort turns by index to ensure temporal order
        turns.sort(key=lambda x: x["turn_idx"])
        
        for d in dirs:
            # Find all turns where this slot is Missing (1)
            missing_turns = [t for t in turns if t["labels"].get(d, 0) == 1]
            # Find all turns where this slot is Filled (0)
            filled_turns = [t for t in turns if t["labels"].get(d, 0) == 0]
            
            if not missing_turns or not filled_turns:
                continue
                
            # For each transition from Missing to Filled
            # To keep it high quality, we look for the *closest* transition
            for m_t in missing_turns:
                # Find the first filled turn that comes AFTER or at this missing turn 
                # (Same level might be synthetic pair from scripts/02, which is still good)
                f_t_candidates = [f for f in filled_turns if f["turn_idx"] >= m_t["turn_idx"]]
                if not f_t_candidates:
                    continue
                
                # Best B is the one with smallest turn_idx difference
                # but we also want difference in text to avoid identity mapping
                best_f_t = None
                for f_t in f_t_candidates:
                    if f_t["text"] != m_t["text"]:
                        best_f_t = f_t
                        break
                
                if best_f_t:
                    paired_data.append({
                        "pair_id": f"{d_id}::trans::{d}::t{m_t['turn_idx']}_to_t{best_f_t['turn_idx']}",
                        "text_A": m_t["text"],
                        "text_B": best_f_t["text"],
                        "label_idx": dirs.index(d),
                        "direction": d
                    })
                    # Use each missing turn only once per direction to avoid over-representation
                    break

    print(f"Generated {len(paired_data)} real dialogue transition pairs.")
    write_jsonl(output_path, paired_data)

if __name__ == "__main__":
    main()
