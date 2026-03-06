import json
from pathlib import Path
from collections import defaultdict
import argparse

DIRS = ["who", "what", "when", "where", "why", "how", "which"]

def calculate_label_diff(labels_a, labels_b, target_label):
    """Calculates how many OTHER labels changed between A and B."""
    diff = 0
    for d in DIRS:
        if d != target_label:
            if labels_a.get(d, 0) != labels_b.get(d, 0):
                diff += 1
    return diff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_jsonl", type=str, default="data/processed/mixed/dirunc/all.jsonl")
    parser.add_argument("--out_dir", type=str, default="runs/balanced/experiment7_neurons/corpus_pairs")
    parser.add_argument("--min_pairs", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by dialogue ID
    dialogues = defaultdict(list)
    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # We need turn_idx and dialogue_id to sort and pair correctly
            meta = data.get("_meta", {})
            did = meta.get("dialogue_id")
            tidx = meta.get("turn_idx")
            
            # Handling QASRL or cases without turn_idx (fallback to list order if needed, but we prefer SGD/MW)
            if did is not None and tidx is not None:
                dialogues[did].append((tidx, data))

    # Sort each dialogue by turn idx
    for did in dialogues:
        dialogues[did].sort(key=lambda x: x[0])

    # Extract pairs for each label
    extracted_counts = {}
    
    for target_label in DIRS:
        pairs = []
        for did, turns in dialogues.items():
            # Look for 1 -> 0 transition for target_label where turn_A < turn_B
            for i in range(len(turns)):
                tidx_a, data_a = turns[i]
                labels_a = data_a.get("labels", {})
                
                if labels_a.get(target_label, 0) == 1: # Missing in A
                    # Find a subsequent turn where it's filled
                    for j in range(i+1, len(turns)):
                        tidx_b, data_b = turns[j]
                        labels_b = data_b.get("labels", {})
                        
                        if labels_b.get(target_label, 0) == 0: # Filled in B
                            # Calculate confounding factor (changes in other labels)
                            confounding_diff = calculate_label_diff(labels_a, labels_b, target_label)
                            
                            pairs.append({
                                "dialogue_id": did,
                                "turn_a": tidx_a,
                                "turn_b": tidx_b,
                                "text_a": data_a["text"],
                                "text_b": data_b["text"],
                                "labels_a": labels_a,
                                "labels_b": labels_b,
                                "confounding_diff": confounding_diff
                            })
                            break # Found the first resolution, move to next possible A

        # Sort pairs by confounding_diff (ascending) to prioritize "clean" pairs
        pairs.sort(key=lambda x: x["confounding_diff"])
        
        # We take top N or all if less than N
        # Let's take up to 200 pairs to keep tracking fast but statistically significant
        max_pairs = 200
        selected_pairs = pairs[:max_pairs]
        extracted_counts[target_label] = len(selected_pairs)
        
        if len(selected_pairs) > 0:
            out_file = out_dir / f"corpus_pairs_{target_label}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for p in selected_pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Extraction complete. Pairs saved to {out_dir}")
    print("Pairs per label:")
    for d, count in extracted_counts.items():
        print(f"  {d:>6}: {count} pairs extracted")

if __name__ == "__main__":
    main()
