import json
from pathlib import Path

def find_failures():
    corpus_path = Path("runs/balanced/experiment7_neurons/corpus_shift_data.json")
    all_jsonl = Path("data/processed/mixed/dirunc/all.jsonl")
    output_path = Path("runs/balanced/experiment9_visuals/failure_candidates.json")

    with open(corpus_path, "r") as f:
        data = json.load(f)

    # 1. Find the worst failures (Smallest shift, but high initial probability)
    candidates = []
    for label, content in data.items():
        pairs = content.get("pairs", [])
        for p in pairs:
            # Type: Still high probability after filling
            if p["A_prob"] > 0.8 and p["B_prob"] > 0.8:
                candidates.append({
                    "label": label,
                    "did": p["dialogue_id"],
                    "turn_a": p["turn_a"],
                    "turn_b": p["turn_b"],
                    "A_prob": p["A_prob"],
                    "B_prob": p["B_prob"],
                    "delta": p["B_prob"] - p["A_prob"]
                })

    # Sort by how little they changed (worst failure)
    candidates.sort(key=lambda x: abs(x["delta"]))
    
    # 2. Extract text from corpus_pairs_{label}.jsonl for these IDs
    pairs_dir = Path("runs/balanced/experiment7_neurons/corpus_pairs")
    
    # Map (did, turn_a, turn_b) -> (text_a, text_b)
    final_targets = []
    
    # Group candidates by label to minimize file openings
    from collections import defaultdict
    candidates_by_label = defaultdict(list)
    for c in candidates:
        candidates_by_label[c["label"]].append(c)

    for label, label_candidates in candidates_by_label.items():
        pairs_file = pairs_dir / f"corpus_pairs_{label}.jsonl"
        if not pairs_file.exists(): continue
        
        # Build lookup for this label
        lookup = {}
        with open(pairs_file, "r") as f:
            for line in f:
                item = json.loads(line)
                key = (item["dialogue_id"], item["turn_a"], item["turn_b"])
                lookup[key] = (item["text_a"], item["text_b"])
        
        for c in label_candidates[:5]: # Take top 5 per label
            key = (c["did"], c["turn_a"], c["turn_b"])
            if key in lookup:
                text_a, text_b = lookup[key]
                final_targets.append({
                    "id": f"fail_{label}_{c['did']}",
                    "label": label,
                    "text_A": text_a,
                    "text_B": text_b,
                    "A_prob": c["A_prob"],
                    "B_prob": c["B_prob"]
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_targets, f, indent=2)

    print(f"Found {len(final_targets)} failure candidates.")
    for t in final_targets[:3]:
        print(f"--- {t['id']} ---")
        print(f"Text A: {t['text_A']}")
        print(f"Text B: {t['text_B']}")
        print(f"Probs: A={t['A_prob']:.2f} -> B={t['B_prob']:.2f}")

if __name__ == "__main__":
    find_failures()
