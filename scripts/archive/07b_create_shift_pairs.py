import json
from pathlib import Path
import sys
import os

# Add the project root to sys.path to allow importing scripts.common
sys.path.append(os.getcwd())
from scripts.common import QUERY_TOKENS_STR

def main():
    # Define minimal pairs for each label
    # Each pair is (A: missing, B: filled)
    pairs = [
        {
            "label": "when",
            "A": "I want to find a restaurant.",
            "B": "I want to find a restaurant for tonight."
        },
        {
            "label": "where",
            "A": "I need to book a taxi.",
            "B": "I need to book a taxi to the airport."
        },
        {
            "label": "who",
            "A": "I want to reserve a table.",
            "B": "I want to reserve a table for two people."
        },
        {
            "label": "what",
            "A": "I'm looking for a place to stay.",
            "B": "I'm looking for a hotel with free wifi."
        },
        {
            "label": "how",
            "A": "Could you book me a ride?",
            "B": "Could you book me a shared ride?"
        },
        {
            "label": "which",
            "A": "I am looking for a restaurant in London.",
            "B": "I am looking for a cheap restaurant in London."
        }
    ]

    # Process to add query tokens
    processed_pairs = []
    for p in pairs:
        processed_pairs.append({
            "label": p["label"],
            "A_raw": p["A"],
            "B_raw": p["B"],
            "A": p["A"] + QUERY_TOKENS_STR,
            "B": p["B"] + QUERY_TOKENS_STR
        })

    out_path = Path("runs/balanced/experiment7_neurons/shift_pairs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Shift pairs saved to {out_path}")
    for p in processed_pairs:
        print(f"  {p['label']:<7}: A: {p['A_raw']} -> B: {p['B_raw']}")

if __name__ == "__main__":
    main()
