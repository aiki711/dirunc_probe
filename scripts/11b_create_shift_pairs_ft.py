import json
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())
# Final Token版ではプロンプトにクエリトークンを付与しない
# from scripts.common import QUERY_TOKENS_STR

def main():
    # Define minimal pairs for each label (same as 07b, but without QUERY_TOKENS_STR)
    pairs = [
        {"label": "when",  "A": "I want to find a restaurant.",          "B": "I want to find a restaurant for tonight."},
        {"label": "where", "A": "I need to book a taxi.",                "B": "I need to book a taxi to the airport."},
        {"label": "who",   "A": "I want to reserve a table.",            "B": "I want to reserve a table for two people."},
        {"label": "what",  "A": "I'm looking for a place to stay.",      "B": "I'm looking for a hotel with free wifi."},
        {"label": "how",   "A": "Could you book me a ride?",             "B": "Could you book me a shared ride?"},
        {"label": "which", "A": "I am looking for a restaurant in London.", "B": "I am looking for a cheap restaurant in London."},
    ]

    # Final Token: NO query tokens appended ("A" and "B" are the raw texts)
    processed_pairs = []
    for p in pairs:
        processed_pairs.append({
            "label":  p["label"],
            "A_raw":  p["A"],
            "B_raw":  p["B"],
            "A":      p["A"],   # No QUERY_TOKENS_STR appended
            "B":      p["B"],   # No QUERY_TOKENS_STR appended
        })

    out_path = Path("runs/balanced/experiment7_neurons_ft/shift_pairs_ft.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(processed_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Shift pairs (Final Token, no query tokens) saved to {out_path}")
    for p in processed_pairs:
        print(f"  {p['label']:<7}: A: {p['A_raw']} -> B: {p['B_raw']}")

if __name__ == "__main__":
    main()
