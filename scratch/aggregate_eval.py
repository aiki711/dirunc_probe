import json
from pathlib import Path
import re

def parse_score(text, key):
    # Extracts scores like "Naturalness: 4" from the eval text
    match = re.search(f"{key}:\\s*(\\d)", text)
    if match:
        return int(match.group(1))
    return None

def main():
    path = Path("pilot_results_v5.json")
    if not path.exists():
        print("File not found.")
        return
    
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics = {
        "mech": {"Naturalness": [], "Omission": [], "MinimalChange": []},
        "natural": {"Naturalness": [], "Omission": [], "MinimalChange": []},
        "filled": {"Naturalness": [], "Omission": [], "MinimalChange": []},
    }
    
    keys = ["Naturalness", "Omission", "MinimalChange"]
    
    for d in data:
        for m_type, eval_key in [("mech", "eval_mech_v5"), ("natural", "eval_natural_v5"), ("filled", "eval_filled_v5")]:
            if eval_key in d:
                for k in keys:
                    score = parse_score(d[eval_key], k)
                    if score is not None:
                        metrics[m_type][k].append(score)
    
    print("| Method | Naturalness | Omission | MinimalChange | N |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    for m_type in ["filled", "mech", "natural"]:
        row = [m_type.capitalize()]
        n = 0
        for k in keys:
            scores = metrics[m_type][k]
            if scores:
                avg = sum(scores) / len(scores)
                row.append(f"{avg:.2f}")
                n = len(scores)
            else:
                row.append("-")
        row.append(str(n))
        print("| " + " | ".join(row) + " |")

if __name__ == "__main__":
    main()
