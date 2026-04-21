import json
import re
from pathlib import Path

def detect_artifacts(path):
    print(f"Scanning {path} for syntactic artifacts...")
    bad_count = 0
    total_count = 0
    
    # Prepositions to look for at the end of clauses or before other prepositions
    preps = [r"at", r"on", r"in", r"to", r"from", r"by", r"of", r"with", r"for"]
    
    # Pattern 1: Preposition followed by punctuation (dangling at the end)
    # Pattern 2: Consecutive prepositions (e.g., "from to")
    patterns = [
        re.compile(r"\b(" + "|".join(preps) + r")\s*[.,!?]", re.IGNORECASE),
        re.compile(r"\b(" + "|".join(preps) + r")\s+(?:at|on|in|to|from|by|of|with|for)\b", re.IGNORECASE)
    ]

    with open(path, "r") as f:
        for line in f:
            total_count += 1
            row = json.loads(line)
            text = row.get("llm_missing", "")
            
            is_bad = False
            for p in patterns:
                if p.search(text):
                    is_bad = True
                    break
            
            if is_bad:
                bad_count += 1
                if bad_count <= 10:
                    print(f"  [Artifact found] ID: {row['id']}")
                    print(f"    Text: {text}")

    print(f"\nSummary:")
    print(f"  Total samples: {total_count}")
    print(f"  Problematic samples: {bad_count} ({bad_count/total_count*100:.1f}%)")

if __name__ == "__main__":
    detect_artifacts("data/processed/case_grammar/cg_train_natural_fixed.jsonl")
