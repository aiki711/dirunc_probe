import json
from collections import Counter
from pathlib import Path

def analyze_what_patterns(path: Path):
    what_next_words = Counter()
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta = row.get("meta", {})
            question = meta.get("question", "").lower()
            
            if question.startswith("what"):
                parts = question.split()
                if len(parts) > 1:
                    next_word = parts[1]
                    what_next_words[next_word] += 1

    print("Top words after 'what':", what_next_words.most_common(20))

if __name__ == "__main__":
    analyze_what_patterns(Path("data/processed/qasrl/dirunc/train.jsonl"))
