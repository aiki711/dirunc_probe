# scripts/check_strip.py
from pathlib import Path
import json

QUERY_TOKENS = " [WHO?] [WHAT?] [WHEN?] [WHERE?] [WHY?] [HOW?]"

def read_jsonl(p: Path):
    rows=[]
    for line in p.open():
        line=line.strip()
        if line:
            rows.append(json.loads(line))
    return rows

def strip_query_tokens(text: str) -> str:
    t = text.rstrip()
    if t.endswith(QUERY_TOKENS):
        return t[: -len(QUERY_TOKENS)].rstrip()
    return text

rows = read_jsonl(Path("data/processed/sgd/dirunc/dev.jsonl"))
n = len(rows)
ends = sum(1 for r in rows if r["text"].rstrip().endswith(QUERY_TOKENS))
after_ends = sum(1 for r in rows if strip_query_tokens(r["text"]).rstrip().endswith(QUERY_TOKENS))

print("N:", n)
print("endswith QUERY_TOKENS (before):", ends, f"({ends/n:.3f})")
print("endswith QUERY_TOKENS (after strip):", after_ends, f"({after_ends/n:.3f})")

# 例を1つ表示
t0 = rows[0]["text"]
print("\n--- example before ---\n", t0[-120:])
print("\n--- example after ----\n", strip_query_tokens(t0)[-120:])
