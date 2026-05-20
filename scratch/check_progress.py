import json
from pathlib import Path

BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")
DATA_DIR = BASE_DIR / "data/processed/case_grammar"

dev_src = DATA_DIR / "natural_dev.jsonl"
train_src = DATA_DIR / "natural_train.jsonl"
dev_out = DATA_DIR / "natural_dev_gemini.jsonl"
train_out = DATA_DIR / "natural_train_gemini.jsonl"

def get_target_ids(path):
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("dataset") == "qasrl":
                continue
            if row["id"].endswith("::filled"):
                ids.add(row["id"])
    return ids

print("Loading target IDs...")
dev_targets = get_target_ids(dev_src)
train_targets = get_target_ids(train_src)
all_targets = dev_targets.union(train_targets)

def get_finished_ids(paths):
    finished = set()
    for p in paths:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        finished.add(json.loads(line)["id"])
                    except Exception as e:
                        pass
    return finished

print("Loading finished IDs...")
finished = get_finished_ids([dev_out, train_out])

dev_finished = dev_targets.intersection(finished)
train_finished = train_targets.intersection(finished)

dev_total = len(dev_targets)
train_total = len(train_targets)
dev_done = len(dev_finished)
train_done = len(train_finished)

print(f"Dev: {dev_done} / {dev_total} ({dev_done/dev_total*100:.2f}%) finished")
print(f"Train: {train_done} / {train_total} ({train_done/train_total*100:.2f}%) finished")
print(f"Total: {dev_done + train_done} / {dev_total + train_total} ({(dev_done+train_done)/(dev_total+train_total)*100:.2f}%) finished")

remaining = all_targets - finished
print(f"Remaining in target set: {len(remaining)}")
