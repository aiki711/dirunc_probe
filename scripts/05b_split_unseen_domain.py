import json
import argparse
from pathlib import Path
import random

def get_domain(row):
    source = row["source"]
    meta = row.get("_meta", {})
    if source == "sgd":
        return meta.get("service", "")
    elif source == "multiwoz":
        return meta.get("domain", "")
    else:
        return "qasrl_domain"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/mixed/dirunc")
    parser.add_argument("--out_dir", type=str, default="data/processed/mixed/dirunc_unseen_split")
    parser.add_argument("--unseen_sgd", nargs="+", default=["Flights_1", "Flights_2", "Buses_1", "Buses_2", "Trains_1", "Travel_1"])
    parser.add_argument("--unseen_multiwoz", nargs="+", default=["train", "taxi"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unseen_domains = set(args.unseen_sgd + args.unseen_multiwoz)

    for split in ["train", "dev", "test"]:
        in_file = data_dir / f"{split}.jsonl"
        if not in_file.exists():
            continue

        seen_out = out_dir / f"seen_{split}.jsonl"
        unseen_out = out_dir / f"unseen_{split}.jsonl"

        seen_count = 0
        unseen_count = 0

        with open(in_file, "r") as f_in, \
             open(seen_out, "w") as f_seen, \
             open(unseen_out, "w") as f_unseen:
            
            for line in f_in:
                row = json.loads(line)
                domain = get_domain(row)

                if domain in unseen_domains:
                    f_unseen.write(line)
                    unseen_count += 1
                else:
                    f_seen.write(line)
                    seen_count += 1

        print(f"[{split}] Seen: {seen_count}, Unseen: {unseen_count}")

if __name__ == "__main__":
    main()
