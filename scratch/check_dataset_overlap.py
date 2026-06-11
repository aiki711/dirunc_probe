#!/usr/bin/env python3
import json
from pathlib import Path

def check_and_align():
    root = Path("data/processed/case_grammar")
    soft_path = root / "paired_dev_gemini_soft.jsonl"
    strong_path = root / "paired_dev_gemini_strong.jsonl"
    mech_path = root / "paired_dev_gemini_mechanical.jsonl"
    
    def get_pids(path):
        pids = set()
        with path.open("r") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                pids.add(row["id"].rsplit("::", 1)[0])
        return pids

    pids_soft = get_pids(soft_path)
    pids_strong = get_pids(strong_path)
    pids_mech = get_pids(mech_path)

    print("Soft dev pairs:", len(pids_soft))
    print("Strong dev pairs:", len(pids_strong))
    print("Mech dev pairs before alignment:", len(pids_mech))

    # Keep only mechanical pairs that are also in soft dev set (or strong dev set)
    # They should have the same overlap. Let us filter mechanical by soft dev set keys.
    common_pids = pids_soft.intersection(pids_strong)
    print("Common pids between Soft & Strong:", len(common_pids))

    # Let us rewrite paired_dev_gemini_mechanical.jsonl to have exactly the same keys as paired_dev_gemini_soft.jsonl
    # We will also filter paired_train_gemini_mechanical.jsonl to have exactly the same keys as paired_train_gemini_soft.jsonl
    def align_file(in_path, keys_reference_path, out_path):
        # get pids from reference
        ref_pids = get_pids(keys_reference_path)
        
        # read all pairs from input
        pairs = {}
        with in_path.open("r") as f:
            current_pair = []
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                current_pair.append(row)
                if len(current_pair) == 2:
                    pid = current_pair[0]["id"].rsplit("::", 1)[0]
                    pairs[pid] = current_pair
                    current_pair = []
                    
        # write out only those in reference
        written = 0
        with out_path.open("w") as f:
            for pid in sorted(ref_pids):
                if pid in pairs:
                    f.write(json.dumps(pairs[pid][0], ensure_ascii=False) + "\n")
                    f.write(json.dumps(pairs[pid][1], ensure_ascii=False) + "\n")
                    written += 1
        print(f"Aligned {in_path.name} -> {out_path.name} (wrote {written} pairs)")

    align_file(mech_path, soft_path, mech_path)
    
    # Also align train set
    train_mech_path = root / "paired_train_gemini_mechanical.jsonl"
    train_soft_path = root / "paired_train_gemini_soft.jsonl"
    align_file(train_mech_path, train_soft_path, train_mech_path)

if __name__ == "__main__":
    check_and_align()
