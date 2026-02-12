# scripts/merge_datasets.py
from pathlib import Path

def merge_files(source_files, target_file):
    print(f"Merging into {target_file}...")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    
    with target_file.open("w", encoding="utf-8") as outfile:
        for fname in source_files:
            p = Path(fname)
            if not p.exists():
                print(f"Warning: {p} not found, skipping.")
                continue
            
            print(f"  Appending {p}...")
            with p.open("r", encoding="utf-8") as infile:
                for line in infile:
                    outfile.write(line)
    
    print(f"Done. Saved to {target_file}")

def main():
    qasrl_dir = Path("data/processed/qasrl/dirunc")
    multiwoz_dir = Path("data/processed/multiwoz/dirunc")
    out_dir = Path("data/processed/mixed/dirunc")
    
    # 1. Train: QA-SRL (Balanced) + MultiWOZ
    merge_files(
        [qasrl_dir / "train.jsonl", multiwoz_dir / "train.jsonl"],
        out_dir / "train.jsonl"
    )
    
    # 2. Dev: QA-SRL Dev + MultiWOZ Dev
    merge_files(
        [qasrl_dir / "dev.jsonl", multiwoz_dir / "dev.jsonl"],
        out_dir / "dev.jsonl"
    )
    
    # 3. Test: QA-SRL Test + MultiWOZ Test
    merge_files(
        [qasrl_dir / "test.jsonl", multiwoz_dir / "test.jsonl"],
        out_dir / "test.jsonl"
    )

if __name__ == "__main__":
    main()
