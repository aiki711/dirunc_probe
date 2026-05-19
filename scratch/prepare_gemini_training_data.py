import json
import os
from pathlib import Path

BASE_DIR = Path("/home/admin/work/s2550009/dirunc_probe")
DATA_DIR = BASE_DIR / "data/processed/case_grammar"

def process_file(orig_file: Path, gemini_file: Path, out_soft: Path, out_strong: Path):
    if not orig_file.exists() or not gemini_file.exists():
        print(f"Skipping {orig_file.name} (not found)")
        return

    # Load gemini rewrites mapped by base_id
    gemini_dict = {}
    with gemini_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            # base id is id without ::filled or ::missing
            base_id = row["id"].rsplit("::", 1)[0]
            gemini_dict[base_id] = {
                "soft": row.get("gemini_soft", ""),
                "strong": row.get("gemini_strong", "")
            }

    print(f"Loaded {len(gemini_dict)} rewrites from {gemini_file.name}")

    count = 0
    with orig_file.open("r", encoding="utf-8") as f_in, \
         out_soft.open("w", encoding="utf-8") as fs, \
         out_strong.open("w", encoding="utf-8") as fst:
         
        for line in f_in:
            if not line.strip(): continue
            row = json.loads(line)
            base_id = row["id"].rsplit("::", 1)[0]
            
            # If we don't have Gemini rewrite for this pair, skip it to keep train/dev clean
            if base_id not in gemini_dict:
                continue
                
            is_filled = row["id"].endswith("::filled")
            
            row_soft = dict(row)
            row_strong = dict(row)
            
            if not is_filled:
                # It's a missing row, inject the Gemini rewrite
                text = row["text"]
                if "\nUser: " in text:
                    prefix, _ = text.rsplit("\nUser: ", 1)
                    # For Soft
                    soft_text = gemini_dict[base_id]["soft"]
                    if soft_text:
                        row_soft["text"] = prefix + "\nUser: " + soft_text
                    
                    # For Strong
                    strong_text = gemini_dict[base_id]["strong"]
                    if strong_text:
                        row_strong["text"] = prefix + "\nUser: " + strong_text
                else:
                    # No User prefix, just replace completely
                    if gemini_dict[base_id]["soft"]: row_soft["text"] = gemini_dict[base_id]["soft"]
                    if gemini_dict[base_id]["strong"]: row_strong["text"] = gemini_dict[base_id]["strong"]
            
            # Write out
            fs.write(json.dumps(row_soft, ensure_ascii=False) + "\n")
            fst.write(json.dumps(row_strong, ensure_ascii=False) + "\n")
            count += 1
            
    print(f"Wrote {count} lines to {out_soft.name} & {out_strong.name}")

if __name__ == "__main__":
    print("Processing Dev data...")
    process_file(
        DATA_DIR / "natural_dev.jsonl",
        DATA_DIR / "natural_dev_gemini.jsonl",
        DATA_DIR / "natural_dev_soft.jsonl",
        DATA_DIR / "natural_dev_strong.jsonl"
    )
    print("Processing Train data...")
    process_file(
        DATA_DIR / "natural_train.jsonl",
        DATA_DIR / "natural_train_gemini.jsonl",
        DATA_DIR / "natural_train_soft.jsonl",
        DATA_DIR / "natural_train_strong.jsonl"
    )
