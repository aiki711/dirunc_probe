#!/usr/bin/env python3
import json
from pathlib import Path

def merge_gemini_results():
    root = Path("data/processed/case_grammar")
    # Gemini生成データ（Missingのみ）
    gemini_dev_src = root / "natural_dev_gemini.jsonl"
    # 元のFilledデータが含まれるファイル（複数検索）
    filled_sources = [root / "cg_dev.jsonl", root / "cg_train.jsonl"]
    
    out_eval = root / "temp_eval_paired_gemini.jsonl"

    print("Loading filled/missing rows from multiple sources...")
    filled_rows = {}
    missing_labels = {}
    for src in filled_sources:
        if not src.exists(): continue
        with src.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pid = row["id"].rsplit("::", 1)[0]
                if row["condition"] == "filled":
                    filled_rows[pid] = row
                elif row["condition"] == "missing":
                    missing_labels[pid] = row["labels"]

    print(f"Merging Gemini results from {gemini_dev_src}...")
    count = 0
    with out_eval.open("w", encoding="utf-8") as out_f:
        if not gemini_dev_src.exists():
            print(f"Error: {gemini_dev_src} not found.")
            return
            
        with gemini_dev_src.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pid = row["id"].rsplit("::", 1)[0]
                
                if pid in filled_rows:
                    gemini_text = row.get("gemini_strong") or row.get("gemini_soft")
                    if gemini_text and pid in missing_labels:
                        filled_row = filled_rows[pid]
                        row["labels"] = missing_labels[pid]
                        
                        if "\nUser: " in filled_row["text"]:
                            context_prefix = filled_row["text"].rsplit("\nUser: ", 1)[0] + "\nUser: "
                            row["text"] = context_prefix + gemini_text
                        else:
                            row["text"] = gemini_text
                        
                        row["condition"] = "missing"
                        
                        out_f.write(json.dumps(filled_row, ensure_ascii=False) + "\n")
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1

    print(f"Created {count} evaluation pairs -> {out_eval}")

if __name__ == "__main__":
    merge_gemini_results()
