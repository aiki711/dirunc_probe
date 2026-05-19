#!/usr/bin/env python3
import json
from pathlib import Path

def merge_gemini_results():
    root = Path("data/processed/case_grammar")
    filled_sources = [root / "cg_dev.jsonl", root / "cg_train.jsonl"]
    
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

    splits = [
        ("natural_dev_gemini.jsonl", "paired_dev_gemini.jsonl"),
        ("natural_train_gemini.jsonl", "paired_train_gemini.jsonl")
    ]
    
    for in_name, out_name_base in splits:
        gemini_src = root / in_name
        if not gemini_src.exists():
            print(f"Skip: {gemini_src} not found.")
            continue
            
        for strength in ["soft", "strong"]:
            out_name = out_name_base.replace(".jsonl", f"_{strength}.jsonl")
            out_eval = root / out_name
            
            print(f"Merging Gemini {strength} results from {gemini_src}...")
            count = 0
            with out_eval.open("w", encoding="utf-8") as out_f:
                with gemini_src.open("r", encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        pid = row["id"].rsplit("::", 1)[0]
                        
                        if pid in filled_rows:
                            gemini_text = row.get(f"gemini_{strength}")
                            if gemini_text and pid in missing_labels:
                                filled_row = filled_rows[pid]
                                row["labels"] = missing_labels[pid]
                                
                                if "\nUser: " in filled_row["text"]:
                                    context_prefix = filled_row["text"].rsplit("\nUser: ", 1)[0] + "\nUser: "
                                    row["text"] = context_prefix + gemini_text
                                else:
                                    row["text"] = gemini_text
                                
                                row["condition"] = "missing"
                                # metadata に強度を記録
                                row["metadata"]["strength"] = strength
                                
                                out_f.write(json.dumps(filled_row, ensure_ascii=False) + "\n")
                                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                count += 1
            print(f"Created {count} evaluation pairs -> {out_eval}")


if __name__ == "__main__":
    merge_gemini_results()
