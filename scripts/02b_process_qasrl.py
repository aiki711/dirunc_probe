# scripts/02b_process_qasrl.py
from __future__ import annotations

import argparse
import json
import gzip
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from common import DIRS, QUERY_TOKENS_STR, map_slot_to_dir, PLACEHOLDER_BY_DIR, normalize_text, replace_values_in_text

# --- QA-SRL Specific Constants ---

WH_MAPPING = {
    "who": "who",
    "what": "what",
    "when": "when",
    "where": "where",
    "why": "why",
    "how": "how",
    "which": "which",
}

def get_dir_from_question(question: str) -> str:
    q_lower = question.lower().strip()
    first_word = q_lower.split(" ")[0]
    if first_word in WH_MAPPING:
        return WH_MAPPING[first_word]
    # Fallback for "how much", "how long" etc.
    if first_word == "how":
        return "how"
    return "what"

def extract_answers_from_spans(tokens: List[str], judgments: List[Dict[str, Any]]) -> List[str]:
    """
    Extract answer strings from token spans.
    judgments: list of dicts with 'isValid' and 'spans' [[start, end), ...]
    """
    answers = []
    for judg in judgments:
        if not judg.get("isValid", False):
            continue
        spans = judg.get("spans", [])
        for (start, end) in spans:
            # QA-SRL 2.0 spans are usually [start, end) token indices
            if start < 0 or end > len(tokens):
                continue
            ans_tokens = tokens[start:end]
            ans_str = " ".join(ans_tokens)
            if ans_str:
                answers.append(ans_str)
    return list(set(answers)) # Unique answers

def process_file(
    input_path: Path,
    output_path: Path,
    split_name: str,
    seed: int = 42,
    limit: int = 0,
    max_per_class: int = 0
) -> None:
    
    print(f"Processing {input_path} -> {output_path} (limit={limit}, max_per_class={max_per_class})")
    
    rows_out: List[Dict[str, Any]] = []
    dir_counts = Counter()
    
    # Store candidates for balancing if max_per_class > 0 and split is train
    do_balance = (max_per_class > 0 and split_name == "train")
    candidates_by_dir: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DIRS}
    final_rows: List[Dict[str, Any]] = []

    rng = random.Random(seed)
    
    # Read GZ file
    with gzip.open(input_path, "rt", encoding="utf-8") as f:
        count = 0
        for line in f:
            if limit and count >= limit:
                break
                
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Parse Structure
            # {
            #   "sentenceTokens": ["Both", "occur", ...],
            #   "verbEntries": {
            #       "1": {
            #           "verbIndex": 1,
            #           "questionLabels": {
            #               "What occurs?": {
            #                   "questionString": "...",
            #                   "answerJudgments": [...]
            #               }
            #           }
            #       }
            #   }
            # }

            tokens = row.get("sentenceTokens", [])
            if not tokens:
                continue
            
            sentence_text = " ".join(tokens)
            
            verb_entries = row.get("verbEntries", {})
            
            for v_key, v_data in verb_entries.items():
                verb_idx = v_data.get("verbIndex")
                q_labels = v_data.get("questionLabels", {})
                
                if not q_labels:
                    continue
                
                # Collect QAs for this verb
                qas = []
                for q_str, q_data in q_labels.items():
                    question_text = q_data.get("questionString", q_str)
                    judgments = q_data.get("answerJudgments", [])
                    answers = extract_answers_from_spans(tokens, judgments)
                    
                    if not answers:
                        # Skip if no valid answer (e.g. all invalid judgments)
                        continue
                        
                    q_dir = get_dir_from_question(question_text)
                    dir_counts[q_dir] += 1
                    
                    qas.append({
                        "question": question_text,
                        "answers": answers,
                        "dir": q_dir
                    })
                
                if not qas:
                    continue
                    
                # Generate Examples
                # 1. Resolved
                
                relevant_dirs = set(qa["dir"] for qa in qas)
                db_id = f"{split_name}::{row.get('sentenceId', hash(sentence_text))}::v{verb_idx}"
                
                text_resolved = normalize_text(sentence_text) + QUERY_TOKENS_STR
                labels_resolved = {d: 0 for d in DIRS}
                
                if do_balance:
                   # For balancing, we track "resolved" separately or just include them? 
                   # Usually resolved match the number of unresolved. 
                   # Let's save resolved to final_rows directly (or skip if we only want balanced positive examples)
                   # Strategy: We want balanced POSITIVE samples. Resolved samples are negative (all 0).
                   # We can keep all resolved, OR sample them to match total positives.
                   # For simplicity, let's keep all resolved but maybe downsample later if too many.
                   # Actually, too many resolved (negatives) might be fine.
                   final_rows.append({
                       "id": f"{db_id}::resolved",
                       "text": text_resolved,
                       "labels": labels_resolved,
                       "split": split_name,
                       "meta": {"type": "resolved", "verb_idx": verb_idx, "sentence": sentence_text}
                   })
                else:
                    rows_out.append({
                    "id": f"{db_id}::resolved",
                    "text": text_resolved,
                    "labels": labels_resolved,
                    "split": split_name,
                    "meta": {"type": "resolved", "verb_idx": verb_idx, "sentence": sentence_text}
                })
                
                # 2. Unresolved
                for i, target_qa in enumerate(qas):
                    tgt_dir = target_qa["dir"]
                    tgt_answers = target_qa["answers"]
                    
                    for level in [0, 1]:
                        if level == 0:
                            text_perturbed = replace_values_in_text(sentence_text, tgt_answers, mode="delete")
                        else:
                            ph = PLACEHOLDER_BY_DIR.get(tgt_dir, "something")
                            text_perturbed = replace_values_in_text(sentence_text, tgt_answers, mode="placeholder", placeholder=ph)
                        
                        if normalize_text(text_perturbed) == normalize_text(sentence_text):
                            continue
                            
                        text_final = normalize_text(text_perturbed) + QUERY_TOKENS_STR
                        
                        labels = {d: 0 for d in DIRS}
                        labels[tgt_dir] = 1
                        
                        if do_balance:
                            candidates_by_dir[tgt_dir].append({
                                "id": f"{db_id}::qa{i}::lvl{level}",
                                "text": text_final,
                                "labels": labels,
                                "split": split_name,
                                "meta": {
                                    "type": "unresolved",
                                    "level": level,
                                    "target_dir": tgt_dir,
                                    "question": target_qa["question"]
                                }
                            })
                        else:
                            rows_out.append({
                            "id": f"{db_id}::qa{i}::lvl{level}",
                            "text": text_final,
                            "labels": labels,
                            "split": split_name,
                            "meta": {
                                "type": "unresolved",
                                "level": level,
                                "target_dir": tgt_dir,
                                "question": target_qa["question"]
                            }
                        })
                
            count += 1
            
    # Apply Balancing
    if do_balance:
        print("Applying balancing...")
        balanced_positives = []
        for d in DIRS:
            cands = candidates_by_dir[d]
            n_cands = len(cands)
            print(f"  {d}: found {n_cands}")
            if n_cands > max_per_class:
                selected = rng.sample(cands, max_per_class)
                balanced_positives.extend(selected)
            else:
                balanced_positives.extend(cands)
        
        # Merge resolved and balanced positives
        # Optional: Downsample resolved to match positives ratio? 
        # For now, keep all resolved to maintain variety of "complete" sentences.
        final_rows.extend(balanced_positives)
        rows_out = final_rows
        
        # Shuffle
        rng.shuffle(rows_out)
        
        # Stats
        label_counts = Counter()
        for r in rows_out:
            for k, v in r["labels"].items():
                if v == 1: label_counts[k] += 1
        print("Balanced Stats:", label_counts)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(rows_out)} examples to {output_path}")
    print("DIR Stats:", dir_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="temp_qasrl/qasrl-bank/data/qasrl-v2/orig", help="Input directory")
    parser.add_argument("--out_dir", type=str, default="data/processed/qasrl/dirunc", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Limit rows per file")
    parser.add_argument("--max_per_class", type=int, default=20000, help="Max examples per class for training balancing")
    args = parser.parse_args()
    
    in_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    
    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}")
        return

    # Files to process
    # Map input filename to output split name
    files_map = {
        "train.jsonl.gz": "train",
        "dev.jsonl.gz": "dev",
        "test.jsonl.gz": "test"
    }
    
    for fname, split in files_map.items():
        fpath = in_dir / fname
        if fpath.exists():
            out_path = out_dir / f"{split}.jsonl"
            process_file(fpath, out_path, split, limit=args.limit, max_per_class=args.max_per_class)
        else:
            print(f"Skipping {fname}, not found.")

if __name__ == "__main__":
    main()
