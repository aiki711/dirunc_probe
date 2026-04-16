# scripts/02b_process_qasrl.py
# QA-SRL から "Natural Missing" データを抽出するスクリプト。
#
# QA-SRL v2 の特性:
#   全ての質問には有効な回答スパンが付与されている（answerJudgments は全て isValid: true）。
#   そのため「回答なし = Missing」ではなく、
#   「述語に対してアノテーターが生成しなかった WH 質問の DIR = Missing」
#   （= 述語が自然に要求するはずの役割が省略されている）という解釈を採用する。
#
# ロジック:
#   各述語（動詞）について、生成された質問のWH疑問詞から DIR 集合（answered_dirs）を求める。
#   DIRS から answered_dirs を引いた差分 = missing_dirs。
#   ただし、全 DIR が missing（質問が0件）の述語はスキップする。
#   また、意味的に必然的でない DIR（who/whichなど）のみ missing にならないよう、
#   answered がある述語のみを対象とする（質問が1件以上ある述語のみ）。
#
# 出力スキーマ（統一 JSONL）:
#   {
#     "id": "qasrl::{sent_id}::v{verb_idx}::natural",
#     "source": "qasrl",
#     "split": "train",
#     "level": "natural",
#     "text": "Geologists study how rocks form. [WHO?] ...",
#     "labels": {"who": 0, ..., "where": 1},
#     "intent_or_predicate": "study"
#   }

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from common import (
    DIRS,
    QUERY_TOKENS_STR,
    build_label_dict,
    normalize_text,
    write_jsonl,
)

# ---------- QA-SRL ユーティリティ ----------

WH_MAPPING = {
    "who": "who",
    "what": "what",
    "when": "when",
    "where": "where",
    "why": "why",
    "how": "how",
    "which": "which",
}

# why/who/which は文脈なしでは必ずしも Missing と言えないが、QA-SRLでは抽出対象に含める
MEANINGFUL_MISSING_DIRS = {"who", "what", "when", "where", "why", "how"}


def get_dir_from_question(question: str) -> str:
    """質問文の先頭語から DIR を返す。"""
    first_word = question.lower().strip().split(" ")[0]
    return WH_MAPPING.get(first_word, "what")


def get_dir_from_slots(slots: Dict[str, str]) -> str:
    """questionSlots の wh フィールド from DIR を返す（より正確）。"""
    wh = slots.get("wh", "").lower()
    return WH_MAPPING.get(wh, "what")


# ---------- ファイル処理 ----------

def process_file(
    input_path: Path,
    output_path: Path,
    split_name: str,
    limit: int = 0,
) -> None:
    print(f"Processing {input_path} -> {output_path} (limit={limit})")

    rows_out: List[Dict[str, Any]] = []
    dir_counts: Counter = Counter()
    n_skipped_no_verb = 0
    n_skipped_no_answered = 0
    n_skipped_no_meaningful_missing = 0

    opener = gzip.open if str(input_path).endswith(".gz") else open

    with opener(input_path, "rt", encoding="utf-8") as f:
        count = 0
        for line in f:
            if limit and count >= limit:
                break

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            tokens = row.get("sentenceTokens", [])
            if not tokens:
                count += 1
                continue

            sentence_text = " ".join(tokens)
            sent_id = row.get("sentenceId", str(hash(sentence_text)))

            verb_entries = row.get("verbEntries", {})
            if not verb_entries:
                n_skipped_no_verb += 1
                count += 1
                continue

            for v_key, v_data in verb_entries.items():
                verb_idx = v_data.get("verbIndex", v_key)
                q_labels = v_data.get("questionLabels", {})

                if not q_labels:
                    continue

                # 述語の表層形を取得
                predicate = (
                    tokens[verb_idx]
                    if isinstance(verb_idx, int) and verb_idx < len(tokens)
                    else str(verb_idx)
                )

                # 生成された質問の DIR 集合を収集（questionSlots.wh を優先使用）
                answered_dirs: Set[str] = set()
                for q_str, q_data in q_labels.items():
                    slots = q_data.get("questionSlots", {})
                    if slots:
                        q_dir = get_dir_from_slots(slots)
                    else:
                        question_text = q_data.get("questionString", q_str)
                        q_dir = get_dir_from_question(question_text)
                    answered_dirs.add(q_dir)

                if not answered_dirs:
                    n_skipped_no_answered += 1
                    continue

                # missing = MEANINGFUL_MISSING_DIRS から answered を引いた差分
                missing_dirs = MEANINGFUL_MISSING_DIRS - answered_dirs

                if not missing_dirs:
                    n_skipped_no_meaningful_missing += 1
                    continue

                labels = build_label_dict(list(missing_dirs))

                # DIR カウント更新
                for d in missing_dirs:
                    dir_counts[d] += 1

                text = normalize_text(sentence_text) + QUERY_TOKENS_STR

                rows_out.append({
                    "id": f"qasrl::{sent_id}::v{verb_idx}::natural",
                    "source": "qasrl",
                    "split": split_name,
                    "level": "natural",
                    "text": text,
                    "labels": labels,
                    "intent_or_predicate": predicate,
                    # デバッグ用
                    "_meta": {
                        "sentence": sentence_text,
                        "verb_idx": verb_idx,
                        "answered_dirs": sorted(answered_dirs),
                        "missing_dirs": sorted(missing_dirs),
                    },
                })

            count += 1

    write_jsonl(output_path, rows_out)

    print(f"Saved {len(rows_out)} examples to {output_path}")
    print(f"  Skipped (no verb entries): {n_skipped_no_verb}")
    print(f"  Skipped (no answered q):   {n_skipped_no_answered}")
    print(f"  Skipped (no meaningful missing): {n_skipped_no_meaningful_missing}")
    print("  DIR distribution (positive labels):")
    for d in DIRS:
        c = dir_counts.get(d, 0)
        pct = c / max(len(rows_out), 1) * 100
        print(f"    {d:>6}: {c:>8}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="QA-SRL から Natural Missing データを抽出する。")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="temp_qasrl/qasrl-bank/data/qasrl-v2/orig",
        help="QA-SRL データディレクトリ（.jsonl.gz ファイルを含む）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/qasrl/natural",
        help="出力ディレクトリ",
    )
    parser.add_argument("--limit", type=int, default=0, help="1ファイルあたりの最大処理行数（0=無制限）")
    args = parser.parse_args()

    in_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if not in_dir.exists():
        print(f"[error] Input directory not found: {in_dir}")
        return

    files_map = {
        "train.jsonl.gz": "train",
        "dev.jsonl.gz": "dev",
        "test.jsonl.gz": "test",
    }

    for fname, split in files_map.items():
        fpath = in_dir / fname
        if fpath.exists():
            out_path = out_dir / f"{split}.jsonl"
            process_file(fpath, out_path, split, limit=args.limit)
        else:
            print(f"[skip] {fname} not found.")


if __name__ == "__main__":
    main()
