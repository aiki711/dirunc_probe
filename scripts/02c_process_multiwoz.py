# scripts/02c_process_multiwoz.py
# MultiWOZ から "Natural Missing" データを逆算アプローチ（ドメイン別アンカー）で抽出するスクリプト。
#
# ロジック（3ステップ）:
#   Step 1 (ゴール収集): 対話の最終システムターンの metadata を参照し、
#                        ドメイン別に充足された semi スロットを収集する。
#   Step 2 (初回アンカー特定): 各ドメインについて、対話を最初から順に確認し、
#                        そのドメインの semi スロットが初めて metadata に記録された
#                        システムターンの直前ユーザーターンをアンカーとする。
#   Step 3 (ラベリング): アンカー発話 vs ゴールスロットで差分 = Missing をラベリング。
#                        ドメインごとに 1 レコード生成。
#
# 出力スキーマ（統一 JSONL）:
#   {
#     "id": "multiwoz::{dlg_id}::domain_{domain}::t{turn_idx}::natural",
#     "source": "multiwoz",
#     "split": "train",
#     "level": "natural",
#     "text": "I am looking for a hotel. [WHO?] ...",
#     "labels": {"who": 0, ..., "where": 1},
#     "intent_or_predicate": "hotel"
#   }

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from common import (
    DIRS,
    QUERY_TOKENS_STR,
    build_label_dict,
    map_slot_to_dir,
    normalize_text,
    write_jsonl,
)

# ---------- 定数 ----------

# metadata の値としてスロット未充足を示す文字列
EMPTY_VALUES = {"", "not mentioned", "none", "dontcare"}


# ---------- ユーティリティ ----------

def read_list_file(path: Path) -> Set[str]:
    """ファイル名リストファイルを読み込んで set で返す。"""
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def is_slot_filled(value: str) -> bool:
    """スロット値が有効な情報を持つかどうか判定する。"""
    return bool(value) and value.strip().lower() not in EMPTY_VALUES


def collect_domain_slots(metadata_domain: Dict[str, Any]) -> Dict[str, str]:
    """1ドメインの metadata から充足されたスロット（semi + book）を返す。"""
    # 検索条件 (semi)
    semi = metadata_domain.get("semi", {})
    slots = {slot: val for slot, val in semi.items() if is_slot_filled(val)}
    
    # 予約条件 (book)
    # book 内には 'booked' (予約済み情報のリスト) と 'people', 'day', 'time', 'stay' 等のスロットがある
    book = metadata_domain.get("book", {})
    for slot, val in book.items():
        if slot == "booked":
            continue  # メタ情報なのでスキップ
        if is_slot_filled(val):
            # semi との重複を避ける（通常はないが安全のため）
            if slot not in slots:
                slots[slot] = val
    
    return slots


def value_mentioned_in_text(value: str, text: str) -> bool:
    """値（文字列）がテキスト中に含まれるかどうか（大文字小文字無視）。"""
    return value.strip().lower() in text.lower()


# ---------- 対話ごとの処理 ----------

def process_dialogue(
    dialogue_id: str,
    log: List[Dict[str, Any]],
    split: str,
) -> List[Dict[str, Any]]:
    """1つの MultiWOZ 対話から Natural Missing レコードを生成して返す。"""

    records: List[Dict[str, Any]] = []

    # MultiWOZ のログ構造:
    #   偶数インデックス = ユーザーターン
    #   奇数インデックス = システムターン（metadata あり）

    if len(log) < 2:
        return []

    # ----------------------------------------------------------
    # Step 1: 最終システムターンの metadata からドメイン別ゴールを収集
    # ----------------------------------------------------------
    # 最終システムターンは最後の奇数インデックス
    last_sys_idx = None
    for i in range(len(log) - 1, -1, -1):
        if i % 2 == 1:  # 奇数 = システムターン
            last_sys_idx = i
            break

    if last_sys_idx is None:
        return []

    last_metadata = log[last_sys_idx].get("metadata", {})
    goal_by_domain: Dict[str, Dict[str, str]] = {}

    for domain, domain_data in last_metadata.items():
        filled = collect_domain_slots(domain_data)
        if filled:
            goal_by_domain[domain] = filled

    if not goal_by_domain:
        return []

    # ----------------------------------------------------------
    # Step 2: ドメインごとの「初回アンカーターン」を特定
    # ----------------------------------------------------------
    # anchor_by_domain[domain] = (user_turn_idx, user_text)
    anchor_by_domain: Dict[str, Tuple[int, str]] = {}

    for i in range(0, len(log) - 1, 2):  # ユーザーターン（偶数）を順に確認
        user_text = normalize_text(log[i].get("text", ""))
        sys_idx = i + 1
        if sys_idx >= len(log):
            break

        sys_metadata = log[sys_idx].get("metadata", {})

        for domain in goal_by_domain:
            if domain in anchor_by_domain:
                continue  # すでに初回アンカーを記録済み

            domain_meta = sys_metadata.get(domain, {})
            filled_now = collect_domain_slots(domain_meta)

            if filled_now:
                # このシステムターンで初めてドメインのスロットが記録された
                # → 直前のユーザーターンをアンカーとする
                anchor_by_domain[domain] = (i, user_text)

    if not anchor_by_domain:
        return []

    # ----------------------------------------------------------
    # Step 3: ドメインごとにラベリング
    # ----------------------------------------------------------
    for domain, (t_idx, anchor_text) in anchor_by_domain.items():
        goal_slots = goal_by_domain.get(domain, {})
        if not goal_slots:
            continue

        # アンカーターン発話に既に含まれているスロット（= 言及済み）
        mentioned_slots: Set[str] = set()
        for slot, value in goal_slots.items():
            if value_mentioned_in_text(value, anchor_text):
                mentioned_slots.add(slot)

        # Missing = ゴールスロット − 言及済みスロット
        missing_slots = set(goal_slots.keys()) - mentioned_slots

        if not missing_slots:
            continue  # 全て言及済みはスキップ

        # スロット名 (domain-slot) を DIR にマッピング
        missing_dirs: List[str] = [
            map_slot_to_dir(f"{domain}-{slot}", "")
            for slot in missing_slots
        ]

        labels = build_label_dict(missing_dirs)

        # 全ラベルが 0 の場合は安全のためスキップ
        if not any(labels.values()):
            continue

        text = anchor_text + QUERY_TOKENS_STR

        records.append({
            "id": f"multiwoz::{dialogue_id}::domain_{domain}::t{t_idx}::natural",
            "source": "multiwoz",
            "split": split,
            "level": "natural",
            "text": text,
            "labels": labels,
            "intent_or_predicate": domain,
            # デバッグ用
            "_meta": {
                "dialogue_id": dialogue_id,
                "domain": domain,
                "anchor_turn_idx": t_idx,
                "missing_slots": sorted(missing_slots),
                "missing_dirs": sorted(set(missing_dirs)),
                "mentioned_slots": sorted(mentioned_slots),
                "goal_slots": sorted(goal_slots.keys()),
            },
        })

    return records


# ---------- メインパイプライン ----------

def process_multiwoz(
    data_path: Path,
    val_list_path: Path,
    test_list_path: Path,
    out_dir: Path,
    limit: int = 0,
) -> None:
    print(f"Loading data from {data_path}...")
    with data_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    val_files = read_list_file(val_list_path)
    test_files = read_list_file(test_list_path)
    train_files = set(data.keys()) - val_files - test_files

    splits = {
        "train": train_files,
        "dev": val_files,
        "test": test_files,
    }

    for split_name, filenames in splits.items():
        print(f"\nProcessing split: {split_name} ({len(filenames)} dialogues)...")
        all_records: List[Dict[str, Any]] = []
        dir_counts: Counter = Counter()
        count = 0

        for fname in sorted(filenames):
            if limit and count >= limit:
                break
            if fname not in data:
                continue

            dialogue = data[fname]
            dialogue_id = fname.replace(".json", "")
            log = dialogue.get("log", [])

            records = process_dialogue(dialogue_id, log, split_name)
            all_records.extend(records)

            for r in records:
                for d, v in r["labels"].items():
                    if v:
                        dir_counts[d] += 1

            count += 1

        out_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(out_path, all_records)

        print(f"[done] split={split_name}  dialogues={count}  records={len(all_records)}  -> {out_path}")
        print("  DIR distribution (positive labels):")
        for d in DIRS:
            c = dir_counts.get(d, 0)
            pct = c / max(len(all_records), 1) * 100
            print(f"    {d:>6}: {c:>8}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="MultiWOZ から Natural Missing データを抽出する。")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw/multiwoz",
        help="MultiWOZ データディレクトリ（data.json を含む）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed/multiwoz/natural",
        help="出力ディレクトリ",
    )
    parser.add_argument("--limit", type=int, default=0, help="スプリットあたりの最大対話数（0=無制限）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    process_multiwoz(
        data_path=data_dir / "data.json",
        val_list_path=data_dir / "valListFile.json",
        test_list_path=data_dir / "testListFile.json",
        out_dir=out_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
