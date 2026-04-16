# scripts/check_mapping.py
# スロット → DIR マッピングの診断スクリプト。
# SGD / MultiWOZ の全スロット（および QA-SRL の疑問詞）が
# map_slot_to_dir によってどの DIR に分類されるかを確認し、
# CSV とテキストレポートを data/stats/ に出力する。
#
# 使用例:
#   python3 scripts/check_mapping.py
#   python3 scripts/check_mapping.py --sgd_meta data/processed/sgd/slot_meta_by_service_slot.json
#                                    --multiwoz_data data/raw/multiwoz/data.json
#                                    --out_dir data/stats

from __future__ import annotations

import argparse
import csv
import json
import gzip
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import DIRS, map_slot_to_dir

# ---------- SGD ----------

def check_sgd_slots(
    meta_path: Path,
) -> List[Tuple[str, str, str, str]]:
    """
    SGD の slot_meta_by_service_slot.json を読み込み、
    全スロットのマッピング結果を返す。
    Returns: [(service_slot, slot_name, description, mapped_dir), ...]
    """
    if not meta_path.exists():
        print(f"[warn] SGD slot_meta not found: {meta_path}")
        return []

    with meta_path.open(encoding="utf-8") as f:
        slot_meta: Dict[str, Any] = json.load(f)

    results = []
    for key, meta in slot_meta.items():
        # key = "Service::slot_name"
        parts = key.split("::")
        slot_name = parts[-1] if len(parts) >= 2 else key
        desc = str(meta.get("description", "") or "")
        mapped = map_slot_to_dir(slot_name, desc)
        results.append((key, slot_name, desc, mapped))

    return results


# ---------- MultiWOZ ----------

def collect_multiwoz_slots(data_path: Path) -> List[Tuple[str, str, str]]:
    """
    MultiWOZ の data.json から全対話の metadata.semi を走査し、
    ユニークなスロット名を収集する。
    Returns: [(domain_slot, slot_name, mapped_dir), ...]
    """
    if not data_path.exists():
        print(f"[warn] MultiWOZ data.json not found: {data_path}")
        return []

    with data_path.open(encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    unique_slots: set = set()
    for dialogue in data.values():
        for turn in dialogue.get("log", []):
            metadata = turn.get("metadata", {})
            for domain, domain_data in metadata.items():
                # 検索条件 (semi)
                semi = domain_data.get("semi", {})
                for slot in semi.keys():
                    unique_slots.add(f"{domain}-{slot}")
                
                # 予約条件 (book)
                book = domain_data.get("book", {})
                for slot in book.keys():
                    if slot != "booked":
                        unique_slots.add(f"{domain}-{slot}")

    results = []
    for domain_slot in sorted(unique_slots):
        mapped = map_slot_to_dir(domain_slot, "")
        parts = domain_slot.split("-", 1)
        slot_name = parts[1] if len(parts) == 2 else domain_slot
        results.append((domain_slot, slot_name, mapped))

    return results


# ---------- QA-SRL ----------

def check_qasrl_data(
    data_path: Path,
    limit_lines: int = 500,
) -> List[Tuple[str, str, str]]:
    """
    QA-SRL の train.jsonl.gz から質問をサンプリングし、
    そのマッピング結果を返す。
    Returns: [(question, first_word, mapped_dir), ...]
    """
    if not data_path.exists():
        print(f"[warn] QA-SRL data not found: {data_path}")
        return []

    unique_questions: Dict[str, str] = {}
    
    opener = gzip.open if str(data_path).endswith(".gz") else open
    with opener(data_path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit_lines:
                break
            try:
                row = json.loads(line)
                for v_key, v_data in row.get("verbEntries", {}).items():
                    for q_str, q_data in v_data.get("questionLabels", {}).items():
                        q_text = q_data.get("questionString", q_str)
                        first_word = q_text.split(" ")[0].lower().replace("?", "")
                        unique_questions[q_text] = first_word
            except Exception:
                continue

    results = []
    for q_text, first in sorted(unique_questions.items()):
        # map_slot_to_dir に疑問詞そのものを渡す
        mapped = map_slot_to_dir(first, "")
        results.append((q_text, first, mapped))

    return results


# ---------- レポート出力 ----------

def print_by_dir(title: str, entries: List[Tuple], dir_col: int = -1) -> None:
    """DIR ごとにグループ化して stdout に出力する。"""
    by_dir: Dict[str, List] = defaultdict(list)
    for entry in entries:
        by_dir[entry[dir_col]].append(entry)

    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    for d in DIRS:
        group = by_dir.get(d, [])
        print(f"\n--- DIR: {d.upper()} ({len(group)} items) ---")
        for entry in group:
            display = "  - " + " | ".join(str(x) for x in entry[:-1])
            # desc がある場合（SGD）
            if title.startswith("SGD") and len(entry) > 3:
                desc = entry[2]
                print(f"{display[:80]}")
                print(f"      desc: \"{desc[:60]}\"")
            else:
                print(display)


def write_csv(out_path: Path, header: List[str], rows: List[Tuple]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
    print(f"  -> Saved: {out_path}")


def write_text_report(out_path: Path, title: str, entries: List[Tuple], dir_col: int = -1) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    by_dir: Dict[str, List] = defaultdict(list)
    for entry in entries:
        by_dir[entry[dir_col]].append(entry)

    lines = [f"{title} (Total: {len(entries)})\n"]
    for d in DIRS:
        group = by_dir.get(d, [])
        lines.append(f"\n=== DIR: {d.upper()} ({len(group)} items) ===")
        for entry in group:
            lines.append("  - " + " | ".join(str(x) for x in entry))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> Saved: {out_path}")


# ---------- メイン ----------

def main() -> None:
    parser = argparse.ArgumentParser(description="スロット・質問 → DIR マッピング診断スクリプト")
    parser.add_argument(
        "--sgd_meta",
        type=str,
        default="data/processed/sgd/slot_meta_by_service_slot.json",
    )
    parser.add_argument(
        "--multiwoz_data",
        type=str,
        default="data/raw/multiwoz/data.json",
    )
    parser.add_argument(
        "--qasrl_data",
        type=str,
        default="temp_qasrl/qasrl-bank/data/qasrl-v2/orig/train.jsonl.gz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/stats",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ==================== SGD ====================
    print("\n[SGD] スロットマッピング診断中...")
    sgd_results = check_sgd_slots(Path(args.sgd_meta))
    if sgd_results:
        write_csv(out_dir / "mapping_sgd.csv", ["service_slot", "slot_name", "description", "mapped_dir"], sgd_results)
        write_text_report(out_dir / "mapping_sgd.txt", "SGD Slot Mapping Report", sgd_results, dir_col=3)
        print_by_dir("SGD Slot Mapping", sgd_results, dir_col=3)

    # ==================== MultiWOZ ====================
    print("\n[MultiWOZ] スロットマッピング診断中...")
    mwoz_results = collect_multiwoz_slots(Path(args.multiwoz_data))
    if mwoz_results:
        write_csv(out_dir / "mapping_multiwoz.csv", ["domain_slot", "slot_name", "mapped_dir"], mwoz_results)
        write_text_report(out_dir / "mapping_multiwoz.txt", "MultiWOZ Slot Mapping Report", mwoz_results, dir_col=2)
        print_by_dir("MultiWOZ Slot Mapping", mwoz_results, dir_col=2)

    # ==================== QA-SRL ====================
    print("\n[QA-SRL] 質問マッピング診断中...")
    qasrl_results = check_qasrl_data(Path(args.qasrl_data))
    if qasrl_results:
        write_csv(out_dir / "mapping_qasrl.csv", ["question", "wh_word", "mapped_dir"], qasrl_results)
        write_text_report(out_dir / "mapping_qasrl.txt", "QA-SRL Question Mapping Report", qasrl_results, dir_col=2)
        print_by_dir("QA-SRL Question Mapping", qasrl_results, dir_col=2)

    print("\n診断完了。")


if __name__ == "__main__":
    main()
