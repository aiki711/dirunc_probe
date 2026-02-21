# scripts/02a_process_sgd.py
# SGD から "Natural Missing" データを逆算アプローチで抽出するスクリプト。
#
# ロジック（2ステップ）:
#   Step 1 (ゴール収集): 対話全ターンを走査し、各 (service, intent) ペアについて
#                        最終的に充足された全スロットを収集する（= ユーザーの真のゴール）。
#   Step 2 (ラベリング): active_intent が初めて登場したユーザーターンの発話を text とし、
#                        ゴールスロット − 初回に言及済みスロット = Missing スロットとしてラベリング。
#
# 出力スキーマ（統一 JSONL）:
#   {
#     "id": "sgd::dialog_1::turn_0::natural",
#     "source": "sgd",
#     "split": "train",
#     "level": "natural",
#     "text": "I need to book a flight. [WHO?] [WHAT?] ...",
#     "labels": {"who": 0, ..., "when": 1},
#     "intent_or_predicate": "BookFlight"
#   }

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    DIRS,
    QUERY_TOKENS_STR,
    build_label_dict,
    map_slot_to_dir,
    normalize_text,
    write_jsonl,
)

SPLITS_DEFAULT = ["train", "dev", "test"]

# ---------- IO ユーティリティ ----------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_dialogue_files(split_dir: Path) -> List[Path]:
    files = sorted(split_dir.glob("dialogues_*.json"))
    if not files:
        files = sorted(split_dir.glob("dialogue_*.json"))
    return files


# ---------- SGD データ操作 ----------

def is_user_turn(turn: Dict[str, Any]) -> bool:
    return turn.get("speaker", "").upper() == "USER"


def get_frame_for_service(turn: Dict[str, Any], service: str) -> Optional[Dict[str, Any]]:
    for fr in turn.get("frames", []):
        if fr.get("service") == service:
            return fr
    return None


def extract_active_intent(frame: Dict[str, Any]) -> Optional[str]:
    st = frame.get("state")
    if not st:
        return None
    ai = st.get("active_intent")
    if not ai or ai == "NONE":
        return None
    return ai


def extract_slot_values(frame: Dict[str, Any]) -> Dict[str, List[str]]:
    """フレームから slot_values を {slot: [value, ...]} 形式で返す。"""
    st = frame.get("state") or {}
    sv = st.get("slot_values") or {}
    out: Dict[str, List[str]] = {}
    for k, v in sv.items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = [str(v)]
    return out


# ---------- 対話ごとの処理 ----------

def process_dialogue(
    dialogue: Dict[str, Any],
    split: str,
    required_map: Dict[str, Dict[str, Any]],
    slot_meta: Dict[str, Dict[str, Any]],
    max_dialogues_reached: bool = False,
) -> List[Dict[str, Any]]:
    """1つの対話から Natural Missing レコードを生成して返す。"""

    dialogue_id = str(dialogue.get("dialogue_id", "unknown"))
    turns = dialogue.get("turns", [])

    # ----------------------------------------------------------
    # Step 1: 全ターンを走査し、各 (service, intent) のゴールスロットを収集
    # ゴール = この対話で最終的に充足されたスロット群
    # ----------------------------------------------------------
    # goal_slots[service::intent] = {slot: [value, ...]}
    goal_slots: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    for turn in turns:
        for fr in turn.get("frames", []):
            service = fr.get("service")
            intent = extract_active_intent(fr)
            if not service or not intent:
                continue
            key = f"{service}::{intent}"
            sv = extract_slot_values(fr)
            for slot, vals in sv.items():
                # 既存と同じスロットがあれば値を上書き（最後に充足された値を優先）
                goal_slots[key][slot] = vals

    if not goal_slots:
        return []

    # ----------------------------------------------------------
    # Step 2: ユーザーターンを順に辿り、active_intent の初回登場ターンでラベリング
    # ----------------------------------------------------------
    seen_intent_keys: set = set()
    records: List[Dict[str, Any]] = []

    for t_idx, turn in enumerate(turns):
        if not is_user_turn(turn):
            continue

        for fr in turn.get("frames", []):
            service = fr.get("service")
            intent = extract_active_intent(fr)
            if not service or not intent:
                continue

            key = f"{service}::{intent}"
            if key in seen_intent_keys:
                continue  # 2回目以降のターンはスキップ
            seen_intent_keys.add(key)

            # ゴールスロットが空なら対象外
            if not goal_slots.get(key):
                continue

            # 初回ターン時点で既に言及されているスロット（slot_values に記録済み）
            observed_now = set(extract_slot_values(fr).keys())

            # ゴールと必須スロットの両方を考慮
            # required_map があれば必須スロットを使い、なければゴールスロット全体を使用
            req_key_data = required_map.get(key, {})
            required_slots: List[str] = req_key_data.get("required_slots", [])

            # ゴールのうち required_slots に含まれるものだけを対象
            # required_slots が定義されていない場合はゴール全体を使用
            goal_for_intent = goal_slots[key]
            if required_slots:
                target_slots = {s for s in required_slots if s in goal_for_intent}
            else:
                target_slots = set(goal_for_intent.keys())

            # Missing = ゴールスロット − 初回時点で言及済み
            missing_slots = target_slots - observed_now

            if not missing_slots:
                continue  # 全て言及済みならスキップ

            # Missing スロットを DIR にマッピング
            missing_dirs: List[str] = []
            for slot in missing_slots:
                meta = slot_meta.get(f"{service}::{slot}", {})
                desc = str(meta.get("description", "") or "")
                d = map_slot_to_dir(slot, desc)
                missing_dirs.append(d)

            labels = build_label_dict(missing_dirs)

            # 全ラベルが 0 の場合はスキップ（安全チェック）
            if not any(labels.values()):
                continue

            text = normalize_text(turn["utterance"]) + QUERY_TOKENS_STR

            records.append({
                "id": f"sgd::{dialogue_id}::t{t_idx}::{service}::{intent}::natural",
                "source": "sgd",
                "split": split,
                "level": "natural",
                "text": text,
                "labels": labels,
                "intent_or_predicate": intent,
                # デバッグ用（学習スクリプトは無視）
                "_meta": {
                    "service": service,
                    "dialogue_id": dialogue_id,
                    "turn_idx": t_idx,
                    "missing_slots": sorted(missing_slots),
                    "missing_dirs": sorted(set(missing_dirs)),
                    "goal_slots": sorted(goal_for_intent.keys()),
                    "observed_slots": sorted(observed_now),
                },
            })

    return records


# ---------- メインパイプライン ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="SGD から Natural Missing データを抽出する。")
    ap.add_argument("--sgd_root", type=str, default="data/raw/sgd", help="SGD データのルートディレクトリ")
    ap.add_argument(
        "--required_map",
        type=str,
        default="data/processed/sgd/required_slots_by_service_intent.json",
        help="サービス+インテントごとの必須スロット定義 JSON",
    )
    ap.add_argument(
        "--slot_meta",
        type=str,
        default="data/processed/sgd/slot_meta_by_service_slot.json",
        help="スロットメタ情報（説明文）JSON",
    )
    ap.add_argument("--out_dir", type=str, default="data/processed/sgd/natural", help="出力ディレクトリ")
    ap.add_argument("--splits", type=str, default=",".join(SPLITS_DEFAULT), help="対象スプリット（カンマ区切り）")
    ap.add_argument(
        "--max_dialogues_per_split",
        type=int,
        default=0,
        help="スプリットあたりの最大対話数（0=無制限、デバッグ用）",
    )
    args = ap.parse_args()

    sgd_root = Path(args.sgd_root)
    out_dir = Path(args.out_dir)

    # 必須スロット & スロットメタの読み込み
    required_map_path = Path(args.required_map)
    slot_meta_path = Path(args.slot_meta)

    required_map: Dict[str, Any] = {}
    if required_map_path.exists():
        required_map = json.loads(required_map_path.read_text(encoding="utf-8"))
    else:
        print(f"[warn] required_map not found: {required_map_path} — ゴール全スロットを対象にします")

    slot_meta: Dict[str, Any] = {}
    if slot_meta_path.exists():
        slot_meta = json.loads(slot_meta_path.read_text(encoding="utf-8"))
    else:
        print(f"[warn] slot_meta not found: {slot_meta_path} — スロット説明なしでマッピングします")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    max_d = args.max_dialogues_per_split if args.max_dialogues_per_split > 0 else None

    for split in splits:
        split_dir = sgd_root / split
        if not split_dir.exists():
            print(f"[skip] split dir not found: {split_dir}")
            continue

        files = find_dialogue_files(split_dir)
        if not files:
            print(f"[skip] no dialogue files in {split_dir}")
            continue

        all_records: List[Dict[str, Any]] = []
        n_dialogues = 0

        for fp in files:
            dialogues = read_json(fp)
            for d in dialogues:
                n_dialogues += 1
                if max_d and n_dialogues > max_d:
                    break

                records = process_dialogue(
                    dialogue=d,
                    split=split,
                    required_map=required_map,
                    slot_meta=slot_meta,
                )
                all_records.extend(records)

            if max_d and n_dialogues >= max_d:
                break

        out_path = out_dir / f"{split}.jsonl"
        write_jsonl(out_path, all_records)

        # 統計表示
        from collections import Counter
        dir_counts: Counter = Counter()
        for r in all_records:
            for d, v in r["labels"].items():
                if v:
                    dir_counts[d] += 1

        print(f"[done] split={split}  dialogues={n_dialogues}  records={len(all_records)}  -> {out_path}")
        print("  DIR distribution (positive labels):")
        for d in DIRS:
            c = dir_counts.get(d, 0)
            pct = c / max(len(all_records), 1) * 100
            print(f"    {d:>6}: {c:>8}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
