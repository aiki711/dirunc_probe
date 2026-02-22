# scripts/common.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
import json
import re

# --- Constants ---

# Added "which" to the list of directions
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

# Mapping from direction to query token
QUERY_LABEL_STR = {
    "who": "[WHO?]",
    "what": "[WHAT?]",
    "when": "[WHEN?]",
    "where": "[WHERE?]",
    "why": "[WHY?]",
    "how": "[HOW?]",
    "which": "[WHICH?]",
}

SPECIAL_TOKENS = list(QUERY_LABEL_STR.values())
QUERY_TOKENS = tuple(SPECIAL_TOKENS)

# Helper for string concatenation (e.g. " [WHO?] [WHAT?] ...")
# Some scripts might use a joined string
QUERY_TOKENS_STR = " " + " ".join(SPECIAL_TOKENS)


# --- Slot Mapping ---

WHO_KWS = [
    "who", "whom", "whose",
    "attendee", "contact", "recipient", "person", "customer", "client",
    "doctor", "trainer", "agent", "name of person",
    "stylist", "dentist", "therapist", "receiver", "host", "guest",
    # 追加: 数量せず人物を示す語
    "people", "guests", "passengers", "traveler", "travelers",
    "user", "caller", "owner", "driver", "rider", "riders",
    "number of people", "number of passengers", "number of travelers",
    "party size", "party_size", "number of riders", "number of guests",
    # 追加: adults / children 系
    "adult", "adults", "child", "children", "number of adults",
    # 追加: 共有ライド（誰と乗るか→人物）
    "shared ride", "shared_ride",
]
WHEN_KWS = [
    "when",
    "date", "time", "day", "week", "month", "year", "duration",
    "start time", "end time", "arrive", "leave", "check in", "check out"
]
WHERE_KWS = [
    "where",
    "location", "address", "city", "place", "destination", "origin", "departure",
    "airport", "station", "area", "neighborhood", "venue"
]
# WHY_KWS = ["why", "reason", "purpose", "because", "intent"]
HOW_KWS = ["how", "method", "mode", "transport", "payment", "delivery", "format", "via", "type", "option", "ride"]
HOW_MANY_KWS = ["number", "count", "amount", "quantity", "seats", "riders", "guests", "party_size", "people"]
HOW_MODE_KWS = ["shared", "private", "mode", "option"]

# "Which" often implies selection from a set, or specific attribute like price range, star rating etc.
WHICH_KWS = [
    "which", "what category", "what type", "what kind",
    "price", "range", "star", "rating", "type", "choice", "selection", "internet", "parking"
]

def map_slot_to_dir(slot_name: str, slot_desc: str) -> str:
    """
    スロット名を _ や - で単語に分割し、完全一致で判定する安全なアルゴリズム。
    部分一致の誤爆（例: account を count と誤認する）を完全に防ぎます。
    """
    sn = slot_name.lower()
    # スロット名を _ または - で単語の集合に分割 (例: "account_type" -> {"account", "type"})
    sn_words = set(re.split(r'[_\\-]', sn))
    s = (slot_name + " " + slot_desc).lower()

    # --- 特殊ルールの事前処理 ---
    if "phone" in sn_words:
        return "what"  # phone_number は what に固定

    if "name" in sn_words:
        # 人物名の場合は WHO、それ以外（hotel_name, car_name 等）は WHAT
        if sn_words.intersection({"artist", "stylist", "dentist", "doctor", "therapist", "recipient", "receiver", "contact", "person", "user"}):
            return "who"
        return "what"

    # booleanフラグ系 (has_wifi, is_unisex, shared_ride など) は WHAT に固定
    # 説明文のノイズ（例: shared_ride の説明文にある passengers）を回避
    if sn_words.intersection({"is", "has", "shared", "refundable", "offers"}):
        return "what"

    # --- Step 1: スロット名（単語完全一致）による絶対優先判定 ---
    
    # WHERE (場所)
    if sn_words.intersection({"address", "city", "location", "destination", "origin", "airport", "station", "venue", "departure", "area", "where"}):
        return "where"
    
    # WHEN (日時)
    if sn_words.intersection({"date", "time", "day", "year", "month", "duration", "when", "arriveby", "leaveat"}):
        return "when"
    
    # HOW (金額・数量・程度 / How much, How many)
    if sn_words.intersection({"price", "fare", "rent", "cost", "amount", "balance", "total", "number", "count", "size", "seats", "pricerange", "how"}):
        return "how"
    
    # WHO (人)
    if sn_words.intersection({"passengers", "guests", "people", "adults", "children", "travelers", "who", "party"}):
        return "who"
    
    # WHICH (種類・選択)
    if sn_words.intersection({"type", "category", "rating", "class", "genre", "choice", "option", "internet", "parking", "which"}):
        return "which"

    # --- Step 2: 説明文を含めたフォールバック判定 ---
    # ここでも単語境界(\b)を使って安全に判定する
    def has_kw(kws):
        pattern = r'\b(?:' + '|'.join(re.escape(k) for k in kws) + r')\b'
        return bool(re.search(pattern, s))

    if has_kw(WHO_KWS): return "who"
    if has_kw(WHEN_KWS): return "when"
    if has_kw(WHERE_KWS): return "where"
    # why is completely excluded
    if has_kw(WHICH_KWS): return "which"
    # HOW_KWS, HOW_MANY_KWS が定義されていれば結合して判定
    try:
        if has_kw(HOW_KWS + HOW_MANY_KWS): return "how"
    except NameError:
        pass

    return "what"


# --- Label Utilities ---

def build_label_dict(active_dirs: List[str]) -> Dict[str, int]:
    """
    active_dirs に含まれる方向性を 1、それ以外を 0 にしたラベル辞書を返す。
    重複は無視される。

    Example:
        build_label_dict(["when", "where"]) -> {"who":0, "what":0, "when":1, ...}
    """
    active = set(active_dirs)
    return {d: (1 if d in active else 0) for d in DIRS}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """JSONL 形式で書き出す（ディレクトリは自動作成）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- Text Perturbation Placeholders ---

PLACEHOLDER_BY_DIR = {
    "who": "someone",
    "what": "something",
    "when": "sometime",
    "where": "somewhere",
    "why": "for some reason",
    "how": "somehow",
    "which": "one of them", # placeholder for 'which'
}

# --- Text Utilities ---

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def cleanup_deletion_artifacts(text: str) -> str:
    # "in ." / "at ." / "to ." みたいな痕跡を除去
    text = re.sub(r"\b(in|at|on|to|from|for|with)\s*\.", ".", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(in|at|on|to|from|for|with)\s*,", ",", text, flags=re.IGNORECASE)

    # 二重スペースや " ,"/" ." を整える
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def replace_values_in_text(
    text: str,
    values: Sequence[str],
    mode: str,                 # "delete" or "placeholder"
    placeholder: Optional[str] = None,
) -> str:
    out = text
    for v in values:
        if not v:
            continue
        # case-insensitive replace; keep it simple for v0
        pat = re.compile(re.escape(v), flags=re.IGNORECASE)
        out = pat.sub("" if mode == "delete" else (placeholder or ""), out)
    out = normalize_text(out)
    out = cleanup_deletion_artifacts(out)
    return out
