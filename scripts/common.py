# scripts/common.py
from __future__ import annotations
from typing import Dict, List, Optional, Sequence
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
    "attendee", "contact", "recipient", "person", "customer", "client",
    "doctor", "trainer", "agent", "name of person",
    "stylist", "dentist", "therapist", "receiver", "host", "guest"
]
WHEN_KWS = [
    "date", "time", "day", "week", "month", "year", "duration",
    "start time", "end time", "arrive", "leave", "check in", "check out"
]
WHERE_KWS = [
    "location", "address", "city", "place", "destination", "origin",
    "airport", "station", "area", "neighborhood", "venue"
]
WHY_KWS = ["reason", "purpose", "because", "intent"]
HOW_KWS = ["method", "mode", "transport", "payment", "delivery", "format", "via", "type", "option", "shared", "ride"]
HOW_MANY_KWS = ["number", "count", "amount", "quantity", "seats", "riders", "guests", "party_size", "people"]
HOW_MODE_KWS = ["shared", "private", "mode", "option"]

# "Which" often implies selection from a set, or specific attribute like price range, star rating etc.
WHICH_KWS = [
    "price", "range", "star", "rating", "type", "choice", "selection", "internet", "parking"
]

def map_slot_to_dir(slot_name: str, slot_desc: str) -> str:
    """
    Maps a slot (name + description) to one of the DIRS (who, what, when, where, why, how, which).
    """
    s = (slot_name + " " + slot_desc).lower()

    # Priority check
    if any(k in s for k in WHO_KWS):
        return "who"
    if any(k in s for k in WHEN_KWS):
        return "when"
    if any(k in s for k in WHERE_KWS):
        return "where"
    if any(k in s for k in WHY_KWS):
        return "why"
    
    # "Which" vs "How" vs "What" can be tricky.
    # "type" is in HOW_KWS in original code, but often it's "which type".
    # For now, let's keep original logic for 'how' candidates if they are strong indicators of method.
    
    if any(k in s for k in HOW_KWS):
        return "how"
    if any(k in s for k in HOW_MANY_KWS):
        return "how" # or 'how much' ? Original mapped to 'how'
    if any(k in s for k in HOW_MODE_KWS):
        return "how"
        
    # Check for "which" candidates
    if any(k in s for k in WHICH_KWS):
        return "which"

    # fallback
    return "what"


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
