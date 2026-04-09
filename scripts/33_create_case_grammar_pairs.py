#!/usr/bin/env python3
# scripts/33_create_case_grammar_pairs.py
"""
Case Grammar Minimal Pair Dataset Generator (v1)

Extends 31_create_semantic_minimal_pairs.py with:
  - Predicate + theme_domain identification per intent/domain
  - Deep Case Role labeling (Agent, Theme, Location, Source, Goal, Time, Manner)
  - CaseStateTracker: tracks which case roles have been filled over a K-turn window
  - Two saturation metrics per sample:
      saturation_score  : float  0.0–1.0  (filled mandatory / total mandatory)
      is_saturated      : bool   True when ALL mandatory cases are filled

Output JSONL format (per row):
{
  "id": "sgd::d_001::t4::origin::missing",
  "text": "...",
  "labels": { "who": 0, "what": 0, "when": 0, "where": 1, "why": 0, "how": 0, "which": 0 },
  "condition": "missing",
  "dataset": "sgd",
  "metadata": {
    "predicate": "search",
    "theme_domain": "Flight",
    "case_role": "Source",
    "slot": "from_location",
    "dropped_span": "New York",
    "saturation_score": 0.33,
    "is_saturated": false,
    "filled_roles_before": ["Time"]
  }
}
"""

from __future__ import annotations
import os, sys, json, random, re, gzip
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Sequence, Tuple

sys.path.insert(0, os.getcwd())

try:
    from scripts.common import (
        DIRS, build_label_dict, write_jsonl,
        replace_values_in_text, cleanup_deletion_artifacts, map_slot_to_dir,
    )
    from scripts.case_frames import (
        CaseFrame, slot_to_role, get_frame_for_intent,
        get_frame_for_multiwoz_domain,
        ROLE_AGENT, ROLE_THEME, ROLE_LOCATION, ROLE_SOURCE, ROLE_GOAL,
        ROLE_TIME, ROLE_MANNER, ALL_ROLES,
    )
except ImportError as e:
    print(f"Import error: {e}. Run from project root with PYTHONPATH=.")
    sys.exit(1)

random.seed(42)

# ---------------------------------------------------------------------------
# Case-role -> 5W1H direction (for backward compatibility with probe labels)
# ---------------------------------------------------------------------------
ROLE_TO_DIR: Dict[str, str] = {
    ROLE_AGENT:    "who",
    ROLE_THEME:    "what",
    ROLE_LOCATION: "where",
    ROLE_SOURCE:   "where",
    ROLE_GOAL:     "where",
    ROLE_TIME:     "when",
    ROLE_MANNER:   "how",
}


# ---------------------------------------------------------------------------
# CaseStateTracker
# ---------------------------------------------------------------------------
class CaseStateTracker:
    """Tracks which case roles have been observed in a dialogue window.

    Inspired by the window_idxs logic in 02_make_dirunc_dataset.py.
    Stores per-turn role sets; exposes filled_roles(k) = union of last k turns.
    """

    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._history: List[Set[str]] = []  # indexed by turn index

    def push(self, roles: Set[str]) -> None:
        """Record the set of case roles surfaced in one turn."""
        self._history.append(set(roles))

    def filled_roles(self, k: Optional[int] = None) -> Set[str]:
        """Return union of case roles seen in the last k turns (default: window)."""
        k = k or self.window
        recent = self._history[-k:] if len(self._history) >= k else self._history
        result: Set[str] = set()
        for s in recent:
            result |= s
        return result

    def clear(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Pair balancing (reused from script 31)
# ---------------------------------------------------------------------------
def _balance_pairs(rows: List[dict], seed: int = 42, max_per_label: int = 5000) -> List[dict]:
    """Balance minimal pairs so each 5W1H direction has equal representation."""
    if not rows:
        return []
    rng = random.Random(seed)

    # Build pair dict
    pairs: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        pid = r["id"].rsplit("::", 1)[0]
        cond = r["condition"]
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][cond] = r

    valid_pairs: List[Tuple[dict, dict]] = [
        (d["filled"], d["missing"])
        for d in pairs.values()
        if "filled" in d and "missing" in d
    ]

    # Index pairs by the missing direction
    label_to_pair_idxs: Dict[str, List[int]] = {d: [] for d in DIRS}
    for i, (_, miss) in enumerate(valid_pairs):
        for d in DIRS:
            if miss["labels"].get(d, 0) == 1:
                label_to_pair_idxs[d].append(i)

    counts = {d: len(v) for d, v in label_to_pair_idxs.items()}
    non_zero = [c for c in counts.values() if c > 0]
    if not non_zero:
        return []
    min_count = min(min(non_zero), max_per_label)

    selected: Set[int] = set()
    for d in DIRS:
        idxs = label_to_pair_idxs[d]
        already = [i for i in idxs if i in selected]
        needed = min_count - len(already)
        if needed <= 0:
            continue
        candidates = [i for i in idxs if i not in selected]
        rng.shuffle(candidates)
        selected.update(candidates[:needed])

    out: List[dict] = []
    for i in sorted(selected):
        filled, missing = valid_pairs[i]
        out.append(filled)
        out.append(missing)
    return out


# ---------------------------------------------------------------------------
# Helper: build metadata dict
# ---------------------------------------------------------------------------
def _meta(
    predicate: str,
    theme_domain: str,
    case_role: str,
    slot: str,
    value: str,
    saturation_score: float,
    is_saturated: bool,
    filled_roles_before: List[str],
    **extra,
) -> dict:
    return {
        "predicate": predicate,
        "theme_domain": theme_domain,
        "case_role": case_role,
        "slot": slot,
        "dropped_span": value,
        "saturation_score": round(saturation_score, 4),
        "is_saturated": is_saturated,
        "filled_roles_before": sorted(filled_roles_before),
        **extra,
    }


# ---------------------------------------------------------------------------
# 1. SGD Processing
# ---------------------------------------------------------------------------
def process_sgd(limit_dialogues: int = 0) -> List[dict]:
    print("Processing SGD (Case Grammar) ...")
    root = Path("data/raw/sgd/train")
    files = sorted(root.glob("dialogues_*.json"))
    if not files:
        print("  Warning: no SGD dialogue files found."); return []

    req_map_path = Path("data/processed/sgd/required_slots_by_service_intent.json")
    if not req_map_path.exists():
        print("  Warning: required_slots map not found."); return []
    req_map = json.loads(req_map_path.read_text())

    meta_path = Path("data/processed/sgd/slot_meta_by_service_slot.json")
    slot_meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out_rows: List[dict] = []
    dlg_cnt = 0
    tracker = CaseStateTracker(window=5)

    for fp in files:
        if limit_dialogues and dlg_cnt >= limit_dialogues:
            break
        dialogues = json.loads(fp.read_text())
        for dlg in dialogues:
            if limit_dialogues and dlg_cnt >= limit_dialogues:
                break
            dlg_cnt += 1
            did = dlg.get("dialogue_id", "")
            turns = dlg["turns"]
            context_turns: List[str] = []
            tracker.clear()

            for t_idx, turn in enumerate(turns):
                is_user = turn.get("speaker", "").upper() == "USER"
                text = turn["utterance"]
                spk = "User: " if is_user else "Assistant: "

                if is_user:
                    # Identify active intent and frame
                    for fr in turn.get("frames", []):
                        svc   = fr.get("service", "")
                        state = fr.get("state", {})
                        intent = state.get("active_intent", "")

                        if not svc or not intent or intent == "NONE":
                            continue

                        frame = get_frame_for_intent(intent)
                        predicate   = frame.predicate   if frame else intent.lower()
                        theme_domain = frame.theme_domain if frame else svc.rsplit("_", 1)[0]

                        key = f"{svc}::{intent}"
                        if key not in req_map:
                            continue
                        req_slots = req_map[key].get("required_slots", [])
                        if not req_slots:
                            continue

                        sv = state.get("slot_values", {})

                        # What roles are already filled this turn (for tracker)
                        turn_roles: Set[str] = set()
                        for sl, vals in sv.items():
                            if vals:
                                role = slot_to_role(sl, svc)
                                turn_roles.add(role)

                        # Roles filled BEFORE this turn (history window)
                        filled_before = tracker.filled_roles()

                        for sl in req_slots:
                            vals = sv.get(sl, [])
                            if not vals:
                                continue
                            val_str = str(vals[0]) if isinstance(vals, list) else str(vals)
                            if not val_str:
                                continue

                            # Case role of this slot
                            case_role = slot_to_role(sl, svc)
                            dir_label = ROLE_TO_DIR.get(case_role, "what")
                            mdesc = slot_meta.get(f"{svc}::{sl}", {}).get("description", "")
                            dir_fallback = map_slot_to_dir(sl, mdesc)
                            # Prefer case-role mapping; fall back to keyword heuristic
                            target_dir = dir_label if dir_label != "what" else dir_fallback

                            # Compute saturation: roles filled before this slot drop
                            sat_roles = filled_before | (turn_roles - {case_role})
                            if frame:
                                sat_score, is_sat = frame.saturation(sat_roles)
                            else:
                                # No frame: simple ratio over required slots
                                n_filled = len([s for s in req_slots if sv.get(s)])
                                n_total  = len(req_slots)
                                sat_score = (n_filled - 1) / n_total if n_total else 0.0
                                is_sat    = False

                            # Build context prefix
                            dom_str = f"[Domain: {svc} / Intent: {intent}]\n"
                            hist    = context_turns[-3:]
                            ctx     = dom_str + "\n".join(hist) + "\n" if hist else dom_str

                            resolved_text   = ctx + spk + text
                            ablat_text      = replace_values_in_text(text, [val_str], mode="delete")
                            unresolved_text = ctx + spk + ablat_text

                            if resolved_text == unresolved_text:
                                continue

                            pid = f"sgd::{did}::t{t_idx}::{sl}"
                            meta_base = _meta(
                                predicate, theme_domain, case_role, sl, val_str,
                                sat_score, is_sat, sorted(filled_before),
                                service=svc, intent=intent,
                            )

                            out_rows.append({
                                "id":        pid + "::filled",
                                "text":      resolved_text,
                                "labels":    build_label_dict([]),
                                "condition": "filled",
                                "dataset":   "sgd",
                                "metadata":  {**meta_base, "original_span": val_str},
                            })
                            out_rows.append({
                                "id":        pid + "::missing",
                                "text":      unresolved_text,
                                "labels":    build_label_dict([target_dir]),
                                "condition": "missing",
                                "dataset":   "sgd",
                                "metadata":  meta_base,
                            })

                    # Update tracker with all roles mentioned in this user turn
                    all_turn_roles: Set[str] = set()
                    for fr in turn.get("frames", []):
                        sv = fr.get("state", {}).get("slot_values", {})
                        svc = fr.get("service", "")
                        for sl, vals in sv.items():
                            if vals:
                                all_turn_roles.add(slot_to_role(sl, svc))
                    tracker.push(all_turn_roles)

                else:
                    tracker.push(set())  # system turn: no new user-side roles

                context_turns.append(spk + text)

    print(f"  SGD: {len(out_rows)} rows before balance.")
    return _balance_pairs(out_rows)


# ---------------------------------------------------------------------------
# 2. MultiWOZ Processing
# ---------------------------------------------------------------------------
_MWOZ_SLOT_TO_DIR: Dict[str, str] = {
    "time": "when", "day": "when", "leaveat": "when", "arriveby": "when",
    "destination": "where", "departure": "where", "area": "where",
    "people": "how",  "price": "how", "internet": "how",
    "parking": "how", "stay": "how",  "pricerange": "how",
    "food": "what",   "type": "what", "name": "what",
    "department": "what", "stars": "what",
}


def process_multiwoz(data_path: str = "data/raw/multiwoz/data.json",
                     limit_dialogues: int = 0) -> List[dict]:
    print("Processing MultiWOZ (Case Grammar) ...")
    p = Path(data_path)
    if not p.exists():
        print(f"  Warning: {data_path} not found."); return []

    data = json.loads(p.read_text())
    out_rows: List[dict] = []
    dlg_cnt = 0
    tracker = CaseStateTracker(window=5)

    for file_id, dialog in data.items():
        if limit_dialogues and dlg_cnt >= limit_dialogues:
            break
        dlg_cnt += 1
        log = dialog.get("log", [])
        context_turns: List[str] = []
        tracker.clear()
        prev_meta: dict = {}

        for turn_idx, turn in enumerate(log):
            text    = turn["text"].strip()
            is_user = (turn_idx % 2 == 0)
            spk     = "User: " if is_user else "Assistant: "

            if is_user:
                spans    = turn.get("span_info", [])
                # MultiWOZ user turns lack metadata; it is stored in the subsequent system turn.
                metadata = log[turn_idx + 1].get("metadata", {}) if turn_idx + 1 < len(log) else {}

                # Fallback: if span_info is empty, reconstruct from metadata diff
                if not spans:
                    for domain, dinfo in metadata.items():
                        semi = dinfo.get("semi", {})
                        prev_semi = prev_meta.get(domain, {}).get("semi", {})
                        for sl, val in semi.items():
                            if val and val not in ("", "not mentioned", "none") and val != prev_semi.get(sl):
                                # Try to find val in text
                                val_str = str(val)
                                if val_str.lower() in text.lower():
                                    # Create pseudo-span: [act, slot, val, start_char, end_char]
                                    # Note: 33_create_case_grammar_pairs.py uses word-level indices for start/end
                                    words = text.split()
                                    val_words = val_str.split()
                                    # Find contiguous val_words in words
                                    for i in range(len(words) - len(val_words) + 1):
                                        if " ".join(words[i:i+len(val_words)]).lower() == val_str.lower():
                                            spans.append([f"{domain}-Inform", sl, val_str, i, i + len(val_words) - 1])
                                            break
                        
                        book = dinfo.get("book", {})
                        prev_book = prev_meta.get(domain, {}).get("book", {})
                        for sl, val in book.items():
                            if sl != "booked" and val and val not in ("", "none") and val != prev_book.get(sl):
                                val_str = str(val)
                                if val_str.lower() in text.lower():
                                    words = text.split()
                                    val_words = val_str.split()
                                    for i in range(len(words) - len(val_words) + 1):
                                        if " ".join(words[i:i+len(val_words)]).lower() == val_str.lower():
                                            spans.append([f"{domain}-Inform", sl, val_str, i, i + len(val_words) - 1])
                                            break

                active_domains = [
                    dom for dom, dinfo in metadata.items()
                    if any(v for v in dinfo.get("semi", {}).values()
                           if v not in ("", "not mentioned", "none"))
                ]
                dom_str = f"[Domain: {', '.join(active_domains)}]\n" if active_domains else ""

                turn_roles: Set[str] = set()
                filled_before = tracker.filled_roles()

                for span in spans:
                    if len(span) < 5:
                        continue
                    act_type, slot, value, start, end = span[:5]
                    # act_type for pseudo-spans is domain-Inform
                    if not act_type.lower().endswith("-inform") or act_type.lower() == "general-inform":
                        continue

                    domain = act_type.split("-")[0].lower()
                    frame  = get_frame_for_multiwoz_domain(domain)
                    predicate    = frame.predicate    if frame else "find"
                    theme_domain = frame.theme_domain if frame else domain.capitalize()

                    sl_lower     = slot.lower()
                    case_role    = slot_to_role(sl_lower)
                    target_dir   = _MWOZ_SLOT_TO_DIR.get(sl_lower,
                                   ROLE_TO_DIR.get(case_role, "what"))

                    words = text.split()
                    if not (start < len(words) and end < len(words) and start <= end):
                        continue

                    dropped_span = " ".join(words[start:end + 1])
                    ablat_words  = words[:start] + words[end + 1:]
                    # Ensure at least a period if empty
                    ablat_text   = " ".join(ablat_words) if ablat_words else "."

                    hist = context_turns[-3:]
                    ctx  = dom_str + "\n".join(hist) + "\n" if hist else dom_str

                    resolved_text   = ctx + spk + text
                    unresolved_text = ctx + spk + ablat_text

                    if resolved_text == unresolved_text:
                        continue

                    sat_roles = filled_before | (turn_roles - {case_role})
                    if frame:
                        sat_score, is_sat = frame.saturation(sat_roles)
                    else:
                        sat_score, is_sat = 0.0, False

                    turn_roles.add(case_role)

                    pid = f"multiwoz::{file_id}::t{turn_idx}::{domain}-{slot}"
                    meta_base = _meta(
                        predicate, theme_domain, case_role, slot, dropped_span,
                        sat_score, is_sat, sorted(filled_before),
                        domain=domain, original_span=dropped_span,
                    )

                    out_rows.append({
                        "id":        pid + "::filled",
                        "text":      resolved_text,
                        "labels":    build_label_dict([]),
                        "condition": "filled",
                        "dataset":   "multiwoz",
                        "metadata":  meta_base,
                    })
                    out_rows.append({
                        "id":        pid + "::missing",
                        "text":      unresolved_text,
                        "labels":    build_label_dict([target_dir]),
                        "condition": "missing",
                        "dataset":   "multiwoz",
                        "metadata":  meta_base,
                    })

                tracker.push(turn_roles)
            else:
                tracker.push(set())
                prev_meta = turn.get("metadata", {})

            context_turns.append(spk + text)


    print(f"  MultiWOZ: {len(out_rows)} rows before balance.")
    return _balance_pairs(out_rows)


# ---------------------------------------------------------------------------
# 3. QA-SRL Processing (unchanged — no intent/frame; use wh-word directly)
# ---------------------------------------------------------------------------
def _map_qasrl_wh(wh: str) -> str:
    wh = wh.lower().strip()
    return wh if wh in DIRS else "what"


def _qasrl_wh_to_role(wh: str) -> str:
    mapping = {
        "who": ROLE_AGENT, "what": ROLE_THEME, "where": ROLE_LOCATION,
        "when": ROLE_TIME,  "why":  ROLE_MANNER, "how":  ROLE_MANNER,
        "which": ROLE_THEME,
    }
    return mapping.get(wh.lower(), ROLE_THEME)


def process_qasrl(data_path: str = "temp_qasrl/qasrl-bank/data/qasrl-v2/orig/dev.jsonl.gz",
                  limit: int = 0) -> List[dict]:
    print(f"Processing QA-SRL ({data_path}) ...")
    if not Path(data_path).exists():
        print(f"  Warning: {data_path} not found."); return []

    out_rows: List[dict] = []
    cnt = 0
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if limit and cnt >= limit:
                break
            cnt += 1
            doc      = json.loads(line)
            sid      = doc.get("sentenceId", f"sent_{cnt}")
            tokens   = doc.get("sentenceTokens", [])
            verbData = doc.get("verbEntries", {})

            for vb_id, vb_info in verbData.items():
                target_verb = vb_info.get("verbInflectedForms", {}).get("stem", "do")
                qLabels     = vb_info.get("questionLabels", {})

                for pq, qinfo in qLabels.items():
                    wh         = qinfo.get("questionSlots", {}).get("wh", "what")
                    target_dir = _map_qasrl_wh(wh)
                    case_role  = _qasrl_wh_to_role(wh)

                    judgments   = qinfo.get("answerJudgments", [])
                    valid_spans = []
                    for j in judgments:
                        if j.get("isValid") and j.get("spans"):
                            valid_spans.extend(j["spans"])

                    if not valid_spans:
                        continue

                    s_start, s_end = valid_spans[0]
                    if not (s_start < len(tokens) and s_end <= len(tokens)):
                        continue

                    dropped_span    = " ".join(tokens[s_start:s_end])
                    ablat_tokens    = tokens[:s_start] + tokens[s_end:]
                    resolved_text   = f"[Target Verb: {target_verb}] " + " ".join(tokens)
                    unresolved_text = cleanup_deletion_artifacts(
                        f"[Target Verb: {target_verb}] " + " ".join(ablat_tokens)
                    )

                    if resolved_text == unresolved_text:
                        continue

                    pid = f"qasrl::{sid}::{target_verb}::{wh}"
                    meta_base = _meta(
                        target_verb, "QA-SRL", case_role, wh, dropped_span,
                        0.0, False, [],   # no frame for QA-SRL
                        verb=target_verb, question=pq,
                    )

                    out_rows.append({
                        "id":        pid + "::filled",
                        "text":      resolved_text,
                        "labels":    build_label_dict([]),
                        "condition": "filled",
                        "dataset":   "qasrl",
                        "metadata":  meta_base,
                    })
                    out_rows.append({
                        "id":        pid + "::missing",
                        "text":      unresolved_text,
                        "labels":    build_label_dict([target_dir]),
                        "condition": "missing",
                        "dataset":   "qasrl",
                        "metadata":  meta_base,
                    })

    print(f"  QA-SRL: {len(out_rows)} rows before balance.")
    return _balance_pairs(out_rows)


# ---------------------------------------------------------------------------
# Saturation distribution reporter
# ---------------------------------------------------------------------------
def _report_saturation(rows: List[dict]) -> None:
    missing = [r for r in rows if r["condition"] == "missing"]
    scores  = [r["metadata"].get("saturation_score", 0.0) for r in missing]
    is_sat  = [r["metadata"].get("is_saturated", False)   for r in missing]
    if not scores:
        return
    bins = Counter()
    for s in scores:
        b = int(s * 5) / 5  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        bins[round(b, 1)] += 1
    print("  Saturation distribution (missing rows):")
    for k in sorted(bins):
        bar = "#" * (bins[k] // max(1, max(bins.values()) // 20))
        print(f"    [{k:.1f}] {bins[k]:5d}  {bar}")
    print(f"  is_saturated=True: {sum(is_sat)} / {len(is_sat)}")

    role_counts = Counter(r["metadata"].get("case_role", "?") for r in missing)
    print("  Case role distribution (missing rows):")
    for role, cnt in role_counts.most_common():
        print(f"    {role:12s} {cnt:5d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root_out = Path("data/processed/case_grammar/")
    root_out.mkdir(parents=True, exist_ok=True)

    # --- SGD ---
    sgd_rows = process_sgd()
    if sgd_rows:
        out_path = root_out / "cg_v1_sgd.jsonl"
        write_jsonl(out_path, sgd_rows)
        print(f"Saved {len(sgd_rows)} SGD rows -> {out_path}")
        _report_saturation(sgd_rows)

    # --- MultiWOZ ---
    mwoz_rows = process_multiwoz()
    if mwoz_rows:
        out_path = root_out / "cg_v1_multiwoz.jsonl"
        write_jsonl(out_path, mwoz_rows)
        print(f"Saved {len(mwoz_rows)} MultiWOZ rows -> {out_path}")
        _report_saturation(mwoz_rows)

    # --- QA-SRL ---
    qasrl_rows = process_qasrl()
    if qasrl_rows:
        out_path = root_out / "cg_v1_qasrl.jsonl"
        write_jsonl(out_path, qasrl_rows)
        print(f"Saved {len(qasrl_rows)} QA-SRL rows -> {out_path}")
        _report_saturation(qasrl_rows)

    print("\nDone. Case Grammar datasets saved to", root_out)


if __name__ == "__main__":
    main()
