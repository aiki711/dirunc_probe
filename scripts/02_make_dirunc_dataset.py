# scripts/02_make_dirunc_dataset.py
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from common import DIRS, QUERY_TOKENS_STR as QUERY_TOKENS, map_slot_to_dir, PLACEHOLDER_BY_DIR, normalize_text, replace_values_in_text, cleanup_deletion_artifacts

SPLITS_DEFAULT = ["train", "dev", "test", "test_seen", "test_unseen"]

# ---------- Utilities ----------

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def clip(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def is_user_turn(turn: Dict[str, Any]) -> bool:
    # SGD usually uses 'USER' for user and 'SYSTEM' for assistant
    return turn.get("speaker", "").upper() == "USER"

# Removed local text utilities (moved to common.py)

# ---------- SGD extraction ----------

def find_dialogue_files(split_dir: Path) -> List[Path]:
    # In SGD repo: dialogues_*.json or dialogue_*.json depending on packaging
    files = sorted(split_dir.glob("dialogues_*.json"))
    if not files:
        files = sorted(split_dir.glob("dialogue_*.json"))
    return files

def extract_active_intents(frame: Dict[str, Any]) -> Optional[str]:
    st = frame.get("state")
    if not st:
        return None
    ai = st.get("active_intent")
    if not ai or ai == "NONE":
        return None
    return ai

def extract_slot_values(frame: Dict[str, Any]) -> Dict[str, List[str]]:
    st = frame.get("state") or {}
    sv = st.get("slot_values") or {}
    # ensure list[str]
    out: Dict[str, List[str]] = {}
    for k, v in sv.items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        else:
            out[k] = [str(v)]
    return out

def collect_service_slots_in_turn(turn: Dict[str, Any], service: str) -> Dict[str, List[str]]:
    vals: Dict[str, List[str]] = {}
    for fr in turn.get("frames", []):
        if fr.get("service") != service:
            continue
        sv = extract_slot_values(fr)
        for k, vs in sv.items():
            vals.setdefault(k, [])
            vals[k].extend(vs)
    # de-dup values
    for k in list(vals.keys()):
        vals[k] = list(dict.fromkeys(vals[k]))
    return vals

def collect_service_slots_in_window(
    dialogue: Dict[str, Any],
    turn_indices: List[int],
    service: str,
    include_assistant: bool,
) -> Dict[str, List[str]]:
    vals: Dict[str, List[str]] = {}
    for i in turn_indices:
        t = dialogue["turns"][i]
        if (not include_assistant) and (not is_user_turn(t)):
            continue
        sv = collect_service_slots_in_turn(t, service)
        for k, vs in sv.items():
            vals.setdefault(k, [])
            vals[k].extend(vs)
    for k in list(vals.keys()):
        vals[k] = list(dict.fromkeys(vals[k]))
    return vals

def get_prev_user_turn_index(dialogue: Dict[str, Any], t_idx: int) -> Optional[int]:
    for j in range(t_idx - 1, -1, -1):
        if is_user_turn(dialogue["turns"][j]):
            return j
    return None

def get_last_k_user_turn_indices(dialogue: Dict[str, Any], t_idx: int, k: int) -> List[int]:
    """Return indices of last k user turns ending at t_idx (which must be a user turn)."""
    user_idxs = []
    for j in range(t_idx, -1, -1):
        if is_user_turn(dialogue["turns"][j]):
            user_idxs.append(j)
            if len(user_idxs) == k:
                break
    return list(reversed(user_idxs))

def get_window_turn_indices(dialogue: Dict[str, Any], user_turn_indices: List[int]) -> List[int]:
    """Include all turns between first and last user turn indices."""
    if not user_turn_indices:
        return []
    a = user_turn_indices[0]
    b = user_turn_indices[-1]
    return list(range(a, b + 1))

def render_context_text(dialogue: Dict[str, Any], window_turn_indices: List[int], phase: int) -> str:
    """Phase1: user-only (two user utterances). Phase2: user+assistant with prefixes."""
    turns = dialogue["turns"]
    if phase == 1:
        parts = []
        for i in window_turn_indices:
            if is_user_turn(turns[i]):
                parts.append(turns[i]["utterance"])
        return normalize_text("\n".join(parts))
    else:
        parts = []
        for i in window_turn_indices:
            spk = "User" if is_user_turn(turns[i]) else "Assistant"
            parts.append(f"{spk}: {turns[i]['utterance']}")
        return normalize_text("\n".join(parts))

# ---------- Example generation ----------

@dataclass
class GenConfig:
    phases: List[int]
    levels: List[int]
    k_values: List[int]  # for phase2
    require_complete_before_drop: bool
    seed: int
    max_dialogues_per_split: Optional[int]
    include_splits: List[str]
    debug_dir_stats: bool
    debug_topk: int
    contrastive: bool = False

def generate_examples_for_turn(
    *,
    dialogue: Dict[str, Any],
    split: str,
    dialogue_id: str,
    turn_idx: int,
    service: str,
    intent: str,
    required_slots: List[str],
    slot_meta: Dict[str, Dict[str, Any]],
    cfg: GenConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic missing-slot examples.
    If cfg.contrastive is True:
        Generate (Resolved, Unresolved) pairs based on context manipulation.
        Target turn is always perturbed (Level 1 typically).
    Else:
        Legacy generation based on phases/levels/k_values.
    """
    out_rows: List[Dict[str, Any]] = []
    
    # Pre-calculate simple user windows for potential use
    prev_u = get_prev_user_turn_index(dialogue, turn_idx)
    phase1_user_idxs = [i for i in [prev_u, turn_idx] if i is not None]
    phase2_user_idxs_by_k = {k: get_last_k_user_turn_indices(dialogue, turn_idx, k) for k in cfg.k_values}

    if cfg.contrastive:
        # Contrastive Mode (Force Phase 2, Level 1 logic mostly)
        # Use simple strategy: Focus on Phase 2 context with max k
        
        target_k = max(cfg.k_values) if cfg.k_values else 5
        user_idxs = get_last_k_user_turn_indices(dialogue, turn_idx, target_k)
        window_idxs = get_window_turn_indices(dialogue, user_idxs)

        # Collect slots in full window (assistant included)
        slot_values = collect_service_slots_in_window(
            dialogue, window_idxs, service, include_assistant=True
        )
        observed_slots = set(slot_values.keys())
        required_set = set(required_slots)

        # Candidates must be required AND observed derived from context
        candidates = sorted(list(required_set.intersection(observed_slots)))
        if not candidates:
            return []

        for sl in candidates:
            vals = slot_values.get(sl, [])
            meta = slot_meta.get(f"{service}::{sl}", {})
            desc = str(meta.get("description", "") or "")
            d = map_slot_to_dir(sl, desc)
            placeholder = PLACEHOLDER_BY_DIR.get(d, "something")

            # -- Target Turn Perturbation (Level 1) --
            target_text_raw = dialogue["turns"][turn_idx]["utterance"]
            target_text_mod = replace_values_in_text(target_text_raw, vals, mode="placeholder", placeholder=placeholder)
            target_text_final = normalize_text(target_text_mod) + QUERY_TOKENS
            
            # -- Context Generation --
            context_idxs = window_idxs[:-1] # exclude target
            if not context_idxs:
                continue

            base_ctx_text = render_context_text(dialogue, context_idxs, phase=2)
            
            # Resolved (Positive): Keep values
            ctx_resolved = cleanup_deletion_artifacts(base_ctx_text)

            # Unresolved (Negative): Delete values
            ctx_unresolved = replace_values_in_text(base_ctx_text, vals, mode="delete")

            if ctx_resolved == ctx_unresolved:
                continue

            pair_id = f"{dialogue_id}::t{turn_idx}::{sl}"

            # Labels logic
            def make_labels(is_unresolved_tgt: bool):
                lbls = {}
                for req_sl in required_slots:
                    req_d = map_slot_to_dir(req_sl, "")
                    is_missing = False
                    if req_sl not in observed_slots:
                        is_missing = True # originally missing
                    else:
                        # If Unresolved Case and this is the target slot, force missing
                        if is_unresolved_tgt and req_sl == sl:
                            is_missing = True
                    
                    # Accumulate to direction (OR logic)
                    prev = lbls.get(req_d, 0)
                    lbls[req_d] = 1 if (is_missing or prev==1) else 0
                
                # Fill rest
                for dd in DIRS:
                    if dd not in lbls:
                        lbls[dd] = 0
                return lbls

            labels_res = make_labels(is_unresolved_tgt=False)
            labels_unr = make_labels(is_unresolved_tgt=True)

            common = {
                "split": split,
                "dialogue_id": dialogue_id,
                "turn_idx": turn_idx,
                "service": service,
                "intent": intent,
                "phase": 2,
                "level": 1, 
                "k": target_k,
                "target_slot": sl,
                "target_dir": d,
                "pair_id": pair_id,
                "required_slots": sorted(list(required_set)),
            }
            
            out_rows.append({
                "id": pair_id + "::resolved",
                "text": ctx_resolved + "\n" + target_text_final,
                "labels": labels_res,
                "condition": "resolved",
                **common
            })
            out_rows.append({
                "id": pair_id + "::unresolved",
                "text": ctx_unresolved + "\n" + target_text_final,
                "labels": labels_unr,
                "condition": "unresolved",
                **common
            })

    # Non-contrastive (Legacy) logic
    else:
        for phase in cfg.phases:
            for level in cfg.levels:
                if phase == 1:
                    user_idxs = phase1_user_idxs
                    window_idxs = get_window_turn_indices(dialogue, user_idxs)
                    include_asst_for_slots = False
                    drop_sizes = [1]
                    k_val = None
                    out_rows.extend(_gen_rows_one_setting(dialogue=dialogue, split=split, dialogue_id=dialogue_id,
                                    turn_idx=turn_idx, service=service, intent=intent, required_slots=required_slots,
                                    slot_meta=slot_meta, phase=phase, level=level, k_val=k_val, window_idxs=window_idxs,
                                    include_asst_for_slots=include_asst_for_slots, drop_sizes=drop_sizes, cfg=cfg, rng=rng))
                else:
                    for k_val in cfg.k_values:
                        user_idxs = phase2_user_idxs_by_k[k_val]
                        window_idxs = get_window_turn_indices(dialogue, user_idxs)
                        include_asst_for_slots = True
                        drop_sizes = [1, 2]
                        out_rows.extend(_gen_rows_one_setting(dialogue=dialogue, split=split, dialogue_id=dialogue_id,
                                    turn_idx=turn_idx, service=service, intent=intent, required_slots=required_slots,
                                    slot_meta=slot_meta, phase=phase, level=level, k_val=k_val, window_idxs=window_idxs,
                                    include_asst_for_slots=include_asst_for_slots, drop_sizes=drop_sizes, cfg=cfg, rng=rng))

    return out_rows

def _gen_rows_one_setting(
    *,
    dialogue: Dict[str, Any],
    split: str,
    dialogue_id: str,
    turn_idx: int,
    service: str,
    intent: str,
    required_slots: List[str],
    slot_meta: Dict[str, Dict[str, Any]],
    phase: int,
    level: int,
    k_val: Optional[int],
    window_idxs: List[int],
    include_asst_for_slots: bool,
    drop_sizes: List[int],
    cfg: GenConfig,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    # full context text
    base_text = render_context_text(dialogue, window_idxs, phase=phase)

    # slot values observed in the window (service-specific)
    slot_values = collect_service_slots_in_window(
        dialogue, window_idxs, service, include_assistant=include_asst_for_slots
    )
    observed_slots = set(slot_values.keys())
    required_set = set(required_slots)

    # If require "complete before drop", ensure all required are present
    if cfg.require_complete_before_drop and not required_set.issubset(observed_slots):
        return []

    # Candidate slots we can drop: only those required and present
    droppable = sorted(list(required_set.intersection(observed_slots)))
    if not droppable:
        return []

    rows: List[Dict[str, Any]] = []
    for drop_n in drop_sizes:
        if len(droppable) < drop_n:
            continue

        drop_slots = rng.sample(droppable, drop_n)
        perturbed_text = base_text
        missing_dirs = set()

        for sl in drop_slots:
            vals = slot_values.get(sl, [])
            meta = slot_meta.get(f"{service}::{sl}", {})
            desc = str(meta.get("description", "") or "")
            d = map_slot_to_dir(sl, desc)
            missing_dirs.add(d)

            if level == 0:
                perturbed_text = replace_values_in_text(perturbed_text, vals, mode="delete")
            elif level == 1:
                ph = PLACEHOLDER_BY_DIR.get(d, "something") # Safe get
                perturbed_text = replace_values_in_text(perturbed_text, vals, mode="placeholder", placeholder=ph)
            else:
                raise ValueError(f"Unsupported level: {level}")

        perturbed_text = normalize_text(perturbed_text) + QUERY_TOKENS

        labels = {d: (1 if d in missing_dirs else 0) for d in DIRS}
        uid = f"{split}::{dialogue_id}::t{turn_idx}::{service}::{intent}::p{phase}::l{level}::k{k_val or 0}::drop{drop_n}"

        rows.append(
            {
                "id": uid,
                "split": split,
                "dialogue_id": dialogue_id,
                "turn_idx": turn_idx,
                "service": service,
                "intent": intent,
                "phase": phase,
                "level": level,
                "k": k_val,
                "text": perturbed_text,
                "labels": labels,          # dict form
                "label_order": DIRS,       # fixed order
                "missing_slots": drop_slots,
                "missing_dirs": sorted(list(missing_dirs)),
                "required_slots": sorted(list(required_set)),
                "observed_slots_full": sorted(list(observed_slots)),  # before drop
            }
        )

    return rows

# ---------- Debug helpers ----------

def slot_dir(service: str, slot: str, slot_meta: Dict[str, Dict[str, Any]]) -> str:
    meta = slot_meta.get(f"{service}::{slot}", {})
    desc = str(meta.get("description", "") or "")
    return map_slot_to_dir(slot, desc)

def print_dir_distribution(title: str, ctr: Counter, denom: Optional[int] = None) -> None:
    denom = denom if denom is not None else (sum(ctr.values()) or 1)
    print(title)
    for d in DIRS:
        c = int(ctr.get(d, 0))
        pct = c / denom * 100.0
        print(f"  {d:>5}: {c:>10}  ({pct:6.2f}%)")

def print_top(title: str, ctr: Counter, topk: int) -> None:
    print(title)
    for k, c in ctr.most_common(topk):
        print(f"  {str(k):>25} : {c}")

# ---------- Main pipeline ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgd_root", type=str, default="data/raw/sgd")
    ap.add_argument("--required_map", type=str, default="data/processed/sgd/required_slots_by_service_intent.json")
    ap.add_argument("--slot_meta", type=str, default="data/processed/sgd/slot_meta_by_service_slot.json")
    ap.add_argument("--out_dir", type=str, default="data/processed/sgd/dirunc")
    ap.add_argument("--splits", type=str, default=",".join(SPLITS_DEFAULT))
    ap.add_argument("--phases", type=str, default="1,2")
    ap.add_argument("--levels", type=str, default="0,1")
    ap.add_argument("--k_values", type=str, default="3,5")  # Phase2
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_dialogues_per_split", type=int, default=0, help="0 means no limit")
    ap.add_argument(
        "--allow_incomplete_context",
        action="store_true",
        help="If set, also generate drops even when required slots are not fully present originally (not recommended for v0)."
    )
    ap.add_argument("--debug_dir_stats", action="store_true", help="Enable debug prints for dir distributions.")
    ap.add_argument("--debug_topk", type=int, default=30, help="TopK slots to print in debug.")
    ap.add_argument("--contrastive", action="store_true", help="Enable contrastive pair generation (Level1-based).")
    args = ap.parse_args()

    sgd_root = Path(args.sgd_root)
    out_dir = Path(args.out_dir)

    required_map: Dict[str, Dict[str, List[str]]] = read_json(Path(args.required_map))
    slot_meta: Dict[str, Dict[str, Any]] = read_json(Path(args.slot_meta))

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    max_d = args.max_dialogues_per_split if args.max_dialogues_per_split > 0 else None

    cfg = GenConfig(
        phases=phases,
        levels=levels,
        k_values=k_values,
        require_complete_before_drop=(not args.allow_incomplete_context),
        seed=args.seed,
        max_dialogues_per_split=max_d,
        include_splits=splits,
        debug_dir_stats=args.debug_dir_stats,
        debug_topk=args.debug_topk,
        contrastive=args.contrastive,
    )
    rng = random.Random(cfg.seed)

    for split in cfg.include_splits:
        split_dir = sgd_root / split
        if not split_dir.exists():
            print(f"[skip] split not found: {split_dir}")
            continue

        files = find_dialogue_files(split_dir)
        if not files:
            print(f"[skip] no dialogue files in {split_dir}")
            continue

        rows_all: List[Dict[str, Any]] = []
        n_dialogues = 0

        # ---- debug counters (per split) ----
        req_dir_ctr = Counter()
        drop_dir_ctr = Counter()
        req_slot_ctr = Counter()
        drop_slot_ctr = Counter()

        # Optional: track phase/level distributions of dropped dirs
        drop_dir_by_setting: Dict[str, Counter] = {}

        for fp in files:
            dialogues = read_json(fp)
            for d in dialogues:
                n_dialogues += 1
                if cfg.max_dialogues_per_split and n_dialogues > cfg.max_dialogues_per_split:
                    break

                dialogue_id = str(d.get("dialogue_id", f"{fp.stem}::{n_dialogues}"))
                turns = d["turns"]

                for t_idx, turn in enumerate(turns):
                    # We anchor examples on USER turns (current user turn)
                    if not is_user_turn(turn):
                        continue

                    # For each frame with active_intent, generate one example per (service,intent)
                    for fr in turn.get("frames", []):
                        service = fr.get("service")
                        intent = extract_active_intents(fr)
                        if not service or not intent:
                            continue

                        key = f"{service}::{intent}"
                        if key not in required_map:
                            continue
                        required_slots = required_map[key]["required_slots"]
                        if not required_slots:
                            continue

                        # ---- debug: required distribution (upper stream) ----
                        if cfg.debug_dir_stats:
                            for sl in required_slots:
                                req_slot_ctr[sl] += 1
                                ddir = slot_dir(service, sl, slot_meta)
                                req_dir_ctr[ddir] += 1

                        rows = generate_examples_for_turn(
                            dialogue=d,
                            split=split,
                            dialogue_id=dialogue_id,
                            turn_idx=t_idx,
                            service=service,
                            intent=intent,
                            required_slots=required_slots,
                            slot_meta=slot_meta,
                            cfg=cfg,
                            rng=rng,
                        )

                        # ---- debug: dropped distribution (down stream) ----
                        if cfg.debug_dir_stats:
                            for rr in rows:
                                for sl in rr.get("missing_slots", []):
                                    drop_slot_ctr[sl] += 1
                                # dirs
                                for ddir in rr.get("missing_dirs", []):
                                    drop_dir_ctr[ddir] += 1
                                # by setting
                                st_key = f"p{rr.get('phase')}/l{rr.get('level')}/k{rr.get('k') or 0}"
                                if st_key not in drop_dir_by_setting:
                                    drop_dir_by_setting[st_key] = Counter()
                                for ddir in rr.get("missing_dirs", []):
                                    drop_dir_by_setting[st_key][ddir] += 1

                        rows_all.extend(rows)

            if cfg.max_dialogues_per_split and n_dialogues >= cfg.max_dialogues_per_split:
                break

        out_path = out_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows_all)

        print(f"[done] split={split} dialogues={n_dialogues} examples={len(rows_all)} -> {out_path}")

        # ---- print debug summary ----
        if cfg.debug_dir_stats:
            print("\n========== DEBUG DIR STATS ==========")
            print(f"split={split}  dialogues={n_dialogues}  examples={len(rows_all)}")
            print_dir_distribution("[debug] required slot dir distribution (upper stream):", req_dir_ctr)
            print_dir_distribution("[debug] dropped slot dir distribution (down stream):", drop_dir_ctr)

            print("\n[debug] dropped dir distribution by (phase/level/k):")
            for st_key in sorted(drop_dir_by_setting.keys()):
                ctr = drop_dir_by_setting[st_key]
                denom = sum(ctr.values()) or 1
                print_dir_distribution(f"  - {st_key}:", ctr, denom=denom)

            print()
            print_top("[debug] top required_slots (by frequency in required_map hits):", req_slot_ctr, cfg.debug_topk)
            print()
            print_top("[debug] top dropped_slots (actually dropped):", drop_slot_ctr, cfg.debug_topk)
            print("=====================================\n")

if __name__ == "__main__":
    main()
