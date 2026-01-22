# scripts/04_stats_dirunc.py
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DIRS = ["who", "what", "when", "where", "why", "how"]

# NOTE:
# This mapping must be kept consistent with scripts/02_make_dirunc_dataset.py
WHO_KWS = [
    "attendee", "contact", "recipient", "person", "customer", "client",
    "doctor", "trainer", "agent", "name of person",
]
WHEN_KWS = [
    "date", "time", "day", "week", "month", "year", "duration",
    "start time", "end time", "arrive", "leave", "check in", "check out",
]
WHERE_KWS = [
    "location", "address", "city", "place", "destination", "origin",
    "airport", "station", "area", "neighborhood", "venue",
]
WHY_KWS = ["reason", "purpose", "because"]
HOW_KWS = ["method", "mode", "transport", "payment", "delivery", "format", "via"]
HOW_MANY_KWS = ["number", "count", "amount", "quantity", "seats", "riders", "guests", "party_size", "people"]
HOW_MODE_KWS = ["shared", "private", "mode", "option"]

def map_slot_to_dir(slot_name: str, slot_desc: str) -> str:
    s = (slot_name + " " + slot_desc).lower()
    if any(k in s for k in WHO_KWS):
        return "who"
    if any(k in s for k in WHEN_KWS):
        return "when"
    if any(k in s for k in WHERE_KWS):
        return "where"
    if any(k in s for k in WHY_KWS):
        return "why"
    if any(k in s for k in HOW_KWS):
        return "how"
    if any(k in s for k in HOW_MANY_KWS):
        return "how"
    if any(k in s for k in HOW_MODE_KWS):
        return "how"
    return "what"

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def short_text(s: str, n: int = 180) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= n:
        return s
    return s[: n - 3] + "..."

def safe_get_slot_desc(slot_meta: Dict[str, Dict[str, Any]], service: str, slot: str) -> str:
    meta = slot_meta.get(f"{service}::{slot}", {})
    return str(meta.get("description", "") or "")

@dataclass
class SlotAuditRow:
    split: str
    service: str
    intent: str
    slot: str
    mapped_dir: str
    required_count: int
    dropped_count: int
    slot_desc: str
    example_1: str
    example_2: str

def write_slot_audit_csv(path: Path, rows: List[SlotAuditRow]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "split", "service", "intent", "slot", "mapped_dir",
            "required_count", "dropped_count", "slot_desc",
            "example_1", "example_2",
        ])
        for r in rows:
            w.writerow([
                r.split, r.service, r.intent, r.slot, r.mapped_dir,
                r.required_count, r.dropped_count, r.slot_desc,
                r.example_1, r.example_2,
            ])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/sgd/dirunc")
    ap.add_argument("--slot_meta", type=str, default="data/processed/sgd/slot_meta_by_service_slot.json")
    ap.add_argument("--splits", type=str, default="train,dev")
    ap.add_argument("--out_dir", type=str, default="runs/dirunc_stats")
    ap.add_argument("--topk_patterns", type=int, default=30)
    ap.add_argument("--topk_slots", type=int, default=50)
    ap.add_argument("--examples_per_slot", type=int, default=2)
    ap.add_argument("--text_max_len", type=int, default=180)
    ap.add_argument("--topk_services", type=int, default=30)
    ap.add_argument("--topk_service_intents", type=int, default=50)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    slot_meta: Dict[str, Dict[str, Any]] = read_json(Path(args.slot_meta))

    # --- Global stats ---
    stats: Dict[str, Any] = {
        "splits": splits,
        "dirs": DIRS,
        "by_split": {},
    }

    # --- Audit tables (collect per split, also merged) ---
    all_audit_rows: List[SlotAuditRow] = []

    for split in splits:
        path = data_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"[skip] not found: {path}")
            continue

        # Direction counts (from labels)
        label_pos = Counter()
        
        # service distributions
        service_ctr = Counter()
        service_intent_ctr = Counter()
        service_pos = defaultdict(Counter)

        n_rows = 0

        # Missing-dir pattern counts
        pattern_ctr = Counter()

        # slot-level counts (by service/intent/slot)
        # required_count: slot appears in required_slots for that example
        # dropped_count: slot appears in missing_slots for that example
        req_ctr = Counter()    # key=(service,intent,slot)
        drop_ctr = Counter()   # key=(service,intent,slot)

        # examples per slot (store short text samples where dropped)
        examples: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

        # optional: which mapped_dir each slot goes to (check consistency)
        mapped_dir_by_key: Dict[Tuple[str, str, str], str] = {}

        for r in read_jsonl(path):
            n_rows += 1

            service = str(r.get("service", ""))
            intent = str(r.get("intent", ""))
            required_slots = r.get("required_slots", []) or []
            missing_slots = r.get("missing_slots", []) or []
            missing_dirs = r.get("missing_dirs", []) or []
            labels = r.get("labels", {}) or {}
            text = str(r.get("text", ""))

            # service distributions
            if service:
                service_ctr[service] += 1
            if service and intent:
                service_intent_ctr[(service, intent)] += 1

            # label distribution (positive counts)
            for d in DIRS:
                if int(labels.get(d, 0)) == 1:
                    label_pos[d] += 1

            # service-wise label positives
            if service:
                for d in DIRS:
                    if int(labels.get(d, 0)) == 1:
                        service_pos[service][d] += 1

            # missing_dirs pattern distribution
            patt = tuple(sorted(set(str(x) for x in missing_dirs)))
            pattern_ctr[patt] += 1

            # required slot counts
            for sl in required_slots:
                key = (service, intent, str(sl))
                req_ctr[key] += 1

                # record mapped dir (from slot_meta) for audit
                if key not in mapped_dir_by_key:
                    desc = safe_get_slot_desc(slot_meta, service, str(sl))
                    mapped_dir_by_key[key] = map_slot_to_dir(str(sl), desc)

            # dropped slot counts + examples
            for sl in missing_slots:
                key = (service, intent, str(sl))
                drop_ctr[key] += 1

                if key not in mapped_dir_by_key:
                    desc = safe_get_slot_desc(slot_meta, service, str(sl))
                    mapped_dir_by_key[key] = map_slot_to_dir(str(sl), desc)

                if len(examples[key]) < args.examples_per_slot:
                    examples[key].append(short_text(text, n=args.text_max_len))

        # Build slot audit rows (top slots by required frequency, but include all that appear)
        keys_all = set(req_ctr.keys()) | set(drop_ctr.keys())
        # sort by dropped_count then required_count
        keys_sorted = sorted(
            keys_all,
            key=lambda k: (drop_ctr.get(k, 0), req_ctr.get(k, 0)),
            reverse=True,
        )

        audit_rows: List[SlotAuditRow] = []
        for (service, intent, slot) in keys_sorted:
            desc = safe_get_slot_desc(slot_meta, service, slot)
            mapped = mapped_dir_by_key.get((service, intent, slot), map_slot_to_dir(slot, desc))
            exs = examples.get((service, intent, slot), [])
            ex1 = exs[0] if len(exs) >= 1 else ""
            ex2 = exs[1] if len(exs) >= 2 else ""
            audit_rows.append(
                SlotAuditRow(
                    split=split,
                    service=service,
                    intent=intent,
                    slot=slot,
                    mapped_dir=mapped,
                    required_count=int(req_ctr.get((service, intent, slot), 0)),
                    dropped_count=int(drop_ctr.get((service, intent, slot), 0)),
                    slot_desc=desc,
                    example_1=ex1,
                    example_2=ex2,
                )
            )

        # Save per-split audit CSV (limit topk if user wants)
        audit_out = out_dir / f"slot_audit_{split}.csv"
        write_slot_audit_csv(audit_out, audit_rows)

        # Add to merged
        all_audit_rows.extend(audit_rows)

        # Save per-split stats
        total = n_rows or 1
        stats["by_split"][split] = {
            "n_rows": n_rows,
            "label_pos_counts": {d: int(label_pos[d]) for d in DIRS},
            "label_pos_rates": {d: float(label_pos[d]) / total for d in DIRS},
            "top_missing_dirs_patterns": [
                {"pattern": list(p), "count": int(c), "rate": float(c) / total}
                for p, c in pattern_ctr.most_common(args.topk_patterns)
            ],
            "top_slots_by_dropped": [
                {"service": k[0], "intent": k[1], "slot": k[2], "mapped_dir": mapped_dir_by_key.get(k, ""),
                 "dropped_count": int(drop_ctr.get(k, 0)), "required_count": int(req_ctr.get(k, 0))}
                for k in keys_sorted[: args.topk_slots]
            ],
            "service_distribution": [
                {"service": s, "count": int(c), "rate": float(c) / total}
                for s, c in service_ctr.most_common(args.topk_services)
            ],
            "service_intent_distribution": [
                {"service": si[0], "intent": si[1], "count": int(c), "rate": float(c) / total}
                for si, c in service_intent_ctr.most_common(args.topk_service_intents)
            ],
            "service_label_pos_rates": [
                {
                    "service": s,
                    "count": int(service_ctr[s]),
                    "pos_counts": {d: int(service_pos[s][d]) for d in DIRS},
                    "pos_rates": {d: (float(service_pos[s][d]) / service_ctr[s]) for d in DIRS},
                }
                for s, _c in service_ctr.most_common(args.topk_services)
            ],
        }

        print(f"[done] split={split} rows={n_rows} -> {audit_out}")

    # Save merged audit CSV
    merged_out = out_dir / "slot_audit_all.csv"
    write_slot_audit_csv(merged_out, all_audit_rows)

    # Save stats json
    stats_out = out_dir / "stats_dirunc.json"
    stats_out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] merged audit -> {merged_out}")
    print(f"[done] stats -> {stats_out}")

if __name__ == "__main__":
    main()
