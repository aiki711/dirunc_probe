# scripts/01_build_required_slots.py
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

SPLITS = ["train", "dev", "test", "test_seen", "test_unseen"]

def load_schema_files(sgd_root: Path) -> List[Path]:
    schema_files: List[Path] = []
    for sp in SPLITS:
        p = sgd_root / sp / "schema.json"
        if p.exists():
            schema_files.append(p)
    # de-dup (train/dev/test often share services)
    return list(dict.fromkeys(schema_files))

def main() -> None:
    sgd_root = Path("data/raw/sgd")
    out_dir = Path("data/processed/sgd")
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_files = load_schema_files(sgd_root)
    if not schema_files:
        raise FileNotFoundError("No schema.json found. Did you run scripts/00_fetch_sgd.py ?")

    # key: "service::intent" -> {"required_slots":[...], "optional_slots":[...]}
    required_map: Dict[str, Dict[str, List[str]]] = {}
    # optional: slot descriptions for later slot->5W1H mapping refinement
    slot_meta: Dict[str, Dict[str, Any]] = {}

    for sf in schema_files:
        data = json.loads(sf.read_text(encoding="utf-8"))
        for svc in data:
            service_name = svc["service_name"]
            # slots meta
            for sl in svc.get("slots", []):
                slot_meta[f"{service_name}::{sl['name']}"] = sl
            # intents
            for it in svc.get("intents", []):
                intent_name = it["name"]
                key = f"{service_name}::{intent_name}"
                required_map[key] = {
                    "required_slots": list(it.get("required_slots", [])),
                    "optional_slots": list(it.get("optional_slots", [])),
                }

    (out_dir / "required_slots_by_service_intent.json").write_text(
        json.dumps(required_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "slot_meta_by_service_slot.json").write_text(
        json.dumps(slot_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[done] wrote: {out_dir/'required_slots_by_service_intent.json'}")
    print(f"[done] wrote: {out_dir/'slot_meta_by_service_slot.json'}")
    print(f"  intents: {len(required_map)}")
    print(f"  slots  : {len(slot_meta)}")

if __name__ == "__main__":
    main()
