import os
import sys
import json
import random
from pathlib import Path
from collections import Counter
import gzip
import re

# Add project root to sys.path
sys.path.append(os.getcwd())
try:
    from scripts.common import DIRS, build_label_dict, write_jsonl, replace_values_in_text
except ImportError:
    print("Cannot import from scripts.common - Ensure you run from project root.")
    sys.exit(1)

random.seed(42)

# --- MultiWOZ Mapping ---
def map_multiwoz_to_dir(domain: str, slot: str) -> str:
    # WHEN: Time, Day, LeaveAt, ArriveBy
    # WHERE: Destination, Departure, Area
    # HOW: People, Price, Internet, Parking, Stay (stay nights -> how many)
    # WHAT: Food, Type, Name, Department (hospital), Stars
    # WHO: None/rare.
    slot = slot.lower()
    
    when_slots = ["time", "day", "leaveat", "arriveby"]
    where_slots = ["destination", "departure", "area"]
    how_slots = ["people", "price", "internet", "parking", "stay", "pricerange"]
    what_slots = ["food", "type", "name", "department", "stars"]
    
    if slot in when_slots:
        return "when"
    elif slot in where_slots:
        return "where"
    elif slot in how_slots:
        return "how"
    elif slot in what_slots:
        return "what"
    
    # Fallback
    return "what"

# --- QA-SRL Mapping ---
def map_qasrl_wh(wh: str) -> str:
    wh = wh.lower().strip()
    if wh in DIRS:
        return wh
    if wh == "who":
        return "who"
    if wh == "what":
        return "what"
    if wh == "when":
        return "when"
    if wh == "where":
        return "where"
    if wh == "why":
        return "why"
    if wh == "how":
        return "how"
    if wh == "which":
        return "which"
    return "what"

def _balance_dataset(rows, seed=42):
    if not rows:
        return []
    rng = random.Random(seed)
    label_to_idxs = {d: [] for d in DIRS}
    for i, row in enumerate(rows):
        for d in DIRS:
            if row["labels"].get(d, 0) == 1:
                label_to_idxs[d].append(i)
    counts = {d: len(idxs) for d, idxs in label_to_idxs.items()}
    non_zero_counts = [c for c in counts.values() if c > 0]
    min_count = min(non_zero_counts) if non_zero_counts else 0
    
    print(f"Balancing: min count per label = {min_count}")
    
    selected_idxs = set()
    for d in DIRS:
        idxs = label_to_idxs[d]
        already_selected = [idx for idx in idxs if idx in selected_idxs]
        if len(already_selected) >= min_count:
            continue
        needed = min_count - len(already_selected)
        candidates = [idx for idx in idxs if idx not in selected_idxs]
        rng.shuffle(candidates)
        selected_idxs.update(candidates[:needed])
        
    balanced_rows = [rows[i] for i in sorted(list(selected_idxs))]
    # Also add equivalent number of resolved (we extracted the specific required negative pairs)
    # Actually wait. If we just select random unresolved, what happens to the corresponding resolved pair?
    # Our generated list has resolved/unresolved sequentially.
    # We should select the PAIR.
    # Let's fix this for pairs.
    
    # pair_id -> [filled, missing]
    pairs = {}
    for r in rows:
        pid = r["id"].rsplit("::", 1)[0]
        if pid not in pairs:
            pairs[pid] = {}
        pairs[pid][r["condition"]] = r
        
    # extract valid pairs containing both filled and missing
    valid_pairs = []
    for pid, d in pairs.items():
        if "filled" in d and "missing" in d:
            valid_pairs.append((d["filled"], d["missing"]))
            
    # count based on missing labels
    label_to_pair_idxs = {d: [] for d in DIRS}
    for i, (res, unr) in enumerate(valid_pairs):
        for d in DIRS:
            if unr["labels"].get(d, 0) == 1:
                label_to_pair_idxs[d].append(i)
                
    counts = {d: len(idxs) for d, idxs in label_to_pair_idxs.items()}
    non_zero_counts = [c for c in counts.values() if c > 0]
    min_count = min(non_zero_counts) if non_zero_counts else 0
    min_count = min(min_count, 5000) # Cap at 5000 pairs max per label for safety
    
    selected_pair_idxs = set()
    for d in DIRS:
        idxs = label_to_pair_idxs[d]
        already_selected = [idx for idx in idxs if idx in selected_pair_idxs]
        if len(already_selected) >= min_count:
            continue
        needed = min_count - len(already_selected)
        candidates = [idx for idx in idxs if idx not in selected_pair_idxs]
        rng.shuffle(candidates)
        selected_pair_idxs.update(candidates[:needed])
        
    final_rows = []
    for i in sorted(list(selected_pair_idxs)):
        res, unr = valid_pairs[i]
        final_rows.append(res)
        final_rows.append(unr)
        
    return final_rows

def load_json(fp):
    with open(fp, "r") as f:
        return json.load(f)

# -------------
# 1. MultiWOZ
# -------------
def process_multiwoz(data_path="data/raw/multiwoz/data.json", limit_dialogues=0):
    print("Processing MultiWOZ...")
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return []
        
    with open(data_path, "r") as f:
        data = json.load(f)
        
    out_rows = []
    dlg_cnt = 0
    for file_id, dialog in data.items():
        if limit_dialogues and dlg_cnt >= limit_dialogues:
            break
        dlg_cnt += 1
        
        log = dialog.get("log", [])
        context_turns = []
        
        for turn_idx, turn in enumerate(log):
            text = turn["text"].strip()
            
            # Simple heuristic for User vs System
            is_user = (turn_idx % 2 == 0)
            speaker_prefix = "User: " if is_user else "Assistant: "
            
            if is_user:
                spans = turn.get("span_info", [])
                metadata = turn.get("metadata", {})
                
                # Active domains from metadata 
                active_domains = [dom for dom, dinfo in metadata.items() if any(v for k,v in dinfo.get("semi", {}).items() if v not in ["", "not mentioned", "none"])]
                dom_str = f"[Domain: {', '.join(active_domains)}]\n" if active_domains else ""
                
                # Find informative spans
                for span in spans:
                    if len(span) >= 5:
                        act_type, slot, value, start, end = span[:5]
                        if act_type.endswith("-Inform") and act_type != "general-inform":
                            domain = act_type.split("-")[0]
                            target_dir = map_multiwoz_to_dir(domain, slot)
                            
                            words = text.split()
                            if start < len(words) and end < len(words) and start <= end:
                                words_ablat = words[:start] + words[end + 1:]
                                dropped_span = " ".join(words[start:end + 1])
                                ablat_text = " ".join(words_ablat)
                                if not ablat_text:
                                    ablat_text = "."
                                    
                                hist = context_turns[-3:] 
                                context_str = dom_str + "\n".join(hist) + "\n" if hist else dom_str
                                
                                resolved_text = context_str + speaker_prefix + text
                                unresolved_text = context_str + speaker_prefix + ablat_text
                                
                                pid = f"multiwoz::{file_id}::t{turn_idx}::{domain}-{slot}"
                                labels_res = build_label_dict([])
                                labels_unr = build_label_dict([target_dir])
                                
                                out_rows.append({
                                    "id": pid + "::filled",
                                    "text": resolved_text,
                                    "labels": labels_res,
                                    "condition": "filled",
                                    "dataset": "multiwoz",
                                    "metadata": {"domain": domain, "slot": slot, "value": value, "original_span": dropped_span}
                                })
                                out_rows.append({
                                    "id": pid + "::missing",
                                    "text": unresolved_text,
                                    "labels": labels_unr,
                                    "condition": "missing",
                                    "dataset": "multiwoz",
                                    "metadata": {"domain": domain, "slot": slot, "value": value, "dropped_span": dropped_span}
                                })
            
            context_turns.append(speaker_prefix + text)
            
    print(f"MultiWOZ processing done. Generated {len(out_rows)} pairs (pre-balance).")
    return _balance_dataset(out_rows)

# -------------
# 2. QA-SRL
# -------------
def process_qasrl(data_path="temp_qasrl/qasrl-v2/orig/dev.jsonl.gz", limit=0):
    print(f"Processing QA-SRL ({data_path})...")
    if not os.path.exists(data_path):
        print(f"Warning: {data_path} not found.")
        return []
        
    out_rows = []
    cnt = 0
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if limit and cnt >= limit:
                break
            cnt += 1
            doc = json.loads(line)
            sid = doc.get("sentenceId", f"sent_{cnt}")
            tokens = doc.get("sentenceTokens", [])
            verbData = doc.get("verbEntries", {})
            
            for vb_id, vb_info in verbData.items():
                target_verb = vb_info.get("verbInflectedForms", {}).get("stem", "something")
                qLabels = vb_info.get("questionLabels", {})
                
                for pq, qinfo in qLabels.items():
                    wh = qinfo.get("questionSlots", {}).get("wh", "what")
                    target_dir = map_qasrl_wh(wh)
                    
                    judgments = qinfo.get("answerJudgments", [])
                    valid_spans = []
                    for j in judgments:
                        if j.get("isValid") and j.get("spans"):
                            valid_spans.extend(j["spans"])
                    
                    if valid_spans:
                        s_start, s_end = valid_spans[0]
                        if s_start < len(tokens) and s_end <= len(tokens):
                            ablat_tokens = tokens[:s_start] + tokens[s_end:]
                            dropped_span = " ".join(tokens[s_start:s_end])
                            
                            resolved_text = f"[Target Verb: {target_verb}] " + " ".join(tokens)
                            unresolved_text = f"[Target Verb: {target_verb}] " + " ".join(ablat_tokens)
                            from scripts.common import cleanup_deletion_artifacts
                            unresolved_text = cleanup_deletion_artifacts(unresolved_text)
                            
                            if resolved_text == unresolved_text:
                                continue
                            
                            pid = f"qasrl::{sid}::{target_verb}::{wh}"
                            labels_res = build_label_dict([])
                            labels_unr = build_label_dict([target_dir])
                            
                            out_rows.append({
                                "id": pid + "::filled",
                                "text": resolved_text,
                                "labels": labels_res,
                                "condition": "filled",
                                "dataset": "qasrl",
                                "metadata": {"verb": target_verb, "question": pq, "original_span": dropped_span}
                            })
                            out_rows.append({
                                "id": pid + "::missing",
                                "text": unresolved_text,
                                "labels": labels_unr,
                                "condition": "missing",
                                "dataset": "qasrl",
                                "metadata": {"verb": target_verb, "question": pq, "dropped_span": dropped_span}
                            })

    print(f"QA-SRL processing done. Generated {len(out_rows)} pairs (pre-balance).")
    return _balance_dataset(out_rows)


# -------------
# 3. SGD (Simplified)
# -------------
def process_sgd(limit_dialogues=0):
    try:
        from scripts.common import map_slot_to_dir
    except ImportError:
        return []
        
    print("Processing SGD...")
    root = Path("data/raw/sgd/train")
    files = list(root.glob("dialogues_*.json"))
    if not files:
        files = list(root.glob("dialogue_*.json"))
    
    req_map_path = Path("data/processed/sgd/required_slots_by_service_intent.json")
    if not req_map_path.exists():
        print("Required map for SGD not found. Skipping SGD.")
        return []
    req_map = load_json(req_map_path)
    
    meta_path = Path("data/processed/sgd/slot_meta_by_service_slot.json")
    meta = load_json(meta_path) if meta_path.exists() else {}
    
    out_rows = []
    dlg_cnt = 0
    for fp in files:
        if limit_dialogues and dlg_cnt >= limit_dialogues:
            break
        dialogues = load_json(fp)
        for d in dialogues:
            if limit_dialogues and dlg_cnt >= limit_dialogues:
                break
            dlg_cnt += 1
            
            did = d.get("dialogue_id", "")
            turns = d["turns"]
            context_turns = []
            
            for t_idx, turn in enumerate(turns):
                is_user = turn.get("speaker", "").upper() == "USER"
                text = turn["utterance"]
                spk = "User: " if is_user else "Assistant: "
                
                if is_user:
                    for fr in turn.get("frames", []):
                        svc = fr.get("service")
                        state = fr.get("state", {})
                        intent = state.get("active_intent")
                        
                        if not svc or not intent or intent == "NONE":
                            continue
                            
                        key = f"{svc}::{intent}"
                        if key not in req_map:
                            continue
                        
                        req_slots = req_map[key].get("required_slots", [])
                        if not req_slots:
                            continue
                            
                        sv = state.get("slot_values", {})
                        for sl in req_slots:
                            vals = sv.get(sl, [])
                            if vals:
                                val_str = str(vals[0]) if isinstance(vals, list) else str(vals)
                                mdesc = meta.get(f"{svc}::{sl}", {}).get("description", "")
                                target_dir = map_slot_to_dir(sl, mdesc)
                                
                                ablat_text = replace_values_in_text(text, [val_str], mode="delete")
                                
                                dom_str = f"[Domain: {svc} / Intent: {intent}]\n"
                                hist = context_turns[-3:]
                                ctx = dom_str + "\n".join(hist) + "\n" if hist else dom_str
                                
                                resolved_text = ctx + spk + text
                                unresolved_text = ctx + spk + ablat_text
                                
                                if resolved_text == unresolved_text:
                                    continue
                                
                                pid = f"sgd::{did}::t{t_idx}::{sl}"
                                out_rows.append({
                                    "id": pid + "::filled",
                                    "text": resolved_text,
                                    "labels": build_label_dict([]),
                                    "condition": "filled",
                                    "dataset": "sgd",
                                    "metadata": {"service": svc, "intent": intent, "slot": sl, "original_span": val_str}
                                })
                                out_rows.append({
                                    "id": pid + "::missing",
                                    "text": unresolved_text,
                                    "labels": build_label_dict([target_dir]),
                                    "condition": "missing",
                                    "dataset": "sgd",
                                    "metadata": {"service": svc, "intent": intent, "slot": sl, "dropped_span": val_str}
                                })
                
                context_turns.append(spk + text)
                
    print(f"SGD processing done. Generated {len(out_rows)} pairs (pre-balance).")
    return _balance_dataset(out_rows)

def main():
    root_out = Path("data/processed/mixed/")
    root_out.mkdir(parents=True, exist_ok=True)
    
    # QA-SRL Processing
    qasrl_rows = process_qasrl(data_path="temp_qasrl/qasrl-v2/orig/dev.jsonl.gz")
    if qasrl_rows:
        qasrl_out = root_out / "semantic_v3_qasrl.jsonl"
        write_jsonl(qasrl_out, qasrl_rows)
        print(f"Saved {len(qasrl_rows)} QA-SRL examples to {qasrl_out}")
    
    # MultiWOZ Processing
    mwoz_rows = process_multiwoz(data_path="data/raw/multiwoz/data.json")
    if mwoz_rows:
        mwoz_out = root_out / "semantic_v3_multiwoz.jsonl"
        write_jsonl(mwoz_out, mwoz_rows)
        print(f"Saved {len(mwoz_rows)} MultiWOZ examples to {mwoz_out}")
    
    # SGD Processing
    sgd_rows = process_sgd()
    if sgd_rows:
        sgd_out = root_out / "semantic_v3_sgd.jsonl"
        write_jsonl(sgd_out, sgd_rows)
        print(f"Saved {len(sgd_rows)} SGD examples to {sgd_out}")

if __name__ == "__main__":
    main()
