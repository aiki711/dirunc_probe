#!/usr/bin/env python3
"""
Slot-wise Logit Prompting Baseline  (1:1 correspondence with Probing)
======================================================================

Pipeline (mirrors probing exactly):
  Step 1: For each of 5 case-role slots, ask the LLM with a 5-shot prompt:
            "Is [SLOT] missing from the last user utterance? Yes or No:"
          Extract P(Yes) from the FIRST generated token's logit distribution.

  Step 2: Calibrate per-slot threshold on CAL split (same dev_cal_indices.npy).
          Threshold = argmax_t F1(cal, t) for that slot.

  Step 3: Evaluate on TEST split (dev_test_indices.npy).
          Verify  = any slot's P(Yes) ≥ threshold  → Insufficient
          Identify = slot with highest P(Yes) above threshold → role name

This 1:1 mapping lets us directly compare:
  Calibration quality: Are LLM output probabilities as well-calibrated as probe outputs?
  Computational cost : 5× LLM forward passes vs. 1 forward pass + 5 lightweight linear ops
  Interpretability   : Both provide per-slot explanation; probe does it in one LLM forward pass

Outputs:
  runs/identify_verify_comparison/slotwise_logit_results.json
"""
import json, random, os, sys, warnings
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Slot definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLOT_META = {
    "who": {
        "description": "WHO (the person or agent making the request, e.g. a specific person's name if needed)",
        "examples": ["I", "my colleague", "a group of 4", "the patient"]
    },
    "what": {
        "description": "WHAT (the specific item or service being requested, e.g. restaurant type, hotel category, attraction name)",
        "examples": ["an Italian restaurant", "a 3-star hotel", "a science museum", "a taxi"]
    },
    "when": {
        "description": "WHEN (the time, day, or date the action is needed, e.g. tonight, Monday, 7pm, next week)",
        "examples": ["tonight at 7pm", "Monday", "next Friday", "3 nights starting Tuesday"]
    },
    "where": {
        "description": "WHERE (the location, area, origin, or destination, e.g. city centre, Cambridge, north part of town)",
        "examples": ["in the city centre", "near the station", "from Cambridge to London", "in the north part"]
    },
    "how": {
        "description": "HOW (manner, price range, or quantity — e.g. cheap, moderately priced, for 2 people, 3 nights)",
        "examples": ["cheap", "for 4 people", "moderately priced", "2 nights"]
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5-shot examples per slot (dialogue snippets from MultiWOZ-style data)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SLOT_EXAMPLES = {
    "who": [
        # (text, answer)
        ("[system]: Can I help you? [user]: I want to book a table for 4 people.", "No"),
        ("[system]: Who is the reservation for? [user]: I need a hotel room.", "Yes"),
        ("[system]: How many guests? [user]: It will be 2 guests, myself and my partner.", "No"),
        ("[system]: Is this for a business trip? [user]: I just need somewhere to stay tonight.", "Yes"),
        ("[system]: Any special requests? [user]: Yes, it's for a birthday party of 6.", "No"),
    ],
    "what": [
        ("[system]: What can I help you with? [user]: I'd like an Italian restaurant please.", "No"),
        ("[system]: What are you looking for? [user]: I want something in the city centre.", "Yes"),
        ("[system]: What type of accommodation? [user]: A cheap hotel near the station.", "No"),
        ("[system]: How can I assist? [user]: I need to make a booking for tonight.", "Yes"),
        ("[system]: What kind of food? [user]: Chinese food, moderately priced.", "No"),
    ],
    "when": [
        ("[system]: When would you like to arrive? [user]: I'd like to check in on Monday.", "No"),
        ("[system]: What time? [user]: I need a restaurant in the north part of town.", "Yes"),
        ("[system]: What day? [user]: I need a train from Cambridge to London at 9am on Tuesday.", "No"),
        ("[system]: When do you need it? [user]: I just need a moderately priced restaurant.", "Yes"),
        ("[system]: What time works? [user]: Could you book it for 7:30pm tonight for 3 people?", "No"),
    ],
    "where": [
        ("[system]: Where would you like to go? [user]: I want to visit a museum near the centre.", "No"),
        ("[system]: What area? [user]: I need a cheap guesthouse for 3 nights.", "Yes"),
        ("[system]: Where are you departing from? [user]: I need a train to London on Friday.", "Yes"),
        ("[system]: Which area of town? [user]: I'd like a hotel in the north, please.", "No"),
        ("[system]: Do you have a location preference? [user]: I need something for 2 nights starting Thursday.", "Yes"),
    ],
    "how": [
        ("[system]: Any preferences? [user]: I'd like a cheap Italian restaurant for 2 people.", "No"),
        ("[system]: Any price range? [user]: I need a hotel in the city centre for Tuesday.", "Yes"),
        ("[system]: How many people? [user]: I want to book a table at La Mimosa tonight.", "Yes"),
        ("[system]: How many nights? [user]: I need a moderately priced hotel for 3 nights.", "No"),
        ("[system]: Any special requirements? [user]: I need a taxi from the station to the hotel.", "Yes"),
    ],
}

def build_prompt(slot: str, text: str) -> str:
    meta  = SLOT_META[slot]
    desc  = meta["description"]
    shots = SLOT_EXAMPLES[slot]

    shots_str = "\n".join(
        f'Text: "{t}"\nIs {desc} MISSING from the last user utterance? Answer Yes or No: {a}'
        for t, a in shots
    )

    return f"""You are analyzing task-oriented dialogues.

Task: Determine if the following information is MISSING (not explicitly stated) from the last user utterance.
Information type: {desc}

{shots_str}

Text: "{text}"
Is {desc} MISSING from the last user utterance? Answer Yes or No:"""


def get_yes_prob(model, tokenizer, device, prompt_text: str, yes_id: int, no_id: int) -> float:
    """Return P(Yes) / (P(Yes) + P(No)) from the first generated token."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt").to(device)
    except Exception:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs   = model(input_ids=input_ids)
        logits    = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        p_yes = log_probs[yes_id].exp().item()
        p_no  = log_probs[no_id].exp().item()

    return p_yes / (p_yes + p_no + 1e-9)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
}
ALL_SLOTS   = ["who","what","when","where","how"]
ALL_CLASSES = ["who","what","when","where","how","None"]


def build_eval_items(dev_pairs, indices, seed=42):
    """Build per-slot evaluation items from a set of pair indices.
    Returns: list of (pair_idx, condition, slot_label)
      condition ∈ {'missing','filled'}
      slot_label ∈ {'who','what','when','where','how','None'}
    """
    slot_missing = {s: [] for s in ALL_SLOTS}
    filled_pool  = []
    for i in indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot_missing[ROLE_TO_DIR[role]].append(i)
            filled_pool.append(i)

    class_groups = {c: [] for c in ALL_CLASSES}
    for slot, idxs in slot_missing.items():
        for i in idxs:
            class_groups[slot].append((i, "missing", slot))
    for i in filled_pool:
        class_groups["None"].append((i, "filled", "None"))

    num_per = max(1, 300 // len(ALL_CLASSES))
    random.seed(seed)
    items = []
    for c in ALL_CLASSES:
        pool = class_groups[c]
        items.extend(random.sample(pool, min(len(pool), num_per)))
    return items


def main():
    MODEL_NAME = "google/gemma-2-2b-it"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load dev pairs ─────────────────────────────────────────────────
    print("Loading dev pairs...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32       = load_s32()
    dev_ds    = s32.PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    dev_cache = torch.load(
        CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    cal_items  = build_eval_items(dev_pairs, cal_indices,  seed=0)
    test_items = build_eval_items(dev_pairs, test_indices, seed=42)
    print(f"  Cal items:  {len(cal_items)}")
    print(f"  Test items: {len(test_items)}")
    from collections import Counter
    print(f"  Cal  cond dist: {dict(Counter(c for _,c,_ in cal_items))}")
    print(f"  Test cond dist: {dict(Counter(c for _,c,_ in test_items))}")

    # ── 2. Load LLM ──────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]
    print(f"  Yes token: {yes_id} ('{tokenizer.decode([yes_id])}')")
    print(f"  No  token: {no_id}  ('{tokenizer.decode([no_id])}')")

    # ── 3. Compute P(Yes) for each slot on CAL items ──────────────────────
    print("\n=== Computing P(Yes) on CAL items ===")
    # cal_slot_probs[slot] = list of (p_yes, true_label_for_this_slot)
    # true_label: 1 if slot_label == slot, 0 otherwise
    cal_slot_probs = {s: [] for s in ALL_SLOTS}
    for idx, cond, slot_label in tqdm(cal_items, desc="CAL"):
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        for slot in ALL_SLOTS:
            prompt = build_prompt(slot, text)
            p_yes  = get_yes_prob(model, tokenizer, device, prompt, yes_id, no_id)
            true_label = 1 if (cond == "missing" and slot_label == slot) else 0
            cal_slot_probs[slot].append((p_yes, true_label))

    # ── 4. Calibrate per-slot threshold on CAL ────────────────────────────
    print("\n=== Per-slot threshold calibration (CAL) ===")
    slot_thresholds = {}
    cal_slot_f1     = {}
    for slot in ALL_SLOTS:
        probs  = np.array([p for p, _ in cal_slot_probs[slot]])
        labels = np.array([l for _, l in cal_slot_probs[slot]])
        n_pos  = labels.sum()
        if n_pos == 0:
            slot_thresholds[slot] = 0.5
            cal_slot_f1[slot] = 0.0
            print(f"  {slot:6s}: NO POSITIVES — skipping")
            continue
        best_t, best_f = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            f = f1_score(labels, preds, pos_label=1, zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        slot_thresholds[slot] = best_t
        cal_slot_f1[slot]     = best_f
        print(f"  {slot:6s}: threshold={best_t:.3f}  cal_F1={best_f*100:.1f}%  (n_pos={int(n_pos)}/{len(labels)})")

    # ── 5. Compute P(Yes) for each slot on TEST items ─────────────────────
    print("\n=== Computing P(Yes) on TEST items ===")
    test_slot_probs = {s: [] for s in ALL_SLOTS}
    test_meta       = []  # (cond, slot_label)
    for idx, cond, slot_label in tqdm(test_items, desc="TEST"):
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        row_probs = {}
        for slot in ALL_SLOTS:
            prompt = build_prompt(slot, text)
            p_yes  = get_yes_prob(model, tokenizer, device, prompt, yes_id, no_id)
            row_probs[slot] = p_yes
            true_label = 1 if (cond == "missing" and slot_label == slot) else 0
            test_slot_probs[slot].append((p_yes, true_label))
        test_meta.append((cond, slot_label, row_probs))

    # ── 6. Verify: OR over all slots ──────────────────────────────────────
    print("\n=== TEST: Verify (Sufficient vs Insufficient) ===")
    y_true_verify = []
    y_pred_verify = []
    for cond, slot_label, row_probs in test_meta:
        # True label
        true_suff = "Insufficient" if cond == "missing" else "Sufficient"
        y_true_verify.append(true_suff)
        # Prediction: any slot fires above threshold → Insufficient
        any_fire = any(row_probs[s] >= slot_thresholds[s] for s in ALL_SLOTS)
        y_pred_verify.append("Insufficient" if any_fire else "Sufficient")

    acc_v = accuracy_score(y_true_verify, y_pred_verify)
    p_v, r_v, f_v, _ = precision_recall_fscore_support(
        y_true_verify, y_pred_verify, pos_label="Insufficient",
        average="binary", zero_division=0)
    print(f"  Verify Acc:       {acc_v*100:.2f}%")
    print(f"  Verify P/R/F1:    {p_v*100:.2f}% / {r_v*100:.2f}% / {f_v*100:.2f}%")

    # ── 7. Per-slot binary F1 (Test) ──────────────────────────────────────
    print("\n=== TEST: Per-slot Binary F1 ===")
    slot_test_f1 = {}
    slot_test_p  = {}
    slot_test_r  = {}
    for slot in ALL_SLOTS:
        probs  = np.array([p for p, _ in test_slot_probs[slot]])
        labels = np.array([l for _, l in test_slot_probs[slot]])
        n_pos  = labels.sum()
        if n_pos == 0:
            slot_test_f1[slot] = 0.0
            print(f"  {slot:6s}: NO POSITIVES")
            continue
        preds = (probs >= slot_thresholds[slot]).astype(int)
        p = precision_recall_fscore_support(
            labels, preds, pos_label=1, average="binary", zero_division=0)
        slot_test_f1[slot] = p[2]
        slot_test_p[slot]  = p[0]
        slot_test_r[slot]  = p[1]
        print(f"  {slot:6s}: P={p[0]*100:.1f}%  R={p[1]*100:.1f}%  F1={p[2]*100:.1f}%  "
              f"(n_pos={int(n_pos)}/{len(labels)})")

    # ── 8. Identify: highest P(Yes) above threshold ───────────────────────
    print("\n=== TEST: Identify (which slot is missing) ===")
    y_true_id = []
    y_pred_id = []
    for cond, slot_label, row_probs in test_meta:
        true_id = slot_label  # 'who','what','when','where','how','None'
        y_true_id.append(true_id)
        # Prediction: slot with highest P(Yes) above threshold, else 'None'
        firing = {s: row_probs[s] for s in ALL_SLOTS if row_probs[s] >= slot_thresholds[s]}
        if firing:
            pred_id = max(firing, key=firing.get)
        else:
            pred_id = "None"
        y_pred_id.append(pred_id)

    acc_id = accuracy_score(y_true_id, y_pred_id)
    f1_id  = f1_score(y_true_id, y_pred_id, average="macro", zero_division=0)
    print(f"  Identify Acc:   {acc_id*100:.2f}%")
    print(f"  Identify F1 (macro): {f1_id*100:.2f}%")

    # ── 9. Save results ───────────────────────────────────────────────────
    results = {
        "method": f"Slot-wise Logit Prompting (5-shot, per-slot threshold, {MODEL_NAME})",
        "eval_split": "test (held-out 50% of dev, stratified)",
        "eval_size": len(test_items),
        "per_slot_thresholds": {s: float(slot_thresholds[s]) for s in ALL_SLOTS},
        "cal_slot_f1": {s: float(cal_slot_f1[s]) for s in ALL_SLOTS},
        "verify_accuracy":    float(acc_v),
        "verify_f1_omission": float(f_v),
        "verify_precision":   float(p_v),
        "verify_recall":      float(r_v),
        "slot_f1": {s: float(slot_test_f1.get(s, 0)) for s in ALL_SLOTS},
        "slot_precision": {s: float(slot_test_p.get(s, 0)) for s in ALL_SLOTS},
        "slot_recall": {s: float(slot_test_r.get(s, 0)) for s in ALL_SLOTS},
        "identify_accuracy": float(acc_id),
        "identify_f1_macro": float(f1_id),
    }
    out = OUT_DIR / "slotwise_logit_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")

    # ── 10. Summary comparison ────────────────────────────────────────────
    print("\n====== Summary: Slot-wise Logit Prompting ======")
    print(f"  Verify F1 (Omission): {f_v*100:.2f}%")
    print(f"  Per-slot F1:")
    for s in ALL_SLOTS:
        f = slot_test_f1.get(s, 0)
        print(f"    {s:6s}: {f*100:.1f}%")
    print(f"\n  [Compare] Probing Verify F1: 88.44%  | Per-slot F1: when=74.8%, how=74.2%, what=64.7%, where=76.0%")


if __name__ == "__main__":
    main()
