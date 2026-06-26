#!/usr/bin/env python3
"""
Improved One-step Prompting Baseline.

Improvements over eval_onestep_prompting.py:
  1. Clearer task definition with case role criteria
  2. Few-shot examples (3-shot) from MultiWOZ-style dialogues
  3. Evaluated on the held-out TEST split (consistent with probing)
  4. Chain-of-thought option (--cot flag)

Produces: runs/identify_verify_comparison/improved_onestep_results.json
"""
import os, sys, json, random, warnings
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Few-shot examples from MultiWOZ-style dialogues
# (carefully selected to cover various slot types)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW_SHOT_EXAMPLES = """
Here are some examples of the task:

--- Example 1 (Insufficient — missing WHERE) ---
Text:
\"\"\"
[system]: I can help you find a hotel. What kind of hotel are you looking for?
[user]: I need a cheap hotel for 3 nights starting Friday.
\"\"\"
Analysis: The user specified WHAT (hotel), HOW (cheap, 3 nights), WHEN (starting Friday), but WHERE (location/area) is missing. A location is essential to search for a hotel.
Answer: Insufficient

--- Example 2 (Sufficient) ---
Text:
\"\"\"
[system]: What restaurant are you looking for?
[user]: I want a moderately priced Italian restaurant in the centre of town for 2 people tonight at 7pm.
\"\"\"
Analysis: The user specified WHAT (Italian restaurant), HOW (moderately priced, 2 people), WHERE (centre), and WHEN (tonight at 7pm). All essential information for a restaurant booking is present.
Answer: Sufficient

--- Example 3 (Insufficient — missing WHEN) ---
Text:
\"\"\"
[system]: Where would you like to travel?
[user]: I need a train from Cambridge to London.
\"\"\"
Analysis: The user specified WHAT (train), WHERE (Cambridge to London), but WHEN (departure time or day) is missing. Timing is needed to find a specific train.
Answer: Insufficient

--- Example 4 (Sufficient — not all slots required) ---
Text:
\"\"\"
[system]: Can I help you find an attraction?
[user]: I am looking for a museum in the east part of the city.
\"\"\"
Analysis: The user specified WHAT (museum) and WHERE (east part). For finding an attraction, timing and exact quantity are not required. Sufficient information is provided.
Answer: Sufficient

--- Example 5 (Insufficient — missing HOW/quantity) ---
Text:
\"\"\"
[system]: I can make a restaurant reservation for you. What are the details?
[user]: Please book a table at Frankie and Benny's for tonight.
\"\"\"
Analysis: The user specified WHAT (Frankie and Benny's), WHEN (tonight), but HOW MANY (number of people) is missing — essential for a reservation.
Answer: Insufficient
"""

PROMPT_TEMPLATE = """You are an expert at evaluating task-oriented dialogues.

Your task: Determine if the last user utterance provides SUFFICIENT information to complete the user's request, or if important information is MISSING.

Key information types to check (only when relevant to the request):
- WHAT: What service or item is requested? (e.g., hotel type, restaurant cuisine)
- WHERE: What location? (e.g., area of city, destination)
- WHEN: What time or date? (e.g., check-in date, reservation time)
- HOW: What manner, quantity, or specification? (e.g., number of nights, party size, price range)
- WHO: Who is involved? (usually implicit, check only if explicitly needed)

Important: Only mark as Insufficient if truly NECESSARY information is missing for the specific request. Not every slot is required for every task.
{few_shot}
--- Now evaluate this dialogue ---
Text:
\"\"\"
{text}
\"\"\"
Does the last user utterance provide sufficient information to complete the request?
Answer with ONLY one word: Sufficient or Insufficient"""


def run_improved_prompting(model, tokenizer, device, text, use_cot=False):
    if use_cot:
        # Chain-of-thought: ask model to reason first, then answer
        cot_prompt = PROMPT_TEMPLATE.format(few_shot=FEW_SHOT_EXAMPLES, text=text)
        cot_prompt = cot_prompt.replace(
            "Answer with ONLY one word: Sufficient or Insufficient",
            "Think step by step:\n1. What is being requested?\n2. What information is needed?\n3. What information is provided?\n4. Is anything crucial missing?\nFinal answer (one word): Sufficient or Insufficient"
        )
        messages = [{"role": "user", "content": cot_prompt}]
    else:
        prompt = PROMPT_TEMPLATE.format(few_shot=FEW_SHOT_EXAMPLES, text=text)
        messages = [{"role": "user", "content": prompt}]

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt").to(device)
    except Exception:
        combined = messages[0]["content"]
        input_ids = tokenizer.encode(combined, return_tensors="pt").to(device)

    with torch.no_grad():
        gen_out = model.generate(
            input_ids,
            max_new_tokens=80 if use_cot else 15,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_tokens = gen_out[0][input_ids.shape[1]:]
    resp = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

    # Parse: look for last occurrence of sufficient/insufficient in response
    if "insufficient" in resp:
        return "Insufficient", resp
    elif "sufficient" in resp:
        return "Sufficient", resp
    else:
        # Fallback: if neither word found, default to Insufficient (model uncertain)
        return "Insufficient", resp


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="google/gemma-2-2b-it")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--no_fewshot", action="store_true", help="Disable few-shot examples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Few-shot: {not args.no_fewshot}  |  CoT: {args.cot}")

    # ── Load dev pairs and test split ─────────────────────────────────────
    print("Loading dev pairs and test split...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32       = load_s32()
    dev_ds    = s32.PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    dev_cache = torch.load(
        CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")
    ROLE_TO_DIR = {
        "Agent":"who","Theme":"what","Location":"where",
        "Source":"where","Goal":"where","Time":"when","Manner":"how",
    }
    ALL_CLASSES = ["who","what","when","where","how","None"]

    test_slot_missing = {s: [] for s in ["who","what","when","where","how"]}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role","")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    class_groups = {c: [] for c in ALL_CLASSES}
    for slot, idxs in test_slot_missing.items():
        if slot in ALL_CLASSES:
            for i in idxs:
                class_groups[slot].append((i, "missing"))
    for i in test_filled_idx:
        class_groups["None"].append((i, "filled"))

    num_per_class = max(1, 300 // 6)
    random.seed(42)
    sampled_items = []
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled_items.extend(random.sample(idxs, min(len(idxs), num_per_class)))
    print(f"  Test split: {len(sampled_items)} items")

    eval_texts  = []
    y_true_str  = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        eval_texts.append(text)
        y_true_str.append("Insufficient" if cond == "missing" else "Sufficient")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    model.eval()

    # ── Inference ──────────────────────────────────────────────────────────
    mode = "CoT" if args.cot else ("Few-shot" if not args.no_fewshot else "Zero-shot improved")
    print(f"\nRunning {mode} prompting on {len(eval_texts)} samples...")

    pred_str  = []
    raw_resps = []
    few_shot_str = FEW_SHOT_EXAMPLES if not args.no_fewshot else ""
    for text in tqdm(eval_texts):
        pred, raw = run_improved_prompting(model, tokenizer, device, text, use_cot=args.cot)
        pred_str.append(pred)
        raw_resps.append(raw)

    # ── Evaluate ───────────────────────────────────────────────────────────
    acc = accuracy_score(y_true_str, pred_str)
    p, r, f, _ = precision_recall_fscore_support(
        y_true_str, pred_str, pos_label="Insufficient",
        average="binary", zero_division=0)

    print(f"\n====== Improved One-step Prompting ({mode}) — Test Split ======")
    print(f"  Accuracy:             {acc*100:.2f}%")
    print(f"  Precision (Omission): {p*100:.2f}%")
    print(f"  Recall    (Omission): {r*100:.2f}%")
    print(f"  F1        (Omission): {f*100:.2f}%")

    # Prediction distribution
    from collections import Counter
    dist = Counter(pred_str)
    print(f"\n  Prediction distribution: {dict(dist)}")

    # Show a few examples
    print("\n  Sample predictions (first 5):")
    for i in range(min(5, len(pred_str))):
        correct = "✓" if pred_str[i] == y_true_str[i] else "✗"
        print(f"    [{correct}] Gold={y_true_str[i]:13s} Pred={pred_str[i]}")
        print(f"         Raw: {raw_resps[i][:80]}")

    tag = "cot" if args.cot else ("improved_fewshot" if not args.no_fewshot else "improved_zeroshot")
    results = {
        "method": f"Improved One-step Prompting ({mode}, {args.model_name})",
        "eval_split": "test (held-out 50% of dev)",
        "eval_size": len(sampled_items),
        "verify_accuracy":    float(acc),
        "verify_f1_omission": float(f),
        "verify_precision":   float(p),
        "verify_recall":      float(r),
        "prediction_dist": dict(dist),
    }
    out = OUT_DIR / f"improved_onestep_{tag}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
