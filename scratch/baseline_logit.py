#!/usr/bin/env python3
"""
Baseline B: Same-model Logit-based uncertainty detection.

Uses Gemma-2-2b-it's direct logit output for a Yes/No question,
WITHOUT greedy generation (unlike current prompting baselines).

Prompt: "Is information missing from the last user utterance? Answer Yes or No:"
Extract: P(Yes) / P(No) from the first token logits

Calibrate threshold on CAL split, evaluate on TEST split.

Purpose: Shows whether internal representations (linear probe)
         contain more discriminative signal than the model's own
         output probability distribution over Yes/No tokens.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import importlib.util
from tqdm import tqdm

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_yes_prob(model, tokenizer, device, text, yes_id, no_id):
    """Return P(Yes) for a given text, using first predicted token logits."""
    prompt = f"""Text:
\"\"\"{text}\"\"\"

Is information missing from the last user utterance? Answer Yes or No:"""
    messages = [{"role": "user", "content": prompt}]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt").to(device)
    except Exception:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits  = outputs.logits[0, -1, :]        # Last token logits
        log_probs = torch.log_softmax(logits, dim=-1)
        p_yes = log_probs[yes_id].exp().item()
        p_no  = log_probs[no_id].exp().item()
    # Normalized P(Yes)
    denom = p_yes + p_no + 1e-9
    return p_yes / denom


def main():
    MODEL_NAME = "google/gemma-2-2b-it"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load dev pairs for cal and test splits ─────────────────────────
    print("Loading dev pairs...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32       = load_s32()
    dev_ds    = s32.PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt",
                           map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    ROLE_TO_DIR = {
        "Agent":"who","Theme":"what","Location":"where",
        "Source":"where","Goal":"where","Time":"when","Manner":"how",
    }
    ALL_CLASSES = ["who","what","when","where","how","None"]

    def build_eval_set(indices, seed=42):
        slot_missing = {s: [] for s in ["who","what","when","where","how"]}
        filled_idx   = []
        for i in indices:
            pair = dev_pairs[i]
            role = pair.get("case_role","")
            if role in ROLE_TO_DIR:
                slot_missing[ROLE_TO_DIR[role]].append(i)
                filled_idx.append(i)
        groups = {c: [] for c in ALL_CLASSES}
        for slot, idxs in slot_missing.items():
            if slot in ALL_CLASSES:
                for i in idxs:
                    groups[slot].append((i, "missing"))
        for i in filled_idx:
            groups["None"].append((i, "filled"))
        random.seed(seed)
        num_per = max(1, 300 // 6)
        items = []
        for c in ALL_CLASSES:
            idxs = groups[c]
            items.extend(random.sample(idxs, min(len(idxs), num_per)))
        return items

    cal_items  = build_eval_set(cal_indices,  seed=0)
    test_items = build_eval_set(test_indices, seed=42)
    print(f"  Cal:  {len(cal_items)} items")
    print(f"  Test: {len(test_items)} items")

    # ── 2. Load Gemma model ───────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    # Get Yes / No token IDs
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]
    print(f"  Yes token ID: {yes_id} ('{tokenizer.decode([yes_id])}')")
    print(f"  No  token ID: {no_id}  ('{tokenizer.decode([no_id])}')")

    # ── 3. Get P(Yes) on CAL split for threshold calibration ─────────────
    print("\nComputing P(Yes) on CAL split for calibration...")
    cal_probs  = []
    cal_labels = []
    for idx, cond in tqdm(cal_items):
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        prob = get_yes_prob(model, tokenizer, device, text, yes_id, no_id)
        cal_probs.append(prob)
        cal_labels.append(1 if cond == "missing" else 0)

    # Calibrate threshold on CAL
    from sklearn.metrics import f1_score as _f1
    cal_probs_arr  = np.array(cal_probs)
    cal_labels_arr = np.array(cal_labels)
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (cal_probs_arr >= t).astype(int)
        f = _f1(cal_labels_arr, preds, pos_label=1, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    print(f"  Best threshold (cal): {best_t:.3f}  cal_F1={best_f*100:.2f}%")

    # ── 4. Get P(Yes) on TEST split for final evaluation ─────────────────
    print("\nComputing P(Yes) on TEST split...")
    test_probs  = []
    test_y_str  = []
    for idx, cond in tqdm(test_items):
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        prob = get_yes_prob(model, tokenizer, device, text, yes_id, no_id)
        test_probs.append(prob)
        test_y_str.append("Insufficient" if cond == "missing" else "Sufficient")

    test_probs_arr = np.array(test_probs)
    test_preds_bin = (test_probs_arr >= best_t).astype(int)
    test_pred_str  = ["Insufficient" if p == 1 else "Sufficient" for p in test_preds_bin]

    acc = accuracy_score(test_y_str, test_pred_str)
    p, r, f, _ = precision_recall_fscore_support(
        test_y_str, test_pred_str, pos_label="Insufficient",
        average="binary", zero_division=0)

    print(f"\n====== Logit-based (P(Yes)) Results (Test Split) ======")
    print(f"  Threshold:            {best_t:.3f}")
    print(f"  Accuracy:             {acc*100:.2f}%")
    print(f"  Precision (Omission): {p*100:.2f}%")
    print(f"  Recall    (Omission): {r*100:.2f}%")
    print(f"  F1        (Omission): {f*100:.2f}%")

    results = {
        "method": f"Logit-based P(Yes) ({MODEL_NAME})",
        "eval_split": "test (held-out 50% of dev)",
        "eval_size": len(test_items),
        "cal_threshold": float(best_t),
        "verify_accuracy":    float(acc),
        "verify_f1_omission": float(f),
        "verify_precision":   float(p),
        "verify_recall":      float(r),
    }
    out = OUT_DIR / "logit_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
