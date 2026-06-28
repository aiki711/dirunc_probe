#!/usr/bin/env python3
"""
Evaluate Slot-wise Logit Prompting on the exact same balanced slot-level splits as Probing.
This ensures a 100% fair F1 comparison where the random baseline for both is 50%.
Saves to runs/identify_verify_comparison/slotwise_logit_results_balanced.json.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import precision_recall_fscore_support
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")

ALL_SLOTS  = ["who", "what", "when", "where", "how"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]
DIRS       = ["who", "what", "when", "where", "why", "how", "which"]
ROLE_TO_DIR = {
    "Agent": "who", "Theme": "what", "Location": "where", "Source": "where",
    "Goal": "where", "Time": "when", "Manner": "how"
}

SLOT_META = {
    "who": {"description": "WHO (the person or agent making the request)"},
    "what": {"description": "WHAT (the specific item or service being requested)"},
    "when": {"description": "WHEN (the time, day, or date the action is needed)"},
    "where": {"description": "WHERE (the location, area, origin, or destination)"},
    "how": {"description": "HOW (manner, price range, or quantity)"},
}

def build_slot_prompt(slot: str, text: str) -> str:
    desc = SLOT_META[slot]["description"]
    return f'Text: "{text}"\nIs {desc} MISSING from the last user utterance? Answer Yes or No:'

def get_yes_prob(model, tokenizer, device, prompt_text: str, yes_id: int, no_id: int) -> float:
    messages = [{"role": "user", "content": prompt_text}]
    try:
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    except:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs   = model(input_ids=input_ids)
        logits    = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        p_yes = log_probs[yes_id].exp().item()
        p_no  = log_probs[no_id].exp().item()
    return p_yes / (p_yes + p_no + 1e-9)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading dev data...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32 = load_s32()
    dev_pairs = s32.PairedDirUncDataset(dev_rows).pairs
    
    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    # Build slot→indices maps for TEST split
    test_slot_missing = {s: [] for s in ALL_SLOTS}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    # Load thresholds from slotwise_logit_results.json
    sw_meta = json.loads(Path(OUT_DIR / "slotwise_logit_results.json").read_text())
    sw_thresholds = sw_meta["per_slot_thresholds"]

    # Load Gemma
    print("Loading google/gemma-2-2b-it...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    # Isolated balanced evaluation per slot
    eval_slots = ["who", "when", "how", "what", "where"]
    slot_test_f1 = {}
    slot_test_p  = {}
    slot_test_r  = {}

    random.seed(42)  # MUST match random.seed(42) in get_probing_slot_metrics_v3.py exactly!
    for slot in eval_slots:
        pos_idxs = test_slot_missing.get(slot, [])
        n_eval   = min(50, len(pos_idxs))
        
        # sample exact same indices as probing
        eval_pos = random.sample(pos_idxs, n_eval)
        eval_neg = random.sample(test_filled_idx, n_eval)
        
        print(f"\nEvaluating slot '{slot}' (n_eval={n_eval} pos / {n_eval} neg)...")
        
        probs_pos = []
        for idx in tqdm(eval_pos, desc=f"Pos {slot}"):
            pair = dev_pairs[idx]
            text = pair.get("missing_text", pair.get("text", ""))
            prompt = build_slot_prompt(slot, text)
            probs_pos.append(get_yes_prob(model, tokenizer, device, prompt, yes_id, no_id))
            
        probs_neg = []
        for idx in tqdm(eval_neg, desc=f"Neg {slot}"):
            pair = dev_pairs[idx]
            text = pair.get("filled_text", pair.get("text", ""))
            prompt = build_slot_prompt(slot, text)
            probs_neg.append(get_yes_prob(model, tokenizer, device, prompt, yes_id, no_id))

        probs = np.concatenate([probs_pos, probs_neg])
        y_true = np.array([1]*n_eval + [0]*n_eval)
        y_pred = (probs >= sw_thresholds[slot]).astype(int)

        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0)
        
        slot_test_f1[slot] = float(f)
        slot_test_p[slot]  = float(p)
        slot_test_r[slot]  = float(r)
        print(f"Slot {slot:<6} | P: {p*100:6.2f}% | R: {r*100:6.2f}% | F1: {f*100:6.2f}%")

    # Save results
    balanced_results = {
        "method": "Slot-wise Logit Prompting (5-shot, per-slot threshold, google/gemma-2-2b-it) - Balanced Split",
        "eval_split": "slot-level binary-balanced (50 pos / 50 neg per slot)",
        "slot_f1": slot_test_f1,
        "slot_precision": slot_test_p,
        "slot_recall": slot_test_r,
    }
    
    out_path = OUT_DIR / "slotwise_logit_results_balanced.json"
    out_path.write_text(json.dumps(balanced_results, indent=2, ensure_ascii=False))
    print(f"\nSaved balanced slot-wise logit results to {out_path}")

if __name__ == "__main__":
    main()
