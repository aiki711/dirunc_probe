#!/usr/bin/env python3
"""
Evaluate all models on a binary-balanced test set (150 Sufficient, 150 Insufficient).
This eliminates the "Yes-bias" anomaly in F1 score and provides an unbiased comparison.

Binary-Balanced Eval Set Design:
  - 150 Negatives (Sufficient/filled): sampled from filled versions in the test split.
  - 150 Positives (Insufficient/missing): sampled from missing versions in the test split
    (stratified across the 5 slots: who, what, when, where, how).
"""
import json, random, os, sys, warnings
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM, BertForSequenceClassification
import importlib.util
from tqdm import tqdm

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Slots
ALL_SLOTS = ["who", "what", "when", "where", "how"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]
ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
}
DIRS = ["who", "what", "when", "where", "why", "how", "which"]

class DummyZeroClassifier:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0
        return out


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Prompt templates
SLOT_META = {
    "who": "WHO (the person or agent making the request, e.g. a specific person's name if needed)",
    "what": "WHAT (the specific item or service being requested, e.g. restaurant type, hotel category, attraction name)",
    "when": "WHEN (the time, day, or date the action is needed, e.g. tonight, Monday, 7pm, next week)",
    "where": "WHERE (the location, area, origin, or destination, e.g. city centre, Cambridge, north part of town)",
    "how": "HOW (manner, price range, or quantity — e.g. cheap, moderately priced, for 2 people, 3 nights)",
}

# 3-shot prompt for global sufficiency
GLOBAL_PROMPT = """You are an expert at evaluating task-oriented dialogues.
Your task: Determine if the last user utterance provides SUFFICIENT information to complete the user's request, or if important information is MISSING (Insufficient).

--- Examples ---
Text: "[system]: I can help you find a hotel. [user]: I need a cheap hotel for 3 nights starting Friday."
Is information missing? Yes (location/area is missing)

Text: "[system]: What restaurant are you looking for? [user]: I want an Italian restaurant in the city centre for 2 people tonight."
Is information missing? No (all necessary slots provided)

Text: "[system]: Where would you like to travel? [user]: I need a train from Cambridge to London."
Is information missing? Yes (departure time or day is missing)

--- Now evaluate this dialogue ---
Text: "{text}"
Is information missing from the last user utterance? Answer Yes or No:"""

# Few-shot slot prompt template
def build_slot_prompt(slot: str, text: str) -> str:
    desc = SLOT_META[slot]
    return f"""Determine if the following information type is MISSING (not explicitly stated) from the last user utterance.
Information type: {desc}

Text: "{text}"
Is {desc} MISSING from the last user utterance? Answer Yes or No:"""


def get_yes_prob(model, tokenizer, device, prompt_text: str, yes_id: int, no_id: int) -> float:
    try:
        input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], return_tensors="pt").to(device)
    except:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs   = model(input_ids=input_ids)
        logits    = outputs.logits[0, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        p_yes = log_probs[yes_id].exp().item()
        p_no  = log_probs[no_id].exp().item()
    return p_yes / (p_yes + p_no + 1e-9)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load data & align dev pairs ────────────────────────────────────
    print("Loading datasets...")
    dev_rows   = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    s32 = load_s32()
    dev_pairs   = s32.PairedDirUncDataset(dev_rows).pairs
    train_pairs = s32.PairedDirUncDataset(train_rows).pairs

    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    # ── 2. Build binary-balanced eval set (test split) ────────────────────
    # Negatives: 150 filled items from test indices
    # Positives: 150 missing items from test indices (stratified across slots)
    test_slot_missing = {s: [] for s in ALL_SLOTS}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    # Sample negatives
    random.seed(42)
    neg_sampled_idx = random.sample(test_filled_idx, 150)
    neg_items = [(idx, "filled", "None") for idx in neg_sampled_idx]

    # Sample positives
    # "who" is scarce, take all available up to 30
    who_pool = test_slot_missing["who"]
    who_sampled = random.sample(who_pool, min(len(who_pool), 30))
    pos_items = [(idx, "missing", "who") for idx in who_sampled]
    
    # Distribute the remaining among other 4 slots
    rem_to_sample = 150 - len(pos_items)
    slots_left = ["what", "when", "where", "how"]
    per_slot = rem_to_sample // len(slots_left)
    for s in slots_left:
        pool = test_slot_missing[s]
        sampled = random.sample(pool, per_slot)
        pos_items.extend([(idx, "missing", s) for idx in sampled])
    
    # Adjust if short by rounding errors
    if len(pos_items) < 150:
        needed = 150 - len(pos_items)
        extra = random.sample(test_slot_missing["when"], needed)
        pos_items.extend([(idx, "missing", "when") for idx in extra])

    eval_items = neg_items + pos_items
    random.shuffle(eval_items)
    print(f"Balanced evaluation set built: {len(eval_items)} items")
    print(f"  Positives (missing): {sum(1 for _, c, _ in eval_items if c == 'missing')}")
    print(f"  Negatives (filled) : {sum(1 for _, c, _ in eval_items if c == 'filled')}")

    # Build evaluation texts & ground truth
    eval_texts = []
    eval_y_verify = [] # 1 = Insufficient, 0 = Sufficient
    eval_y_role   = [] # 'who', 'what', etc.
    for idx, cond, slot_label in eval_items:
        pair = dev_pairs[idx]
        text = pair["missing_text"] if cond == "missing" else pair["filled_text"]
        eval_texts.append(text)
        eval_y_verify.append(1 if cond == "missing" else 0)
        eval_y_role.append(slot_label)

    eval_y_verify = np.array(eval_y_verify)

    # ── 3. Evaluate TF-IDF + LR ───────────────────────────────────────────
    print("\n--- Evaluating TF-IDF + LR Baseline ---")
    train_texts  = [r["text"] for r in train_rows if "text" in r]
    train_labels = [1 if r["condition"] == "missing" else 0 for r in train_rows if "text" in r]
    
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vec.fit_transform(train_texts)
    X_eval_tfidf  = vec.transform(eval_texts)

    clf_tfidf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
    clf_tfidf.fit(X_train_tfidf, train_labels)
    pred_tfidf = clf_tfidf.predict(X_eval_tfidf)
    tfidf_acc = accuracy_score(eval_y_verify, pred_tfidf)
    tfidf_f1  = f1_score(eval_y_verify, pred_tfidf)
    print(f"  TF-IDF Verify Acc: {tfidf_acc*100:.2f}% | F1: {tfidf_f1*100:.2f}%")

    # ── 4. Evaluate BERT Fine-tune ────────────────────────────────────────
    print("\n--- Evaluating BERT Fine-tune Baseline ---")
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    bert_model.load_state_dict(torch.load(OUT_DIR / "bert_best.pt", map_location=device))
    bert_model = bert_model.to(device)
    bert_model.eval()

    from transformers import BertTokenizerFast
    tokenizer_bert = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    pred_bert = []
    with torch.no_grad():
        for text in eval_texts:
            enc = tokenizer_bert(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(device)
            out = bert_model(**enc).logits
            pred_bert.append(out.argmax(dim=-1).item())

    bert_acc = accuracy_score(eval_y_verify, pred_bert)
    bert_f1  = f1_score(eval_y_verify, pred_bert)
    print(f"  BERT Verify Acc: {bert_acc*100:.2f}% | F1: {bert_f1*100:.2f}%")

    # ── 5. Evaluate Probing (Ours) ────────────────────────────────────────
    print("\n--- Evaluating Probing Baseline ---")
    train_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer26_train.pt", map_location="cpu")
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()
    train_y = np.zeros((train_f_hs.shape[0], 7))
    for i, pair in enumerate(train_pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            d = DIRS.index(ROLE_TO_DIR[role])
            train_y[i, d] = 1.0

    # Fit 7 binary probes
    probes = []
    for d in range(7):
        X = np.concatenate([train_f_hs[:, d, :], train_m_hs[:, d, :]], axis=0)
        y = np.concatenate([np.zeros(train_f_hs.shape[0]), train_y[:, d]], axis=0)
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight='balanced')
            clf.fit(X, y)
        probes.append(clf)

    # Thresholds calibrated on CAL split (same as v3)
    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()
    cal_slot_missing, cal_filled_idx = {s: [] for s in DIRS}, []
    for i in cal_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            cal_slot_missing[ROLE_TO_DIR[role]].append(i)
            cal_filled_idx.append(i)

    thresholds = []
    for d, slot in enumerate(DIRS):
        pos_idxs = cal_slot_missing.get(slot, [])
        if not pos_idxs:
            thresholds.append(0.5)
            continue
        n_cal = min(50, len(pos_idxs))
        cal_pos = random.sample(pos_idxs, n_cal)
        cal_neg = random.sample(cal_filled_idx, n_cal)
        X_cal = np.concatenate([dev_m_hs[cal_pos, d, :], dev_f_hs[cal_neg, d, :]], axis=0)
        y_cal = np.array([1]*n_cal + [0]*n_cal)
        probs_cal = probes[d].predict_proba(X_cal)[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            f = f1_score(y_cal, (probs_cal >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)

    # Evaluate on the new binary-balanced set using OR integration
    pred_probe = []
    for idx, cond, _ in eval_items:
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        any_fire = False
        for d, slot in enumerate(DIRS):
            if slot in ALL_SLOTS:
                p = probes[d].predict_proba(hs[d:d+1, :])[0, 1]
                if p >= thresholds[d]:
                    any_fire = True
                    break
        pred_probe.append(1 if any_fire else 0)

    probe_acc = accuracy_score(eval_y_verify, pred_probe)
    probe_f1  = f1_score(eval_y_verify, pred_probe)
    print(f"  Probing Verify Acc: {probe_acc*100:.2f}% | F1: {probe_f1*100:.2f}%")

    # ── 6. Evaluate Logit P(Yes) & Slot-wise Logit & One-step Prompting ─────
    print("\nLoading Gemma-2-2b-it for prompting baselines...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id  = tokenizer.encode("No",  add_special_tokens=False)[0]

    # Evaluate Logit P(Yes)
    print("  Evaluating Logit P(Yes)...")
    # Calibrate global threshold on CAL (same as baseline_logit.py)
    # Cal split build:
    cal_items = neg_items = [] # placeholder
    # Load calibrated threshold from slotwise_logit or baseline_logit.py
    logit_meta = json.loads(Path(OUT_DIR / "logit_results.json").read_text())
    best_global_t = logit_meta["cal_threshold"]

    pred_logit = []
    for text in tqdm(eval_texts, desc="Logit P(Yes)"):
        prob = get_yes_prob(model, tokenizer, device, GLOBAL_PROMPT.format(text=text), yes_id, no_id)
        pred_logit.append(1 if prob >= best_global_t else 0)

    logit_acc = accuracy_score(eval_y_verify, pred_logit)
    logit_f1  = f1_score(eval_y_verify, pred_logit)
    print(f"  Logit P(Yes) Verify Acc: {logit_acc*100:.2f}% | F1: {logit_f1*100:.2f}%")

    # Evaluate Slot-wise Logit
    print("  Evaluating Slot-wise Logit Prompting...")
    # Load per-slot thresholds from slotwise_logit_results.json
    sw_meta = json.loads(Path(OUT_DIR / "slotwise_logit_results.json").read_text())
    sw_thresholds = sw_meta["per_slot_thresholds"]

    pred_sw = []
    # Evaluate each text slot-by-slot
    for text in tqdm(eval_texts, desc="Slot-wise Logit"):
        any_fire = False
        for slot in ALL_SLOTS:
            prompt = build_slot_prompt(slot, text)
            prob = get_yes_prob(model, tokenizer, device, prompt, yes_id, no_id)
            if prob >= sw_thresholds[slot]:
                any_fire = True
                break
        pred_sw.append(1 if any_fire else 0)

    sw_acc = accuracy_score(eval_y_verify, pred_sw)
    sw_f1  = f1_score(eval_y_verify, pred_sw)
    print(f"  Slot-wise Logit Verify Acc: {sw_acc*100:.2f}% | F1: {sw_f1*100:.2f}%")

    # Evaluate One-step Prompting
    print("  Evaluating One-step Prompting...")
    from scratch.improved_onestep_prompting import FEW_SHOT_EXAMPLES, PROMPT_TEMPLATE, run_improved_prompting
    pred_os = []
    for text in tqdm(eval_texts, desc="One-step"):
        pred, _ = run_improved_prompting(model, tokenizer, device, text, use_cot=False)
        pred_os.append(1 if pred == "Insufficient" else 0)

    os_acc = accuracy_score(eval_y_verify, pred_os)
    os_f1  = f1_score(eval_y_verify, pred_os)
    print(f"  One-step Verify Acc: {os_acc*100:.2f}% | F1: {os_os_f1*100 if 'os_os_f1' in locals() else os_f1*100:.2f}%")

    # ── 7. Save balanced results ──────────────────────────────────────────
    results = {
        "eval_split": "test split (binary-balanced, 150 Sufficient vs 150 Insufficient)",
        "metrics": {
            "BERT fine-tune": {"accuracy": float(bert_acc), "f1": float(bert_f1)},
            "Probing (Ours)": {"accuracy": float(probe_acc), "f1": float(probe_f1)},
            "Logit P(Yes)": {"accuracy": float(logit_acc), "f1": float(logit_f1)},
            "Slot-wise Logit": {"accuracy": float(sw_acc), "f1": float(sw_f1)},
            "TF-IDF + LR": {"accuracy": float(tfidf_acc), "f1": float(tfidf_f1)},
            "One-step Prompting": {"accuracy": float(os_acc), "f1": float(os_f1)},
            "Random / Dummy": {"accuracy": 0.50, "f1": 0.667}
        }
    }
    out_file = OUT_DIR / "binary_balanced_comparison_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved balanced comparison results to {out_file}")


if __name__ == "__main__":
    main()
