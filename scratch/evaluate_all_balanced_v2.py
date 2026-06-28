#!/usr/bin/env python3
"""
Evaluate all 5 key methods (BERT, Probing Multinomial, Probing Binary OR, TF-IDF, Logit P(Yes), Slot-wise Logit)
on the binary-balanced test split, computing Accuracy, Precision, Recall, and F1.
Saves to runs/identify_verify_comparison/binary_balanced_comparison_results.json.
Uses Layer 16 for all Probing classifiers.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
ALL_SLOTS = ["who", "what", "when", "where", "how"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]
ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
}
DIRS = ["who", "what", "when", "where", "why", "how", "which"]
GLOBAL_PROMPT = "Is there any missing information? Yes or No: {text}"

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

# Prompts for slot-wise logit
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

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1, zero_division=0)
    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f)
    }

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

    # Load layer 16 dev cache to get N_dev
    dev_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_dev.pt", map_location="cpu")
    N_dev = dev_cache["f_hs"].shape[0]
    if len(dev_pairs) > N_dev:
        dev_pairs = dev_pairs[:N_dev]

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    # Reconstruct the exact same binary-balanced test items (seed 42)
    test_slot_missing = {s: [] for s in ALL_SLOTS}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    random.seed(42)
    neg_sampled_idx = random.sample(test_filled_idx, 150)
    neg_items = [(idx, "filled", "None") for idx in neg_sampled_idx]

    who_pool = test_slot_missing["who"]
    who_sampled = random.sample(who_pool, min(len(who_pool), 30))
    pos_items = [(idx, "missing", "who") for idx in who_sampled]
    
    rem_to_sample = 150 - len(pos_items)
    slots_left = ["what", "when", "where", "how"]
    per_slot = rem_to_sample // len(slots_left)
    for s in slots_left:
        pool = test_slot_missing[s]
        sampled = random.sample(pool, per_slot)
        pos_items.extend([(idx, "missing", s) for idx in sampled])
    
    if len(pos_items) < 150:
        needed = 150 - len(pos_items)
        extra = random.sample(test_slot_missing["when"], needed)
        pos_items.extend([(idx, "missing", "when") for idx in extra])

    eval_items = neg_items + pos_items
    random.shuffle(eval_items)

    eval_y_verify = [1 if cond == "missing" else 0 for _, cond, _ in eval_items]
    eval_y_verify = np.array(eval_y_verify)

    eval_texts = []
    for idx, cond, _ in eval_items:
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        eval_texts.append(text)

    # Dictionary to hold all final metrics
    final_metrics = {}

    # ── 2. Evaluate BERT Fine-tune ────────────────────────────────────────
    print("\n--- Evaluating BERT Fine-tune ---")
    from transformers import BertForSequenceClassification
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

    final_metrics["BERT fine-tune"] = compute_metrics(eval_y_verify, pred_bert)
    print(f"  BERT: {final_metrics['BERT fine-tune']}")

    # ── 3. Evaluate TF-IDF + LR ───────────────────────────────────────────
    print("\n--- Evaluating TF-IDF + LR ---")
    from sklearn.feature_extraction.text import TfidfVectorizer
    train_texts = []
    train_y_tfidf = []
    for pair in train_pairs:
        train_texts.append(pair.get("filled_text", pair.get("text", "")))
        train_y_tfidf.append(0)
        train_texts.append(pair.get("missing_text", pair.get("text", "")))
        train_y_tfidf.append(1)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_eval_tfidf  = vectorizer.transform(eval_texts)

    lr_tfidf = LogisticRegression(C=1.0, random_state=42)
    lr_tfidf.fit(X_train_tfidf, train_y_tfidf)
    pred_tfidf = lr_tfidf.predict(X_eval_tfidf)

    final_metrics["TF-IDF + LR"] = compute_metrics(eval_y_verify, pred_tfidf)
    print(f"  TF-IDF: {final_metrics['TF-IDF + LR']}")

    # Load layer 16 train cache for Probing
    print("\nLoading Layer 16 train cache...")
    train_cache = torch.load(CACHE_DIR / "final_token_aligned_soft_layer16_train.pt", map_location="cpu")
    train_f_hs  = train_cache["f_hs"].float().numpy()
    train_m_hs  = train_cache["m_hs"].float().numpy()
    N_train = train_f_hs.shape[0]

    # Labels for train
    train_y = np.zeros((N_train, 7))
    for i, pair in enumerate(train_pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            d = DIRS.index(ROLE_TO_DIR[role])
            train_y[i, d] = 1.0

    # ── 4. Evaluate Probing (Ours - Multinomial) ──────────────────────────
    print("\n--- Evaluating Probing (Ours - Multinomial) ---")
    X_multi_list = []
    y_multi_list = []
    for i, pair in enumerate(train_pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            slot = ROLE_TO_DIR[role]
            X_multi_list.append(train_m_hs[i].mean(axis=0))
            y_multi_list.append(slot)
            X_multi_list.append(train_f_hs[i].mean(axis=0))
            y_multi_list.append("None")
    
    X_multi = np.array(X_multi_list)
    y_multi = np.array(y_multi_list)

    pca = PCA(n_components=256, random_state=42)
    X_pca = pca.fit_transform(X_multi)
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_pca, y_multi)
    
    multi_clf = LogisticRegression(max_iter=500, C=1.0, random_state=42, class_weight='balanced')
    multi_clf.fit(X_lda, y_multi)

    dev_f_hs = dev_cache["f_hs"].float().numpy()
    dev_m_hs = dev_cache["m_hs"].float().numpy()

    pred_multi = []
    for idx, cond, _ in eval_items:
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        hs_avg = hs.mean(axis=0, keepdims=True)
        hs_pca = pca.transform(hs_avg)
        hs_lda = lda.transform(hs_pca)
        pred_class = multi_clf.predict(hs_lda)[0]
        pred_multi.append(0 if pred_class == "None" else 1)

    final_metrics["Probing (Ours - Multinomial)"] = compute_metrics(eval_y_verify, pred_multi)
    print(f"  Multinomial Probing: {final_metrics['Probing (Ours - Multinomial)']}")

    # ── 5. Evaluate Probing (Ours - Binary OR) ────────────────────────────
    print("\n--- Evaluating Probing (Ours - Binary OR) ---")
    probes = []
    for d in range(7):
        X = np.concatenate([train_f_hs[:, d, :], train_m_hs[:, d, :]], axis=0)
        y = np.concatenate([np.zeros(N_train), train_y[:, d]], axis=0)
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
        else:
            clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, class_weight='balanced')
            clf.fit(X, y)
        probes.append(clf)

    # Calibrate thresholds on CAL split
    cal_slot_missing, cal_filled_idx = {s: [] for s in DIRS}, []
    for i in cal_indices:
        if i >= N_dev: continue
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
            f = f1_score = precision_recall_fscore_support(y_cal, (probs_cal >= t).astype(int), average='binary', pos_label=1, zero_division=0)[2]
            if f > best_f1:
                best_f1, best_t = f, t
        thresholds.append(best_t)

    pred_or = []
    for idx, cond, _ in eval_items:
        hs = dev_m_hs[idx] if cond == "missing" else dev_f_hs[idx]
        any_fire = False
        for d, slot in enumerate(DIRS):
            if slot in ALL_SLOTS:
                p = probes[d].predict_proba(hs[d:d+1, :])[0, 1]
                if p >= thresholds[d]:
                    any_fire = True
                    break
        pred_or.append(1 if any_fire else 0)

    final_metrics["Probing (Ours)"] = compute_metrics(eval_y_verify, pred_or)
    print(f"  Binary OR Probing: {final_metrics['Probing (Ours)']}")

    # ── 6. Evaluate Logit P(Yes) & Slot-wise Logit Prompting ──────────────
    print("\nLoading google/gemma-2-2b-it for prompting baselines...")
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
    # Load calibrated threshold from slotwise_logit or baseline_logit.py
    logit_meta = json.loads(Path(OUT_DIR / "slotwise_logit_results.json").read_text()) # slotwise has similar thresholds or let's use global threshold 0.5 as fallback, or from transcript:
    # From transcript: Logit P(Yes) F1 and Accuracy were computed with threshold
    # Since it collapses anyway, let's use the standard threshold 0.5, or global cal threshold = 0.52 (who), etc.
    # Actually, we can load threshold from transcript: cal threshold was 0.87 or similar. Let's calibrate threshold on CAL split here:
    # Calibrate Logit P(Yes)
    cal_valid_indices = [int(i) for i in cal_indices if i < N_dev]
    n_cal = min(50, len(cal_valid_indices))
    cal_sampled = random.sample(cal_valid_indices, n_cal)
    
    cal_probs_pos = []
    cal_probs_neg = []
    for i in cal_sampled:
        pair = dev_pairs[i]
        text_pos = pair.get("missing_text", pair.get("text", ""))
        text_neg = pair.get("filled_text", pair.get("text", ""))
        p_pos = get_yes_prob(model, tokenizer, device, GLOBAL_PROMPT.format(text=text_pos), yes_id, no_id)
        p_neg = get_yes_prob(model, tokenizer, device, GLOBAL_PROMPT.format(text=text_neg), yes_id, no_id)
        cal_probs_pos.append(p_pos)
        cal_probs_neg.append(p_neg)
    
    cal_probs = np.concatenate([cal_probs_pos, cal_probs_neg])
    y_cal = np.array([1]*n_cal + [0]*n_cal)
    best_logit_t, best_f = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f = precision_recall_fscore_support(y_cal, (cal_probs >= t).astype(int), average='binary', pos_label=1, zero_division=0)[2]
        if f > best_f:
            best_f, best_logit_t = f, t

    pred_logit = []
    for text in tqdm(eval_texts, desc="Logit P(Yes)"):
        prob = get_yes_prob(model, tokenizer, device, GLOBAL_PROMPT.format(text=text), yes_id, no_id)
        pred_logit.append(1 if prob >= best_logit_t else 0)

    final_metrics["Logit P(Yes)"] = compute_metrics(eval_y_verify, pred_logit)
    print(f"  Logit P(Yes): {final_metrics['Logit P(Yes)']}")

    # Evaluate Slot-wise Logit
    print("  Evaluating Slot-wise Logit Prompting...")
    # Load per-slot thresholds from slotwise_logit_results.json
    sw_meta = json.loads(Path(OUT_DIR / "slotwise_logit_results.json").read_text())
    sw_thresholds = sw_meta["per_slot_thresholds"]

    pred_sw = []
    for text in tqdm(eval_texts, desc="Slot-wise Logit"):
        any_fire = False
        for slot in ALL_SLOTS:
            prompt = build_slot_prompt(slot, text)
            prob = get_yes_prob(model, tokenizer, device, prompt, yes_id, no_id)
            if prob >= sw_thresholds[slot]:
                any_fire = True
                break
        pred_sw.append(1 if any_fire else 0)

    final_metrics["Slot-wise Logit"] = compute_metrics(eval_y_verify, pred_sw)
    print(f"  Slot-wise Logit: {final_metrics['Slot-wise Logit']}")

    # ── 7. Save balanced results ──────────────────────────────────────────
    results = {
        "eval_split": "test split (binary-balanced, 150 Sufficient vs 150 Insufficient)",
        "metrics": {
            "BERT fine-tune": final_metrics["BERT fine-tune"],
            "Probing (Ours - Multinomial)": final_metrics["Probing (Ours - Multinomial)"],
            "Probing (Ours)": final_metrics["Probing (Ours)"],
            "TF-IDF + LR": final_metrics["TF-IDF + LR"],
            "Logit P(Yes)": final_metrics["Logit P(Yes)"],
            "Slot-wise Logit": final_metrics["Slot-wise Logit"]
        }
    }
    
    out_file = OUT_DIR / "binary_balanced_comparison_results.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved balanced comparison results to {out_file}")

if __name__ == "__main__":
    main()
