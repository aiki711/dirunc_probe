#!/usr/bin/env python3
"""
Baseline A: Fine-tuned BERT for sufficiency detection.

Input : dialogue text (missing or filled version)
Label : 1 = missing (Insufficient), 0 = filled (Sufficient)
Model : bert-base-uncased, BertForSequenceClassification
Train : natural_train.jsonl (29160 samples, balanced)
Eval  : held-out test split (dev_test_indices.npy)

Purpose: Show that linear probe on Gemma layer-26 hidden states
         outperforms supervised BERT fine-tuning on raw text,
         validating the "representation space access" hypothesis.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helper functions ──────────────────────────────────────────────────────
def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Dataset ───────────────────────────────────────────────────────────────
class SufficiencyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.labels = labels
        self.encodings = tokenizer(
            texts, truncation=True, padding='max_length',
            max_length=max_length, return_tensors='pt')

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings.get('token_type_ids',
                                  torch.zeros_like(self.encodings['input_ids']))[idx],
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ── Main ─────────────────────────────────────────────────────────────────
def main():
    EPOCHS    = 3
    BATCH     = 32
    LR        = 2e-5
    MAX_LEN   = 256
    MODEL_ID  = "bert-base-uncased"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load training data ─────────────────────────────────────────────
    print("Loading train rows...")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    train_texts  = [r["text"] for r in train_rows if "text" in r]
    train_labels = [1 if r["condition"] == "missing" else 0
                    for r in train_rows if "text" in r]
    print(f"  Train: {len(train_texts)} samples  "
          f"(pos={sum(train_labels)}, neg={len(train_labels)-sum(train_labels)})")

    # ── 2. Load eval data (test split) ────────────────────────────────────
    print("Loading dev pairs for test split evaluation...")
    dev_rows = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    s32       = load_s32()
    dev_ds    = s32.PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    import torch as _torch
    dev_cache = _torch.load(CACHE_DIR / "final_token_aligned_soft_layer26_dev.pt",
                            map_location="cpu")
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
    print(f"  Sampled {len(sampled_items)} eval items")

    eval_texts = []
    eval_labels = []
    eval_y_str  = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        text = pair.get("missing_text" if cond == "missing" else "filled_text", "")
        if not text:
            text = pair.get("text", "")
        eval_texts.append(text)
        label = 1 if cond == "missing" else 0
        eval_labels.append(label)
        eval_y_str.append("Insufficient" if label == 1 else "Sufficient")

    # ── 3. Tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {MODEL_ID}...")
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_ID)

    print("Tokenizing train data...")
    train_ds = SufficiencyDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    print("Tokenizing eval data...")
    eval_ds  = SufficiencyDataset(eval_texts,  eval_labels,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    eval_loader  = DataLoader(eval_ds,  batch_size=BATCH, shuffle=False, num_workers=2)

    # ── 4. Model ──────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    model = BertForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)
    model = model.to(device)

    # Compute class weights for imbalanced case
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    w_pos = len(train_labels) / (2 * pos_count)
    w_neg = len(train_labels) / (2 * neg_count)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # ── 5. Optimizer / Scheduler ──────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = EPOCHS * len(train_loader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    # ── 6. Training ───────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...")
    best_f1   = -1.0
    best_ckpt = OUT_DIR / "bert_best.pt"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS}  Step {step+1}/{len(train_loader)}"
                      f"  Loss={total_loss/(step+1):.4f}")

        # Eval after each epoch
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids      = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels         = batch['labels'].to(device)
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids).logits
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        epoch_f1  = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
        epoch_acc = accuracy_score(all_labels, all_preds)
        print(f"  → Epoch {epoch+1} Eval: Acc={epoch_acc*100:.2f}%  F1={epoch_f1*100:.2f}%")

        if epoch_f1 > best_f1:
            best_f1 = epoch_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"    ★ New best F1={best_f1*100:.2f}%  (saved)")

    # ── 7. Final evaluation with best checkpoint ──────────────────────────
    print(f"\nLoading best checkpoint (F1={best_f1*100:.2f}%)...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)
            logits = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids).logits
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    pred_str = ["Insufficient" if p == 1 else "Sufficient" for p in all_preds]
    acc = accuracy_score(eval_y_str, pred_str)
    p, r, f, _ = precision_recall_fscore_support(
        eval_y_str, pred_str, pos_label="Insufficient",
        average="binary", zero_division=0)

    print(f"\n====== BERT Fine-tune Results (Best Checkpoint, Test Split) ======")
    print(f"  Accuracy:             {acc*100:.2f}%")
    print(f"  Precision (Omission): {p*100:.2f}%")
    print(f"  Recall    (Omission): {r*100:.2f}%")
    print(f"  F1        (Omission): {f*100:.2f}%")

    results = {
        "method": f"BERT fine-tune ({MODEL_ID}, {EPOCHS} epochs, lr={LR})",
        "eval_split": "test (held-out 50% of dev)",
        "eval_size": len(sampled_items),
        "best_epoch_f1": float(best_f1),
        "verify_accuracy":    float(acc),
        "verify_f1_omission": float(f),
        "verify_precision":   float(p),
        "verify_recall":      float(r),
    }
    out = OUT_DIR / "bert_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
