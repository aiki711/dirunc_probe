#!/usr/bin/env python3
"""
Implementation Plan — Baseline Strengthening

Goal: Add 3 additional baselines to comprehensive_comparison.png
  A. BERT fine-tune (bert-base-uncased, trained on natural_train.jsonl)
  B. Logit-based   (Gemma-2-2b-it P(Yes) without greedy decoding)
  C. TF-IDF + LR   (surface text features)
  
Evaluation: held-out TEST split (dev_test_indices.npy)
Task:        binary sufficiency detection (missing=Insufficient, filled=Sufficient)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASELINE A: BERT fine-tune
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
Input : dialogue text (missing version or filled version)
Label : 1=missing (Insufficient), 0=filled (Sufficient)

Model : bert-base-uncased (110M params)
        BertForSequenceClassification with [CLS] pooling

Training:
  - data  : natural_train.jsonl (14580 * 2 = 29160 samples incl. filled)
  - epoch : 3
  - lr    : 2e-5, AdamW, warmup_ratio=0.1
  - batch : 32
  - max_len: 512 tokens

Evaluation:
  - data  : test split (dev_test_indices.npy) → same balanced 300 samples
  - metric: Accuracy, F1 (Omission/Insufficient)

Why this matters:
  If linear probe (88% F1) > BERT fine-tune
  → Gemma's layer-26 representations are more informative for this task
    than supervised text-level features, even with full fine-tuning
  → This validates the "representation space access" hypothesis
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASELINE B: Same-model Logit-based (Gemma-2-2b-it)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
Input : same dialogue text + fixed question prompt
Prompt: "Is information missing from the last user utterance? Yes or No:"
Output: P(Yes) from the model's first decoded token logits

Threshold: calibrate on cal split to maximize F1

Difference from current prompting baselines:
  - Current one-step: generate "Sufficient"/"Insufficient" greedily
  - Logit-based     : read log-softmax P(Yes) directly (no generation)
  → Less stochastic, uses raw model confidence as the uncertainty signal

Why this matters:
  If linear probe > logit-based (same model):
  → Internal representations (layer 26) contain more discriminative signal
    than what is reflected in the model's output probability distribution
  → Supports "hidden states carry implicit uncertainty information"
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BASELINE C: TF-IDF + Logistic Regression
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
Input : raw dialogue text → TF-IDF bag of words (max 5000 features)
Label : same binary label
Model : LogisticRegression(C=1.0, max_iter=1000)

Training: natural_train.jsonl
Eval    : test split

This is the simplest possible text-based baseline.
If probe >> TF-IDF: shows pattern is semantic, not lexical
If BERT ~ TF-IDF: surface features are saturated
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# File Structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
scratch/
  baseline_bert.py        ← BERT fine-tune (most effort, 1-2h)
  baseline_logit.py       ← Logit-based   (moderate, ~2h inference)
  baseline_tfidf.py       ← TF-IDF + LR   (fast, ~10min)
  plot_full_comparison.py ← Regenerate comprehensive figure with all baselines

runs/identify_verify_comparison/
  bert_results.json
  logit_results.json
  tfidf_results.json
  comprehensive_comparison_v2.png   ← Final figure with 5+ baselines
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expected result table (hypothesis)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
Method                    | Verify Acc | Verify F1 (Omission)
--------------------------+------------+---------------------
Probing (Ours, Layer 26)  |   82.2%    |     88.4%     ← target to outperform
One-step Prompting        |   53.7%    |     66.7%
Identify-then-Verify      |   73.0%    |     83.4%
BERT fine-tune (A)        |     ?      |       ?        ← hypothesis: 75-85%
Logit-based (B)           |     ?      |       ?        ← hypothesis: 60-75%
TF-IDF + LR (C)           |     ?      |       ?        ← hypothesis: 55-65%
Random baseline           |   50.0%    |     66.7%
"""

if __name__ == "__main__":
    print("This is a planning document. Run baseline_bert.py, baseline_logit.py, baseline_tfidf.py separately.")
