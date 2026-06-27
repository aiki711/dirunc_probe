#!/usr/bin/env python3
"""
Final comprehensive comparison figure with all baselines.

Reads results from:
  - runs/identify_verify_comparison/comparison_report_v2.json   (Probing)
  - runs/identify_verify_comparison/multilabel_prompting_results.json (Prompting)
  - runs/identify_verify_comparison/bert_results.json           (BERT fine-tune)
  - runs/identify_verify_comparison/tfidf_results.json          (TF-IDF)
  - runs/identify_verify_comparison/logit_results.json          (Logit)
  - runs/identify_verify_comparison/slotwise_logit_results.json (Slot-wise Logit)
  - runs/identify_verify_comparison/slot_metrics_test.json      (Per-slot probing)

Generates:
  - runs/identify_verify_comparison/comprehensive_comparison_v3.png
"""
import json, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
warnings.simplefilter('ignore')

OUT_DIR = Path("runs/identify_verify_comparison")

def load_json(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None

def main():
    # ── Load results ─────────────────────────────────────────────────────
    probe          = load_json(OUT_DIR / "comparison_report_v2.json")
    prompting      = load_json(OUT_DIR / "multilabel_prompting_results.json")
    bert           = load_json(OUT_DIR / "bert_results.json")
    tfidf          = load_json(OUT_DIR / "tfidf_results.json")
    logit          = load_json(OUT_DIR / "logit_results.json")
    slotwise_logit = load_json(OUT_DIR / "slotwise_logit_results.json")
    slot_data      = load_json(OUT_DIR / "slot_metrics_test.json")

    # Extract overall metrics
    p_acc  = probe["probing"]["sufficiency_accuracy"]   * 100
    p_f1   = probe["probing"]["sufficiency_f1_omission"] * 100
    iv_acc = prompting["twostep"]["verify_accuracy"]      * 100
    iv_f1  = prompting["twostep"]["verify_f1_omission"]   * 100
    bt_acc = bert["verify_accuracy"]    * 100
    bt_f1  = bert["verify_f1_omission"] * 100
    tf_acc = tfidf["verify_accuracy"]    * 100
    tf_f1  = tfidf["verify_f1_omission"] * 100

    lg_acc = logit["verify_accuracy"]    * 100
    lg_f1  = logit["verify_f1_omission"] * 100

    sw_acc = slotwise_logit["verify_accuracy"]    * 100
    sw_f1  = slotwise_logit["verify_f1_omission"] * 100

    # ── Palette ───────────────────────────────────────────────────────────
    c_probe   = '#00897B'   # Teal   — our method
    c_bert    = '#1565C0'   # Blue   — BERT fine-tune
    c_tfidf   = '#6A1B9A'   # Purple — TF-IDF
    c_iv      = '#E53935'   # Red    — Identify-then-Verify
    c_logit   = '#00838F'   # Cyan   — Logit P(Yes)
    c_sw      = '#FF8A65'   # Deep Orange — Slot-wise Logit

    # ── Data for left panel (Uncertainty Detection) ───────────────────────
    methods = [
        'BERT\nfine-tune', 
        'Slot-wise\nLogit', 
        'Logit\nP(Yes)', 
        'Probing\n(Ours)', 
        'Identify-\nthen-Verify',
        'TF-IDF\n+LR', 
    ]
    accs   = [bt_acc, sw_acc, lg_acc, p_acc, iv_acc, tf_acc]
    f1s    = [bt_f1,  sw_f1,  lg_f1,  p_f1,  iv_f1,  tf_f1]
    colors = [c_bert, c_sw, c_logit, c_probe, c_iv, c_tfidf]

    # Sort by F1 descending for visual clarity
    order = np.argsort(f1s)[::-1]
    methods = [methods[i] for i in order]
    accs    = [accs[i]    for i in order]
    f1s     = [f1s[i]     for i in order]
    colors  = [colors[i]  for i in order]

    # ── Right panel: Per-slot Probing F1 vs Slot-wise Logit F1 ─────────────
    slot_names = slot_data["slots"]  # ['who', 'when', 'how', 'what', 'where']
    probing_f1 = slot_data["f1"]
    
    # map slot names to values in slotwise_logit
    logit_f1 = [slotwise_logit["slot_f1"][s] * 100 for s in slot_names]

    # ── Figure ────────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.0), dpi=150)
    fig.suptitle("Uncertainty & Semantic Completeness Detection (Gemma-2-2b-it, Layer 26)",
                 fontsize=14, fontweight='bold', y=1.02)

    # ── Left: Uncertainty Detection (F1 bars + Accuracy dots) ─────────────
    x1 = np.arange(len(methods))
    w  = 0.38
    b_acc = ax1.bar(x1 - w/2, accs, w, color=colors, alpha=0.45,
                    edgecolor=colors, linewidth=1.5, linestyle='--',
                    label='Accuracy')
    b_f1  = ax1.bar(x1 + w/2, f1s,  w, color=colors, alpha=0.9,
                    label='F1 (Omission)')

    for bar, v in zip(b_acc, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#555')
    for bar, v in zip(b_f1, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{v:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_title('Task 1: Context Uncertainty Detection\n(Sufficient vs. Insufficient)',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=10)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(methods, fontsize=8.5)
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    style_patches = [
        mpatches.Patch(facecolor='gray', alpha=0.45, edgecolor='gray', linestyle='--', label='Accuracy'),
        mpatches.Patch(facecolor='gray', alpha=0.9, label='F1 (Omission)'),
    ]
    ax1.legend(handles=style_patches, fontsize=8.5, loc='upper right')

    # ── Right: Grouped Bar Chart (Probing vs Slot-wise Logit) ─────────────
    x2  = np.arange(len(slot_names))
    w2  = 0.35
    
    # We want to mute the color of 'who' due to insufficient data
    c_prob_list = [('#B0BEC5' if s == 'who' else c_probe) for s in slot_names]
    c_logit_list = [('#CFD8DC' if s == 'who' else c_sw) for s in slot_names]

    b_prob = ax2.bar(x2 - w2/2, probing_f1, w2, color=c_prob_list, alpha=0.9,
                     edgecolor='white', linewidth=0.5, label='Probing (Ours)')
    b_log  = ax2.bar(x2 + w2/2, logit_f1,   w2, color=c_logit_list, alpha=0.8,
                     edgecolor='white', linewidth=0.5, label='Slot-wise Logit Prompting')

    # Text annotations on bars
    for bar, f in zip(b_prob, probing_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{f:.1f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#004D40')
    for bar, f in zip(b_log, logit_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{f:.1f}%', ha='center', va='bottom', fontsize=8.5, color='#4E342E')

    # Chance line
    ax2.axhline(50.0, color='gray', linestyle='--', linewidth=1.2, label='Chance level (50%)')
    
    if 'who' in slot_names:
        wi = slot_names.index('who')
        ax2.text(wi, max(probing_f1[wi], logit_f1[wi]) + 6, '†', ha='center', fontsize=13, color='#90A4AE')

    ax2.set_title('Task 2: Case Role Completeness Detection\n(Per-slot Binary F1-Score)',
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontsize=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([('who†' if s == 'who' else s) for s in slot_names], fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Legend patches for right panel
    right_patches = [
        mpatches.Patch(color=c_probe, label='Probing (Ours)'),
        mpatches.Patch(color=c_sw,    label='Slot-wise Logit Prompting'),
    ]
    ax2.legend(handles=right_patches, fontsize=8.5, loc='upper right')
    
    ax2.text(0.02, 0.02,
             '† who (Agent): 1.9% of data — unreliable\n'
             '  Eval: held-out test split (50% of dev, stratified)',
             transform=ax2.transAxes, fontsize=7.0, color='gray', va='bottom')

    plt.tight_layout()
    out = OUT_DIR / "comprehensive_comparison_v3.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"Saved to {out}")
    plt.close()

    # ── Print comparison table ─────────────────────────────────────────────
    print("\n====== Final Comparison Table (Test Split) ======")
    print(f"{'Method':<28} {'Acc':>8} {'F1':>8}")
    print("-" * 47)
    for m, a, f in zip(methods, accs, f1s):
        m_clean = m.replace('\n', ' ')
        print(f"{m_clean:<28} {a:>8.2f}% {f:>8.2f}%")


if __name__ == "__main__":
    main()
