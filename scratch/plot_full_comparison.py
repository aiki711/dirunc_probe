#!/usr/bin/env python3
"""
Final comprehensive comparison figure with all baselines.

Reads results from:
  - runs/identify_verify_comparison/comparison_report_v2.json   (Probing)
  - runs/identify_verify_comparison/multilabel_prompting_results.json (Prompting)
  - runs/identify_verify_comparison/bert_results.json           (BERT fine-tune)
  - runs/identify_verify_comparison/tfidf_results.json          (TF-IDF)
  - runs/identify_verify_comparison/logit_results.json          (Logit, optional)
  - runs/identify_verify_comparison/slot_metrics_test.json      (Per-slot probing)

Generates:
  - runs/identify_verify_comparison/comprehensive_comparison_v2.png
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
    probe     = load_json(OUT_DIR / "comparison_report_v2.json")
    prompting = load_json(OUT_DIR / "multilabel_prompting_results.json")
    bert      = load_json(OUT_DIR / "bert_results.json")
    tfidf     = load_json(OUT_DIR / "tfidf_results.json")
    logit     = load_json(OUT_DIR / "logit_results.json")  # optional
    slot_data = load_json(OUT_DIR / "slot_metrics_test.json")

    # Extract values
    p_acc  = probe["probing"]["sufficiency_accuracy"]   * 100
    p_f1   = probe["probing"]["sufficiency_f1_omission"] * 100
    os_acc = prompting["onestep"]["verify_accuracy"]      * 100
    os_f1  = prompting["onestep"]["verify_f1_omission"]   * 100
    iv_acc = prompting["twostep"]["verify_accuracy"]      * 100
    iv_f1  = prompting["twostep"]["verify_f1_omission"]   * 100
    bt_acc = bert["verify_accuracy"]    * 100
    bt_f1  = bert["verify_f1_omission"] * 100
    tf_acc = tfidf["verify_accuracy"]    * 100
    tf_f1  = tfidf["verify_f1_omission"] * 100

    # Logit-based (optional; use placeholder if not yet run)
    if logit:
        lg_acc = logit["verify_accuracy"]    * 100
        lg_f1  = logit["verify_f1_omission"] * 100
        include_logit = True
    else:
        include_logit = False

    # Random baseline
    rnd_acc = 50.0
    rnd_f1  = 66.7  # F1 for Omission class when ~50% missing in balanced eval

    # ── Palette ───────────────────────────────────────────────────────────
    c_probe   = '#00897B'   # Teal   — our method
    c_bert    = '#1565C0'   # Blue   — BERT fine-tune
    c_tfidf   = '#6A1B9A'   # Purple — TF-IDF
    c_iv      = '#E53935'   # Red    — Identify-then-Verify
    c_os      = '#FB8C00'   # Amber  — One-step
    c_logit   = '#00838F'   # Cyan   — Logit-based
    c_rnd     = '#9E9E9E'   # Gray   — Random

    # ── Data for left panel (Uncertainty Detection) ───────────────────────
    if include_logit:
        methods = ['Probing\n(Ours)', 'BERT\nfine-tune', 'Identify-\nthen-Verify',
                   'TF-IDF\n+LR', 'One-step\nPrompting', 'Logit\nP(Yes)', 'Random']
        accs    = [p_acc, bt_acc, iv_acc, tf_acc, os_acc, lg_acc, rnd_acc]
        f1s     = [p_f1,  bt_f1,  iv_f1,  tf_f1,  os_f1,  lg_f1,  rnd_f1]
        colors  = [c_probe, c_bert, c_iv, c_tfidf, c_os, c_logit, c_rnd]
    else:
        methods = ['Probing\n(Ours)', 'BERT\nfine-tune', 'Identify-\nthen-Verify',
                   'TF-IDF\n+LR', 'One-step\nPrompting', 'Random']
        accs    = [p_acc, bt_acc, iv_acc, tf_acc, os_acc, rnd_acc]
        f1s     = [p_f1,  bt_f1,  iv_f1,  tf_f1,  os_f1,  rnd_f1]
        colors  = [c_probe, c_bert, c_iv, c_tfidf, c_os, c_rnd]

    # Sort by F1 descending for visual clarity
    order = np.argsort(f1s)[::-1]
    methods = [methods[i] for i in order]
    accs    = [accs[i]    for i in order]
    f1s     = [f1s[i]     for i in order]
    colors  = [colors[i]  for i in order]

    # ── Right panel: Per-slot probing F1 ─────────────────────────────────
    slot_names = slot_data["slots"]
    slot_f1    = slot_data["f1"]

    # ── Figure ────────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=150)
    fig.suptitle("Uncertainty Detection: Probing vs. Baselines (Gemma-2-2b-it, Layer 26)",
                 fontsize=13, fontweight='bold', y=1.01)

    # ── Left: Uncertainty Detection (F1 bars + Accuracy dots) ─────────────
    x1 = np.arange(len(methods))
    w  = 0.38
    b_acc = ax1.bar(x1 - w/2, accs, w, color=colors, alpha=0.55,
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

    # Reference lines
    ax1.axhline(66.7, color=c_rnd, linestyle=':', linewidth=1.0, alpha=0.7)

    ax1.set_title('Uncertainty Detection\n(Sufficient vs. Insufficient)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=10)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(methods, fontsize=8.5)
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # Legend for bar styles
    from matplotlib.patches import Patch
    style_patches = [
        Patch(facecolor='gray', alpha=0.55, edgecolor='gray', linestyle='--', label='Accuracy'),
        Patch(facecolor='gray', alpha=0.9, label='F1 (Omission)'),
    ]
    ax1.legend(handles=style_patches, fontsize=8, loc='upper right')

    # ── Right: Per-slot Binary F1 (Probing) ───────────────────────────────
    x2  = np.arange(len(slot_names))
    w2  = 0.55
    colors2 = [('#B0BEC5' if s == 'who' else c_probe) for s in slot_names]
    b3 = ax2.bar(x2, slot_f1, w2, color=colors2, alpha=0.88,
                 edgecolor='white', linewidth=0.8)
    for bar, f in zip(b3, slot_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{f:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.axhline(50.0, color='gray', linestyle='--', linewidth=1.2, label='Chance level (50%)')
    if 'who' in slot_names:
        wi = slot_names.index('who')
        ax2.text(wi, slot_f1[wi] + 7, '†', ha='center', fontsize=13, color='#B0BEC5')

    ax2.set_title('Case Role Completeness Detection\n(Probing, binary per-slot)',
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontsize=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([('who†' if s == 'who' else s) for s in slot_names], fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.text(0.02, 0.02,
             '† who (Agent): 1.9% of data — unreliable\n'
             '  Eval: held-out test split (50% of dev, stratified)',
             transform=ax2.transAxes, fontsize=6.5, color='gray', va='bottom')

    plt.tight_layout()
    out = OUT_DIR / "comprehensive_comparison_v2.png"
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
