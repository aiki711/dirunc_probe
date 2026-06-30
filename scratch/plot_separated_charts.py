#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_json(path):
    return json.loads(Path(path).read_text())

def main():
    OUT_DIR = Path("runs/identify_verify_comparison")
    
    # Load data
    balanced_data  = load_json(OUT_DIR / "binary_balanced_comparison_results.json")
    slotwise_logit = load_json(OUT_DIR / "slotwise_logit_results_balanced.json")
    slot_data      = load_json(OUT_DIR / "slot_metrics_test.json")

    metrics = balanced_data["metrics"]

    # Define the 5 main methods for Task 1
    methods = [
        "BERT\nfine-tune",
        "Probing\n(Ours)",
        "TF-IDF\n+LR",
        "Logit\nP(Yes)",
        "Slot-wise\nLogit",
    ]

    # Map the JSON keys to the methods
    key_mapping = {
        "BERT\nfine-tune": "BERT fine-tune",
        "Probing\n(Ours)": "Probing (Ours - Multinomial)",
        "TF-IDF\n+LR": "TF-IDF + LR",
        "Logit\nP(Yes)": "Logit P(Yes)",
        "Slot-wise\nLogit": "Slot-wise Logit",
    }

    # Extract 4 metrics for each method
    accs = []
    precs = []
    recs = []
    f1s = []
    for m in methods:
        key = key_mapping[m]
        accs.append(metrics[key]["accuracy"] * 100)
        precs.append(metrics[key]["precision"] * 100)
        recs.append(metrics[key]["recall"] * 100)
        f1s.append(metrics[key]["f1"] * 100)

    # ── Palette for Task 1 ────────────────────────────────────────────────
    # Premium harmonized HSL/Hex palette
    c_acc  = '#78909C'  # Muted Slate Blue-Grey
    c_prec = '#26A69A'  # Greenish Teal
    c_rec  = '#FF7043'  # Coral Orange
    c_f1   = '#1E88E5'  # Brilliant Blue

    # ── Figure 1: Context Uncertainty Detection (4 Grouped Bars per Method) ──
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    
    fig1, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)
    
    x1 = np.arange(len(methods))
    width = 0.18  # width of each bar
    
    # Plot bars
    b_acc  = ax1.bar(x1 - 1.5*width, accs,  width, label='Accuracy',  color=c_acc,  alpha=0.85, edgecolor=c_acc, linewidth=1)
    b_prec = ax1.bar(x1 - 0.5*width, precs, width, label='Precision', color=c_prec, alpha=0.85, edgecolor=c_prec, linewidth=1)
    b_rec  = ax1.bar(x1 + 0.5*width, recs,  width, label='Recall',    color=c_rec,  alpha=0.85, edgecolor=c_rec,  linewidth=1)
    b_f1   = ax1.bar(x1 + 1.5*width, f1s,   width, label='F1-Score',  color=c_f1,   alpha=0.95, edgecolor=c_f1,   linewidth=1.2)

    # Labels and titles
    ax1.set_title("Task 1: Context Uncertainty Detection (Verify Baselines)\nBinary-Balanced Test Split (150 Sufficient vs. 150 Insufficient)", 
                  fontsize=12, fontweight='bold', pad=15)
    ax1.set_ylabel("Score (%)", fontsize=11, labelpad=8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(methods, fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Add legend
    ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=10)

    # Annotate bar values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=8, fontweight='semibold')

    autolabel(b_acc)
    autolabel(b_prec)
    autolabel(b_rec)
    autolabel(b_f1)

    plt.tight_layout()
    fig1_path = OUT_DIR / "context_uncertainty_detection.png"
    plt.savefig(fig1_path, bbox_inches='tight')
    print(f"Saved separated chart to {fig1_path}")

    # ── Figure 2: Case Role Completeness Detection (3-Panel: Precision, Recall, F1) ──
    slot_names = slot_data["slots"]  # ['who', 'when', 'how', 'what', 'where']
    probing_p = [slot_data["precision"][slot_data["slots"].index(s)] for s in slot_names]
    probing_r = [slot_data["recall"][slot_data["slots"].index(s)] for s in slot_names]
    probing_f1 = [slot_data["f1"][slot_data["slots"].index(s)] for s in slot_names]
    
    logit_p = [slotwise_logit["slot_precision"][s] * 100 for s in slot_names]
    logit_r = [slotwise_logit["slot_recall"][s] * 100 for s in slot_names]
    logit_f1 = [slotwise_logit["slot_f1"][s] * 100 for s in slot_names]

    fig2, (ax_p, ax_r, ax_f1) = plt.subplots(1, 3, figsize=(18, 6.0), dpi=150)
    
    x2 = np.arange(len(slot_names))
    width2 = 0.35
    c_ours = '#00897B'  # Teal
    c_sw   = '#FF8A65'  # Coral/Orange
    formatted_slots = [f"{s}†" if s == 'who' else s for s in slot_names]

    # Subplot 1: Precision
    b_ours_p = ax_p.bar(x2 - width2/2, probing_p, width2, label='Probing (Ours - Layer 16)', color=c_ours, alpha=0.9, edgecolor=c_ours, linewidth=1)
    b_sw_p   = ax_p.bar(x2 + width2/2, logit_p,   width2, label='Slot-wise Logit Prompting (5-shot)', color=c_sw, alpha=0.9, edgecolor=c_sw, linewidth=1)
    ax_p.set_title("Precision", fontsize=12, fontweight='bold')
    ax_p.set_ylabel("Score (%)", fontsize=11)
    ax_p.set_xticks(x2)
    ax_p.set_xticklabels(formatted_slots, fontsize=10, fontweight='bold')
    ax_p.set_ylim(0, 110)
    ax_p.grid(axis='y', linestyle='--', alpha=0.3)
    ax_p.spines['top'].set_visible(False)
    ax_p.spines['right'].set_visible(False)
    ax_p.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Subplot 2: Recall
    b_ours_r = ax_r.bar(x2 - width2/2, probing_r, width2, label='Probing (Ours - Layer 16)', color=c_ours, alpha=0.9, edgecolor=c_ours, linewidth=1)
    b_sw_r   = ax_r.bar(x2 + width2/2, logit_r,   width2, label='Slot-wise Logit Prompting (5-shot)', color=c_sw, alpha=0.9, edgecolor=c_sw, linewidth=1)
    ax_r.set_title("Recall", fontsize=12, fontweight='bold')
    ax_r.set_xticks(x2)
    ax_r.set_xticklabels(formatted_slots, fontsize=10, fontweight='bold')
    ax_r.set_ylim(0, 110)
    ax_r.grid(axis='y', linestyle='--', alpha=0.3)
    ax_r.spines['top'].set_visible(False)
    ax_r.spines['right'].set_visible(False)
    ax_r.axhline(y=50, color='gray', linestyle='--', alpha=0.7)

    # Subplot 3: F1-Score
    b_ours_f1 = ax_f1.bar(x2 - width2/2, probing_f1, width2, label='Probing (Ours - Layer 16)', color=c_ours, alpha=0.9, edgecolor=c_ours, linewidth=1)
    b_sw_f1   = ax_f1.bar(x2 + width2/2, logit_f1,   width2, label='Slot-wise Logit Prompting (5-shot)', color=c_sw, alpha=0.9, edgecolor=c_sw, linewidth=1)
    ax_f1.set_title("F1-Score", fontsize=12, fontweight='bold')
    ax_f1.set_xticks(x2)
    ax_f1.set_xticklabels(formatted_slots, fontsize=10, fontweight='bold')
    ax_f1.set_ylim(0, 110)
    ax_f1.grid(axis='y', linestyle='--', alpha=0.3)
    ax_f1.spines['top'].set_visible(False)
    ax_f1.spines['right'].set_visible(False)
    ax_f1.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    
    # Global title and legend
    fig2.suptitle("Task 2: Case Role Completeness Detection (Per-slot Precision, Recall, and F1)\nHeld-out Test Split (Unbiased Evaluation)", 
                  fontsize=14, fontweight='bold', y=0.98)
    ax_f1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#E0E0E0', fontsize=10)

    # Annotate values
    def autolabel_slot(ax, rects, is_unreliable=False):
        for rect in rects:
            height = rect.get_height()
            label = f'{height:.1f}%'
            x_pos = rect.get_x() + rect.get_width() / 2
            if is_unreliable and rect.get_x() < 0.2: # 'who' slot
                ax.annotate(label, xy=(x_pos, height), xytext=(0, 3),
                             textcoords="offset points", ha='center', va='bottom',
                             fontsize=8.5, fontweight='semibold')
                ax.annotate('†', xy=(x_pos, height + 8), xytext=(0, 0),
                             textcoords="offset points", ha='center', va='bottom',
                             fontsize=11, color='gray')
            else:
                ax.annotate(label, xy=(x_pos, height), xytext=(0, 3),
                             textcoords="offset points", ha='center', va='bottom',
                             fontsize=8.5, fontweight='semibold')

    autolabel_slot(ax_p, b_ours_p, is_unreliable=True)
    autolabel_slot(ax_p, b_sw_p)
    autolabel_slot(ax_r, b_ours_r, is_unreliable=True)
    autolabel_slot(ax_r, b_sw_r)
    autolabel_slot(ax_f1, b_ours_f1, is_unreliable=True)
    autolabel_slot(ax_f1, b_sw_f1)

    # Footnote on the first subplot (left aligned)
    ax_p.text(0.02, 0.03, "† who (Agent): 1.9% of data — unreliable\nEval: held-out test split (50% of dev, stratified)", 
              fontsize=8.0, color='gray', va='bottom', transform=ax_p.transAxes)

    plt.tight_layout()
    # Adjust layout to leave space for suptitle
    plt.subplots_adjust(top=0.85)
    fig2_path = OUT_DIR / "case_role_completeness_detection.png"
    plt.savefig(fig2_path, bbox_inches='tight')
    print(f"Saved separated slot chart to {fig2_path}")

if __name__ == "__main__":
    main()
