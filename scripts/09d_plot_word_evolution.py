"""
Experiment 9-Extension: Word-Level Evolution Plot
==================================================
Visualises the word-by-word checklist state transition.
Target labels (WHO, WHEN, WHERE) are drawn at full opacity;
others are faded to background.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

DIRS = ["who", "what", "when", "where", "why", "how", "which"]
TARGET_LABELS = ["WHO", "WHEN", "WHERE"]

# Provisional thresholds (same as Exp9 sentence-level)
THRESHOLDS = {
    "WHO": 0.15,
    "WHAT": 0.55,
    "WHEN": 0.10,
    "WHERE": 0.35,
    "HOW": 0.25,
    "WHICH": 0.45,
    "WHY": 0.00,
}

def main():
    input_dir = Path("runs/balanced/experiment9_word")
    data_file = input_dir / "word_trajectory.json"

    if not data_file.exists():
        print(f"Error: {data_file} not found. Run 09c_word_evolution.py first.")
        return

    with open(data_file) as f:
        data = json.load(f)

    evolution = data["word_evolution"]
    words = [e["word"] for e in evolution]
    x = np.arange(len(words))

    import seaborn as sns
    palette = {d.upper(): c for d, c in zip(DIRS, sns.color_palette("tab10", n_colors=len(DIRS)))}

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(20, len(words) * 0.55), 8))

    for d in DIRS:
        label = d.upper()
        is_target = label in TARGET_LABELS
        probs = [e["probs"][d] for e in evolution]

        ax.plot(
            x, probs,
            label=label,
            color=palette[label],
            linewidth=3.5 if is_target else 1.2,
            linestyle="-" if is_target else "--",
            alpha=1.0 if is_target else 0.2,
            marker="o" if is_target else None,
            markersize=7,
            zorder=3 if is_target else 1,
        )

        # Threshold lines for targets
        if is_target and label in THRESHOLDS:
            ax.axhline(
                THRESHOLDS[label],
                color=palette[label],
                linestyle=":",
                linewidth=2,
                alpha=0.5,
                zorder=2,
            )
            ax.text(
                len(words) - 0.5,
                THRESHOLDS[label] + 0.015,
                f"Thresh {THRESHOLDS[label]}",
                color=palette[label],
                fontsize=9,
                alpha=0.8,
                fontweight="bold",
            )

    ax.set_title(
        "Word-Level Checklist Evolution (hs[-1] probing)\n"
        "Solid / highlighted = target labels; dashed / faded = others",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("P(Missing Information)", fontsize=13)
    ax.set_xlabel("Words added to context (left → right)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=60, ha="right", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.legend(
        title="Checklist Label",
        title_fontsize=11,
        fontsize=10,
        loc="upper right",
        bbox_to_anchor=(1.18, 1),
    )

    plt.tight_layout()
    out_file = input_dir / "word_evolution_line.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"Saved Word Evolution Line Chart to {out_file}")
    plt.close()


if __name__ == "__main__":
    main()
