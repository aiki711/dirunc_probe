"""
39_plot_4way_comparison.py
===========================
4手法の比較グラフを生成するスクリプト。

手法:
  - 7 Query Tokens (Aligned)
  - 7 Query Tokens (Unaligned)
  - Final Token  (Aligned)
  - Final Token  (Unaligned)

レイアウト: 2行(Soft/Strong) × 3列(Macro F1 / Std Pair Acc / Strict Pair Acc)
"""

import json
import re
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Style settings
# ---------------------------------------------------------------------------
METHODS = [
    {
        "label": "7 Query Tokens (Aligned)",
        "run_dir_key": "query_aligned",
        "color": "#1565C0",   # dark blue (aligned = solid)
        "linestyle": "-",
        "marker": "o",
        "alpha": 1.0,
    },
    {
        "label": "7 Query Tokens (Unaligned)",
        "run_dir_key": "query_unaligned",
        "color": "#64B5F6",   # light blue (unaligned = dashed)
        "linestyle": "--",
        "marker": "o",
        "alpha": 0.9,
    },
    {
        "label": "Final Token (Aligned)",
        "run_dir_key": "final_aligned",
        "color": "#BF360C",   # dark orange-red (aligned = solid)
        "linestyle": "-",
        "marker": "s",
        "alpha": 1.0,
    },
    {
        "label": "Final Token (Unaligned)",
        "run_dir_key": "final_unaligned",
        "color": "#FF8A65",   # light orange (unaligned = dashed)
        "linestyle": "--",
        "marker": "s",
        "alpha": 0.9,
    },
]

METRICS = [
    {"col_name": "macro_f1",              "display": "Macro F1"},
    {"col_name": "pair_accuracy_standard","display": "Std Pair Accuracy"},
    {"col_name": "pair_accuracy_strict",  "display": "Strict Pair Accuracy"},
]

OMISSION_TYPES = ["soft", "strong"]

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
LAYER_RE = re.compile(r"^(?:soft|strong)_layer_(\d+)$")


def load_run_dir(runs_path: Path, omission: str) -> dict[int, dict]:
    """
    runs_path 以下の {omission}_layer_* ディレクトリから
    metrics.json / log.jsonl を読み込み、layer -> metrics dict を返す。
    """
    results: dict[int, dict] = {}
    pattern = f"{omission}_layer_*"
    for layer_dir in sorted(runs_path.glob(pattern)):
        m = LAYER_RE.match(layer_dir.name)
        if m is None:
            continue
        layer_idx = int(m.group(1))

        # ---- Try metrics.json first ----
        metrics_file = layer_dir / "metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                data = json.load(f)
            results[layer_idx] = data
            continue

        # ---- Fall back to log.jsonl (best epoch) ----
        log_file = layer_dir / "log.jsonl"
        if log_file.exists():
            best_score = -1.0
            best_data: dict | None = None
            with log_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    score = (data.get("pair_accuracy_standard", 0.0)
                             + data.get("macro_f1", 0.0))
                    if score > best_score:
                        best_score = score
                        best_data = data
            if best_data is not None:
                results[layer_idx] = best_data

    return results


def extract_series(results: dict[int, dict], metric_key: str):
    """layer インデックス昇順にソートした (layers, values) を返す。"""
    layers = sorted(results.keys())
    values = [results[l].get(metric_key, float("nan")) for l in layers]
    return np.array(layers), np.array(values, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate 2×3 subplot comparison figure for 4 probing configurations."
    )
    parser.add_argument(
        "--runs_dir", type=str, default="runs",
        help="ベースとなる runs/ ディレクトリのパス"
    )
    parser.add_argument(
        "--query_aligned_dir",   type=str, default="layer_sweep_gemini_nq_aligned",
        help="7 Query Tokens (Aligned) の結果フォルダ名"
    )
    parser.add_argument(
        "--query_unaligned_dir", type=str, default="layer_sweep_gemini_nq_unaligned",
        help="7 Query Tokens (Unaligned) の結果フォルダ名"
    )
    parser.add_argument(
        "--final_aligned_dir",   type=str, default="layer_sweep_gemini_final_token_aligned",
        help="Final Token (Aligned) の結果フォルダ名"
    )
    parser.add_argument(
        "--final_unaligned_dir", type=str, default="layer_sweep_gemini_final_token_unaligned",
        help="Final Token (Unaligned) の結果フォルダ名"
    )
    parser.add_argument(
        "--out_path", type=str, default="runs/4way_comparison.png",
        help="出力画像パス"
    )
    parser.add_argument(
        "--dpi", type=int, default=180,
        help="出力解像度 (DPI)"
    )
    args = parser.parse_args()

    base = Path(args.runs_dir)
    run_dirs = {
        "query_aligned":   base / args.query_aligned_dir,
        "query_unaligned": base / args.query_unaligned_dir,
        "final_aligned":   base / args.final_aligned_dir,
        "final_unaligned": base / args.final_unaligned_dir,
    }

    # ---- Load all data ----
    # data[omission][method_key] -> {layer_idx: metrics_dict}
    all_data: dict[str, dict[str, dict[int, dict]]] = {}
    for omission in OMISSION_TYPES:
        all_data[omission] = {}
        for method in METHODS:
            key = method["run_dir_key"]
            rd = run_dirs[key]
            if not rd.exists():
                print(f"[WARNING] Directory not found: {rd}")
                all_data[omission][key] = {}
            else:
                all_data[omission][key] = load_run_dir(rd, omission)

    # ---- Build figure: 2 rows (Soft/Strong) × 3 cols (metrics) ----
    fig, axes = plt.subplots(
        nrows=2, ncols=3,
        figsize=(15, 9),
        dpi=args.dpi,
    )
    fig.subplots_adjust(top=0.88, bottom=0.14, hspace=0.42, wspace=0.32)

    # Global style
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.45,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
    })

    omission_labels = {"soft": "Soft Omission", "strong": "Strong Omission"}

    line_handles = []  # for shared legend

    for row_i, omission in enumerate(OMISSION_TYPES):
        for col_j, metric in enumerate(METRICS):
            ax = axes[row_i][col_j]
            metric_key = metric["col_name"]

            handles = []
            for method in METHODS:
                key = method["run_dir_key"]
                results = all_data[omission].get(key, {})
                if not results:
                    continue
                layers, values = extract_series(results, metric_key)
                ln, = ax.plot(
                    layers, values,
                    color=method["color"],
                    linestyle=method["linestyle"],
                    marker=method["marker"],
                    markersize=5,
                    linewidth=1.8,
                    alpha=method.get("alpha", 1.0),
                    label=method["label"],
                )
                handles.append(ln)

            if row_i == 0 and col_j == 0:
                line_handles = handles  # save for shared legend

            # Row label on leftmost column
            if col_j == 0:
                ax.set_ylabel(
                    f"{omission_labels[omission]}\n{metric['display']}",
                    fontsize=9, fontweight="bold"
                )
            else:
                ax.set_ylabel(metric["display"], fontsize=9)

            ax.set_xlabel("Layer Index", fontsize=9)

            # Column title on top row only
            if row_i == 0:
                ax.set_title(metric["display"], fontsize=10, fontweight="bold", pad=4)

            ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.set_xlim(-1, 28)

            # Set y-axis range appropriate for the metric
            if metric_key == "pair_accuracy_standard":
                ax.set_ylim(0.85, 1.00)
            else:
                ax.set_ylim(0.30, 0.80)

    # Shared legend at the bottom
    if line_handles:
        fig.legend(
            line_handles,
            [m["label"] for m in METHODS],
            loc="lower center",
            ncol=4,
            fontsize=9,
            framealpha=0.9,
            bbox_to_anchor=(0.5, 0.02),
        )

    # Overall figure title
    fig.suptitle(
        "4-Way Probing Comparison: Query Tokens vs Final Token, Aligned vs Unaligned\n"
        "(Gemma-3-4B, Layer Sweep over 8 Checkpoints)",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    # Row labels (Soft / Strong) as figure text on the left side
    for row_i, omission in enumerate(OMISSION_TYPES):
        fig.text(
            0.01,
            0.73 - row_i * 0.44,
            omission_labels[omission],
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            rotation=90,
            color="#333333",
        )

    # Save
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[OK] Figure saved to: {out_path}")

    # Also save individual soft/strong figures
    for row_i, omission in enumerate(OMISSION_TYPES):
        fig_single, axes_single = plt.subplots(
            nrows=1, ncols=3,
            figsize=(14, 4),
            dpi=args.dpi,
            constrained_layout=True,
        )
        for col_j, metric in enumerate(METRICS):
            ax = axes_single[col_j]
            metric_key = metric["col_name"]

            for method in METHODS:
                key = method["run_dir_key"]
                results = all_data[omission].get(key, {})
                if not results:
                    continue
                layers, values = extract_series(results, metric_key)
                ax.plot(
                    layers, values,
                    color=method["color"],
                    linestyle=method["linestyle"],
                    marker=method["marker"],
                    markersize=5,
                    linewidth=1.8,
                    label=method["label"],
                )
            ax.set_xlabel("Layer Index")
            ax.set_ylabel(metric["display"])
            ax.set_title(metric["display"], fontsize=10, fontweight="bold")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.set_xlim(-1, 28)
            if metric_key == "pair_accuracy_standard":
                ax.set_ylim(0.85, 1.00)
            else:
                ax.set_ylim(0.30, 0.80)
            if col_j == 2:
                ax.legend(loc="lower right", framealpha=0.85)

        fig_single.suptitle(
            f"{omission_labels[omission]} — 4-Way Probing Comparison",
            fontsize=11, fontweight="bold",
        )
        single_path = out_path.with_name(f"4way_{omission}.png")
        fig_single.savefig(single_path, bbox_inches="tight")
        print(f"[OK] {omission} figure saved to: {single_path}")
        plt.close(fig_single)

    plt.close(fig)
    print("[Done]")


if __name__ == "__main__":
    main()
