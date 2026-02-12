# scripts/05_viz_dirunc.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

DIRS = ["who", "what", "when", "where", "why", "how", "which"]

# ---------------- I/O ----------------

def read_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_summary(summary_json: Path):
    return json.loads(summary_json.read_text(encoding="utf-8"))

# ---------------- Filters (match 03_train_probe.py) ----------------

def parse_int_list(s: str):
    s = s.strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def filter_rows(rows, phases=None, levels=None, k_values=None):
    """
    phases: list[int] or None
    levels: list[int] or None
    k_values: list[int] or None  (only applied when phase==2)
    """
    out = []
    for r in rows:
        # キーが存在しない場合はフィルタリングをスキップ（None または適当な値を設定）
        ph_raw = r.get("phase")
        ph = int(ph_raw) if ph_raw is not None else None
        
        lv_raw = r.get("level")
        lv = int(lv_raw) if lv_raw is not None else None
        
        kv = r.get("k", None)
        kv_int = None if kv is None else int(kv)

        if phases is not None and (ph is None or ph not in phases):
            continue
        if levels is not None and (lv is None or lv not in levels):
            continue
        if k_values is not None:
            if ph == 2 and (kv_int not in k_values):
                continue

        out.append(r)
    return out

# ---------------- Summary helpers ----------------

def compute_label_support(rows):
    """rows から labels の正例数 support_pos[dir] を返す"""
    support = {d: 0 for d in DIRS}
    for r in rows:
        labels = r.get("labels", {})
        for d in DIRS:
            support[d] += int(labels.get(d, 0))
    return support

def _as_support_dict(x):
    # x が dict ならそのまま、tuple/list なら中の dict を探して返す
    if isinstance(x, dict):
        return x
    if isinstance(x, (tuple, list)):
        for it in x:
            if isinstance(it, dict):
                return it
        # dictが無ければ先頭を返す（最後の保険）
        return x[0]
    raise TypeError(f"Unsupported support type: {type(x)}")

def print_A1_A2_numbers(train_rows, dev_rows) -> None:
    tr_sup = _as_support_dict(compute_label_support(train_rows))
    dv_sup = _as_support_dict(compute_label_support(dev_rows))
    Ntr = len(train_rows)
    Ndv = len(dev_rows)

    print("\n=== [A1] label support (pos counts) train vs dev ===")
    print(f"N_train={Ntr}, N_dev={Ndv}")
    for d in DIRS:
        print(f"{d:>5s}: train={int(tr_sup.get(d,0)):7d}, dev={int(dv_sup.get(d,0)):7d}")

    print("\n=== [A2] label positive rate (support / N) train vs dev ===")
    for d in DIRS:
        tr_rate = (int(tr_sup.get(d,0)) / Ntr) if Ntr else 0.0
        dv_rate = (int(dv_sup.get(d,0)) / Ndv) if Ndv else 0.0
        print(f"{d:>5s}: train={tr_rate:.6f}, dev={dv_rate:.6f}")

def get_best_metrics(block: dict) -> dict:
    """
    block: summary["baseline/layer_5"] etc.
    prefer block["best"]["final_dev"] if exists, else block["best"].
    """
    best = block.get("best", {})
    fd = best.get("final_dev", None)
    return fd if isinstance(fd, dict) else best

def extract_mode_layer_metrics(summary: dict):
    """
    returns:
      mode_to_layers[mode] = sorted list of layer_idx
      metrics[mode][layer] = dict with micro_f1/macro_f1/macro_f1_posonly
    """
    metrics = defaultdict(dict)
    mode_to_layers = defaultdict(set)

    for k, v in summary.items():
        if not isinstance(k, str) or "/" not in k:
            continue
        parts = k.split("/")
        if len(parts) != 2 or not parts[1].startswith("layer_"):
            continue
        
        mode = parts[0]
        layer_str = parts[1]
        layer = int(layer_str.replace("layer_", ""))
        m = get_best_metrics(v)
        metrics[mode][layer] = {
            "micro_f1": float(m.get("micro_f1", 0.0)),
            "macro_f1": float(m.get("macro_f1", 0.0)),
            "macro_f1_posonly": float(m.get("macro_f1_posonly", 0.0)),
            "threshold": float(m.get("threshold", 0.5)),
        }
        mode_to_layers[mode].add(layer)

    mode_to_layers = {m: sorted(list(s)) for m, s in mode_to_layers.items()}
    
    # Update DIRS from summary if possible
    global DIRS
    for k, v in summary.items():
        if "/" in k:
            m = get_best_metrics(v)
            labels = m.get("macro_posonly_labels", [])
            if labels and len(labels) > len(DIRS):
                DIRS = labels
                print(f"Updated DIRS from summary: {DIRS}")
                break
    
    return mode_to_layers, metrics

def extract_best_blocks(summary: dict):
    out = {}
    for k, v in summary.items():
        if isinstance(k, str) and k.endswith("/best"):
            mode = k.split("/")[0]
            best = v.get("best", {})
            fd = best.get("final_dev", None)
            out[mode] = fd if isinstance(fd, dict) else best
    return out

def print_B2_numbers(summary: dict) -> None:
    """B2（best baseline vs best query）で使った数値をprint"""
    best_blocks = extract_best_blocks(summary)
    keys = ["micro_f1", "macro_f1", "macro_f1_posonly"]
    print("\n=== [B2] comparison (final_dev preferred) ===")
    for mode_name, mm in best_blocks.items():
        vals = {k: float(mm.get(k, 0.0)) for k in keys}
        layer = mm.get("layer_idx", None)
        th = mm.get("threshold", None)
        print(f"- {mode_name}: " + ", ".join([f"{k}={vals[k]:.6f}" for k in keys]) + f", threshold={th}")

def print_B4_numbers(summary: dict) -> None:
    """
    B4（想定：ラベル別F1 comparison）で使う数値をprint
    """
    best_blocks = extract_best_blocks(summary)

    def fmt_per_label(mm: dict):
        per = mm.get("per_label_f1", [0.0]*len(DIRS))
        sup = mm.get("support_pos", [0]*len(DIRS))
        out = []
        for i, d in enumerate(DIRS):
            out.append((d, float(per[i]) if i < len(per) else 0.0, int(sup[i]) if i < len(sup) else 0))
        return out

    print("\n=== [B4] per-label F1 (best for each mode) ===")
    for mode_name, mm in best_blocks.items():
        print(f"-- {mode_name} --")
        rows = fmt_per_label(mm)
        for d, f1, sup in rows:
            print(f"  {d:>5s}: f1={f1:.6f}, support_pos={sup}")

# ---------------- A-main: label supports & co-occurrence ----------------

def compute_label_support(rows):
    """
    returns:
      support[d] = #positives
      N = #rows
    """
    support = {d: 0 for d in DIRS}
    N = len(rows)
    for r in rows:
        labels = r.get("labels", {})
        for d in DIRS:
            support[d] += int(labels.get(d, 0) == 1)
    return support, N

def compute_cooccurrence(rows):
    """
    returns:
      mat[i,j] = #samples where (label_i==1 and label_j==1)
      N = #rows
    """
    N = len(rows)
    mat = np.zeros((len(DIRS), len(DIRS)), dtype=np.int64)
    for r in rows:
        labels = r.get("labels", {})
        vec = np.array([int(labels.get(d, 0) == 1) for d in DIRS], dtype=np.int64)
        # outer product => co-occurrence count
        mat += np.outer(vec, vec)
    return mat, N

def plot_A1_support_counts(train_rows, dev_rows, out_path: Path):
    tr_sup, trN = compute_label_support(train_rows)
    dv_sup, dvN = compute_label_support(dev_rows)

    x = np.arange(len(DIRS))
    width = 0.40

    tr = [tr_sup[d] for d in DIRS]
    dv = [dv_sup[d] for d in DIRS]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(x - width/2, tr, width, label=f"train (N={trN})", alpha=0.90)
    ax.bar(x + width/2, dv, width, label=f"dev (N={dvN})", alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(DIRS)
    ax.set_ylabel("positive count (support)")
    ax.set_title("Train vs Dev: label support (positive counts)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_A2_support_rates(train_rows, dev_rows, out_path: Path):
    tr_sup, trN = compute_label_support(train_rows)
    dv_sup, dvN = compute_label_support(dev_rows)

    x = np.arange(len(DIRS))
    width = 0.40

    tr = [tr_sup[d] / trN if trN else 0.0 for d in DIRS]
    dv = [dv_sup[d] / dvN if dvN else 0.0 for d in DIRS]

    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(x - width/2, tr, width, label=f"train (N={trN})", alpha=0.90)
    ax.bar(x + width/2, dv, width, label=f"dev (N={dvN})", alpha=0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(DIRS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("positive rate (support / N)")
    ax.set_title("Train vs Dev: label rates")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_A3_cooccurrence_heatmap(train_rows, dev_rows, out_path: Path, normalize: str = "rate"):
    """
    normalize:
      - "count": show raw co-occurrence counts
      - "rate": show count / N (per-sample co-occurrence rate)
    """
    tr_mat, trN = compute_cooccurrence(train_rows)
    dv_mat, dvN = compute_cooccurrence(dev_rows)

    if normalize == "rate":
        tr_show = tr_mat / trN if trN else tr_mat.astype(np.float64)
        dv_show = dv_mat / dvN if dvN else dv_mat.astype(np.float64)
        fmt = "{:.3f}"
        title_suffix = " (rate=count/N)"
    else:
        tr_show = tr_mat.astype(np.float64)
        dv_show = dv_mat.astype(np.float64)
        fmt = "{:.0f}"
        title_suffix = " (count)"
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 4.8), constrained_layout=True)

    for ax, mat, name, N in [
        (axes[0], tr_show, "train", trN),
        (axes[1], dv_show, "dev", dvN),
    ]:
        im = ax.imshow(mat)  # default cmap
        ax.set_xticks(np.arange(len(DIRS)))
        ax.set_yticks(np.arange(len(DIRS)))
        ax.set_xticklabels(DIRS)
        ax.set_yticklabels(DIRS)
        ax.set_title(f"{name} co-occurrence{title_suffix}\nN={N}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # annotate cells
        for i in range(len(DIRS)):
            for j in range(len(DIRS)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)

# ---------------- Existing B: baseline vs query (keep) ----------------

def plot_B_mode_layer_metrics(summary: dict, out_path: Path):
    mode_to_layers, metrics = extract_mode_layer_metrics(summary)

    all_layers = sorted(set(mode_to_layers.get("baseline", []) + mode_to_layers.get("query", [])))
    if not all_layers:
        raise RuntimeError("No layer results found in summary.json (expected keys like baseline/layer_*, query/layer_*).")

    def series(mode, key):
        xs = []
        ys = []
        for L in all_layers:
            if L in metrics.get(mode, {}):
                xs.append(L)
                ys.append(metrics[mode][L].get(key, 0.0))
        return xs, ys

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), constrained_layout=True)

    for ax, metric_name in zip(axes, ["micro_f1", "macro_f1", "macro_f1_posonly"]):
        for mode in mode_to_layers.keys():
            xs, ys = series(mode, metric_name)
            if xs:
                ax.plot(xs, ys, marker="o", label=mode)
        ax.set_title(f"{metric_name} vs layer")
        ax.set_xlabel("layer_idx")
        ax.set_ylabel(metric_name)
        ax.set_xticks(all_layers)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("baseline vs query: metrics across layers", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_B_best_bars(summary: dict, out_path: Path):
    best_blocks = extract_best_blocks(summary)
    keys = ["micro_f1", "macro_f1", "macro_f1_posonly"]
    
    modes = sorted(best_blocks.keys())
    x = np.arange(len(keys))
    width = 0.8 / max(1, len(modes))

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for i, mode in enumerate(modes):
        vals = [float(best_blocks[mode].get(k, 0.0)) for k in keys]
        ax.bar(x + (i - len(modes)/2 + 0.5) * width, vals, width, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("best baseline vs best query")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def extract_mode_layer_perlabel(summary: dict):
    """
    returns:
      all_layers: sorted unique layer indices
      perlabel[mode][layer] = list of len=6 (who..how)
    """
    perlabel = defaultdict(dict)
    mode_to_layers = defaultdict(set)

    for k, v in summary.items():
        if not isinstance(k, str) or "/" not in k:
            continue
        parts = k.split("/")
        if len(parts) != 2 or not parts[1].startswith("layer_"):
            continue

        mode = parts[0]
        layer_str = parts[1]
        layer = int(layer_str.replace("layer_", ""))
        m = get_best_metrics(v)
        pl = m.get("per_label_f1", None)
        if isinstance(pl, list) and len(pl) == len(DIRS):
            perlabel[mode][layer] = [float(x) for x in pl]
            mode_to_layers[mode].add(layer)

    all_layers = sorted(set().union(*mode_to_layers.values()))
    return all_layers, perlabel

def plot_B_per_label_f1_vs_layer(summary: dict, out_path: Path):
    all_layers, perlabel = extract_mode_layer_perlabel(summary)
    if not all_layers:
        print("Warning: No layer results found for per_label_f1 in summary.json. Skipping B3 plot.")
        return

    n_dirs = len(DIRS)
    nrows = (n_dirs + 2) // 3
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), constrained_layout=True)
    axes = axes.flatten()

    for i, d in enumerate(DIRS):
        ax = axes[i]
        for mode in perlabel.keys():
            xs, ys = [], []
            for L in all_layers:
                if L in perlabel.get(mode, {}):
                    xs.append(L)
                    ys.append(perlabel[mode][L][i])
            if xs:
                ax.plot(xs, ys, marker="o", label=mode)
        ax.set_title(f"per-label F1: {d}")
        ax.set_xlabel("layer_idx")
        ax.set_ylabel("F1")
        ax.set_xticks(all_layers)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 余った subplot を非表示にする
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle("baseline vs query: per-label F1 across layers", fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def plot_B_best_per_label_bar(summary: dict, out_path: Path):
    best_blocks = extract_best_blocks(summary)
    modes = sorted(best_blocks.keys())

    x = np.arange(len(DIRS))
    width = 0.8 / max(1, len(modes))

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for i, mode in enumerate(modes):
        pl = best_blocks[mode].get("per_label_f1", [0.0]*len(DIRS))
        pl = [float(x) for x in pl]
        ax.bar(x + (i - len(modes)/2 + 0.5) * width, pl, width, label=f"{mode}(best)")
    ax.set_xticks(x)
    ax.set_xticklabels(DIRS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.set_title("best baseline vs best query: per-label F1")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--dev_jsonl", type=str, required=True)
    ap.add_argument("--summary_json", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # optional: match training filter (actual used distribution)
    ap.add_argument("--phases", type=str, default="", help="e.g. '1' or '1,2' ; empty=all")
    ap.add_argument("--levels", type=str, default="", help="e.g. '0' or '0,1' ; empty=all")
    ap.add_argument("--k_values", type=str, default="", help="e.g. '3' or '3,5' ; empty=all (only phase2)")

    ap.add_argument("--cooc_norm", type=str, default="rate", choices=["rate", "count"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    train_rows = read_jsonl(Path(args.train_jsonl))
    dev_rows   = read_jsonl(Path(args.dev_jsonl))
    summary    = load_summary(Path(args.summary_json))

    phases = parse_int_list(args.phases)
    levels = parse_int_list(args.levels)
    k_values = parse_int_list(args.k_values)

    # apply same filter as training if specified
    train_rows_use = filter_rows(train_rows, phases=phases, levels=levels, k_values=k_values)
    dev_rows_use   = filter_rows(dev_rows,   phases=phases, levels=levels, k_values=k_values)

    # A (MAIN)
    plot_A1_support_counts(
        train_rows_use, dev_rows_use,
        out_path=out_dir / "A1_train_vs_dev_label_support_counts.png",
    )
    plot_A2_support_rates(
        train_rows_use, dev_rows_use,
        out_path=out_dir / "A2_train_vs_dev_label_support_rates.png",
    )
    plot_A3_cooccurrence_heatmap(
        train_rows_use, dev_rows_use,
        out_path=out_dir / "A3_train_vs_dev_label_cooccurrence_heatmap.png",
        normalize=args.cooc_norm,
    )

    # B (existing)
    plot_B_mode_layer_metrics(summary, out_path=out_dir / "B1_mode_layer_metrics.png")
    plot_B_best_bars(summary, out_path=out_dir / "B2_best_compare.png")

    # B (NEW)
    plot_B_per_label_f1_vs_layer(summary, out_path=out_dir / "B3_per_label_f1_vs_layer.png")
    plot_B_best_per_label_bar(summary, out_path=out_dir / "B4_best_per_label_f1.png")

    print("[done] wrote:")
    print(" -", out_dir / "A1_train_vs_dev_label_support_counts.png")
    print(" -", out_dir / "A2_train_vs_dev_label_support_rates.png")
    print(" -", out_dir / "A3_train_vs_dev_label_cooccurrence_heatmap.png")
    print(" -", out_dir / "B1_mode_layer_metrics.png")
    print(" -", out_dir / "B2_best_compare.png")
    print(" -", out_dir / "B3_per_label_f1_vs_layer.png")
    print(" -", out_dir / "B4_best_per_label_f1.png")
    print_A1_A2_numbers(train_rows, dev_rows)
    #print_B2_numbers(summary)
    #print_B4_numbers(summary)

if __name__ == "__main__":
    main()
