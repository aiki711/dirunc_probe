# scripts/04_viz_generated_plus.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DIRS = ["who", "what", "when", "where", "why", "how"]


# ---------------- IO ----------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def savefig(out_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    recs = []
    for r in rows:
        labels = r.get("labels", {}) or {}
        text = str(r.get("text", ""))
        rec = {
            "service": str(r.get("service", "")),
            "phase": int(r.get("phase", -1)),
            "level": int(r.get("level", -1)),
            "k": (None if r.get("k", None) is None else int(r.get("k"))),
            "text_len": len(text),
        }
        for d in DIRS:
            rec[d] = int(labels.get(d, 0))
        recs.append(rec)
    return pd.DataFrame.from_records(recs)


# ---------------- Core stats ----------------

def service_counts(df: pd.DataFrame) -> pd.Series:
    return df["service"].value_counts(dropna=False)


def service_dist(df: pd.DataFrame) -> pd.Series:
    c = service_counts(df).astype(float)
    s = float(c.sum()) if float(c.sum()) > 0 else 1.0
    return c / s


def per_service_label_rates(df: pd.DataFrame) -> pd.DataFrame:
    # service x label
    g = df.groupby("service")[DIRS].mean()
    g["n"] = df.groupby("service").size().astype(int)
    g = g.sort_values("n", ascending=False)
    return g


def overall_label_rates(df: pd.DataFrame) -> pd.Series:
    return df[DIRS].mean()


def summarize_split(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_services": int(df["service"].nunique()),
        "phase_values": sorted(df["phase"].unique().tolist()),
        "level_values": sorted(df["level"].unique().tolist()),
        "pos_counts": {d: int(df[d].sum()) for d in DIRS},
        "pos_rates": {d: float(df[d].mean()) if len(df) else 0.0 for d in DIRS},
        "text_len": {
            "mean": float(df["text_len"].mean()) if len(df) else 0.0,
            "p50": float(df["text_len"].quantile(0.50)) if len(df) else 0.0,
            "p90": float(df["text_len"].quantile(0.90)) if len(df) else 0.0,
            "max": int(df["text_len"].max()) if len(df) else 0,
        },
    }
    return out


# ---------------- Plots: data distribution ----------------

def plot_service_topk_counts(train_c: pd.Series, dev_c: pd.Series, title: str, out_path: Path, top_k: int = 30) -> None:
    services = list(pd.Index(train_c.index).union(dev_c.index))
    # choose top_k by combined count
    combined = (train_c.reindex(services, fill_value=0) + dev_c.reindex(services, fill_value=0)).sort_values(ascending=False)
    top = combined.head(top_k).index.tolist()

    tr = train_c.reindex(top, fill_value=0).astype(float).values
    dv = dev_c.reindex(top, fill_value=0).astype(float).values

    x = np.arange(len(top))
    width = 0.45
    plt.figure(figsize=(12, max(4, 0.30 * len(top))))
    plt.barh(x - width/2, tr, height=0.40, label="train")
    plt.barh(x + width/2, dv, height=0.40, label="dev")
    plt.yticks(x, top)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("count")
    plt.legend()
    savefig(out_path)


def plot_service_topk_dist(train_p: pd.Series, dev_p: pd.Series, title: str, out_path: Path, top_k: int = 30) -> None:
    services = list(pd.Index(train_p.index).union(dev_p.index))
    combined = (train_p.reindex(services, fill_value=0) + dev_p.reindex(services, fill_value=0)).sort_values(ascending=False)
    top = combined.head(top_k).index.tolist()

    tr = train_p.reindex(top, fill_value=0).astype(float).values
    dv = dev_p.reindex(top, fill_value=0).astype(float).values

    x = np.arange(len(top))
    width = 0.45
    plt.figure(figsize=(12, max(4, 0.30 * len(top))))
    plt.barh(x - width/2, tr, height=0.40, label="train")
    plt.barh(x + width/2, dv, height=0.40, label="dev")
    plt.yticks(x, top)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("probability (service mixture)")
    plt.legend()
    savefig(out_path)


def plot_label_prevalence_compare(train_rates: pd.Series, dev_rates: pd.Series, title: str, out_path: Path) -> None:
    tr = train_rates.reindex(DIRS).values
    dv = dev_rates.reindex(DIRS).values
    x = np.arange(len(DIRS))
    width = 0.45
    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, tr, width=width, label="train")
    plt.bar(x + width/2, dv, width=width, label="dev")
    plt.xticks(x, DIRS)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("positive rate")
    plt.legend()
    savefig(out_path)


def plot_service_label_heatmap(mat: pd.DataFrame, title: str, out_path: Path, top_k: int = 30, vmin: float = 0.0, vmax: float = 1.0) -> None:
    # mat: service x DIRS (and may have 'n')
    use = mat.copy()
    if "n" in use.columns:
        use = use.sort_values("n", ascending=False)
        use = use.head(top_k)
        use = use[DIRS]
    else:
        use = use.head(top_k)

    arr = use.values
    plt.figure(figsize=(10, max(4, 0.35 * len(use))))
    plt.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="positive rate")
    plt.yticks(range(len(use)), use.index.tolist())
    plt.xticks(range(len(DIRS)), DIRS)
    plt.title(title)
    savefig(out_path)


def plot_service_label_diff_heatmap(train_rates: pd.DataFrame, dev_rates: pd.DataFrame, title: str, out_path: Path, top_k: int = 30) -> None:
    # align on union services, choose top_k by combined n
    tr = train_rates.copy()
    dv = dev_rates.copy()
    tr_n = tr["n"] if "n" in tr.columns else pd.Series(0, index=tr.index)
    dv_n = dv["n"] if "n" in dv.columns else pd.Series(0, index=dv.index)
    services = pd.Index(tr.index).union(dv.index)

    comb_n = tr_n.reindex(services, fill_value=0) + dv_n.reindex(services, fill_value=0)
    top = comb_n.sort_values(ascending=False).head(top_k).index

    tr_lab = tr.reindex(top, fill_value=0.0)[DIRS]
    dv_lab = dv.reindex(top, fill_value=0.0)[DIRS]
    diff = dv_lab - tr_lab  # dev - train

    arr = diff.values
    vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
    vmax = max(vmax, 1e-6)

    plt.figure(figsize=(10, max(4, 0.35 * len(diff))))
    plt.imshow(arr, aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="dev - train (positive rate)")
    plt.yticks(range(len(diff)), diff.index.tolist())
    plt.xticks(range(len(DIRS)), DIRS)
    plt.title(title)
    savefig(out_path)


# ---------------- Mixture effect (counterfactual) ----------------

def counterfactual_overall_rates(
    train_srv_p: pd.Series,
    dev_srv_p: pd.Series,
    train_srv_rates: pd.DataFrame,
    dev_srv_rates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute:
      - observed overall label rate (train, dev)
      - counterfactual dev rates under train service mixture
      - counterfactual train rates under dev service mixture
    """
    services = pd.Index(train_srv_p.index).union(dev_srv_p.index)

    # weights
    w_tr = train_srv_p.reindex(services, fill_value=0.0).astype(float)
    w_dv = dev_srv_p.reindex(services, fill_value=0.0).astype(float)

    # per-service label rates (missing -> 0)
    trR = train_srv_rates.reindex(services, fill_value=0.0)[DIRS].astype(float)
    dvR = dev_srv_rates.reindex(services, fill_value=0.0)[DIRS].astype(float)

    # weighted averages
    obs_train = (w_tr.values[:, None] * trR.values).sum(axis=0)
    obs_dev   = (w_dv.values[:, None] * dvR.values).sum(axis=0)

    # counterfactuals
    dev_under_train_mix = (w_tr.values[:, None] * dvR.values).sum(axis=0)
    train_under_dev_mix = (w_dv.values[:, None] * trR.values).sum(axis=0)

    out = pd.DataFrame({
        "observed_train": obs_train,
        "observed_dev": obs_dev,
        "dev_under_train_mix": dev_under_train_mix,
        "train_under_dev_mix": train_under_dev_mix,
    }, index=DIRS)

    out["dev_minus_train_observed"] = out["observed_dev"] - out["observed_train"]
    out["dev_mix_effect_only"] = out["dev_under_train_mix"] - out["observed_dev"]  # how dev changes if mix replaced
    out["train_mix_effect_only"] = out["train_under_dev_mix"] - out["observed_train"]
    return out


def plot_counterfactual_table_bar(cf: pd.DataFrame, title: str, out_path: Path) -> None:
    # grouped bar per label: observed_train, observed_dev, dev_under_train_mix
    x = np.arange(len(DIRS))
    width = 0.28
    plt.figure(figsize=(12, 4))
    plt.bar(x - width, cf.loc[DIRS, "observed_train"].values, width=width, label="observed_train")
    plt.bar(x,         cf.loc[DIRS, "observed_dev"].values,   width=width, label="observed_dev")
    plt.bar(x + width, cf.loc[DIRS, "dev_under_train_mix"].values, width=width, label="dev_under_train_mix")
    plt.xticks(x, DIRS)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("positive rate")
    plt.legend()
    savefig(out_path)


# ---------------- Probe metrics (03 output) ----------------

def read_probe_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_best_records(probe_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Support both:
      - old: best_overall has micro_f1 etc
      - new: best_overall has macro_f1_posonly etc
    """
    out: Dict[str, Dict[str, Any]] = {}

    # common keys present in your summaries
    for k in ["baseline/best", "query/best", "best_overall"]:
        if k in probe_summary and isinstance(probe_summary[k], dict):
            rec = probe_summary[k]
            # sometimes nested {best:{...}} for baseline/best
            if "best" in rec and isinstance(rec["best"], dict):
                out[k] = rec["best"]
            else:
                out[k] = rec
    return out


def plot_probe_overall(best_records: Dict[str, Dict[str, Any]], title: str, out_path: Path) -> None:
    # bar chart: micro and macro (if present)
    labels = list(best_records.keys())
    micros = [float(best_records[k].get("micro_f1", np.nan)) for k in labels]
    macros = [float(best_records[k].get("macro_f1_posonly", np.nan)) for k in labels]

    x = np.arange(len(labels))
    width = 0.40
    plt.figure(figsize=(12, 4))
    if not np.all(np.isnan(micros)):
        plt.bar(x - width/2, micros, width=width, label="micro_f1")
    if not np.all(np.isnan(macros)):
        plt.bar(x + width/2, macros, width=width, label="macro_f1_posonly")
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("F1")
    plt.legend()
    savefig(out_path)


def plot_probe_per_label(best_records: Dict[str, Dict[str, Any]], title: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    for k, rec in best_records.items():
        per = rec.get("per_label_f1", None)
        if per is None:
            continue
        if not isinstance(per, (list, tuple)) or len(per) != len(DIRS):
            continue
        plt.figure(figsize=(10, 4))
        plt.bar(DIRS, [float(x) for x in per])
        plt.ylim(0, 1)
        plt.title(f"{title}: {k}")
        plt.ylabel("F1")
        savefig(out_dir / f"per_label_f1__{k.replace('/','_')}.png")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, required=True, help="dir containing train/dev jsonl (04 output)")
    ap.add_argument("--train_file", type=str, default="train.jsonl")
    ap.add_argument("--dev_file", type=str, default="dev.jsonl")

    ap.add_argument("--out_dir", type=str, default="runs/dirunc_viz_plus")
    ap.add_argument("--top_k_services", type=int, default=30)

    # optional: 03 probe summary.json path
    ap.add_argument("--probe_summary", type=str, default="",
                    help="path to runs/.../summary.json produced by scripts/03_train_probe.py")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # ---- load data ----
    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / args.train_file)
    dev_rows = read_jsonl(data_dir / args.dev_file)

    df_tr = rows_to_df(train_rows)
    df_dv = rows_to_df(dev_rows)

    # ---- summaries ----
    summ = {"train": summarize_split(df_tr), "dev": summarize_split(df_dv)}
    (out_dir / "summary_data.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- service distributions ----
    tr_c = service_counts(df_tr)
    dv_c = service_counts(df_dv)
    tr_p = service_dist(df_tr)
    dv_p = service_dist(df_dv)

    plot_service_topk_counts(tr_c, dv_c, "service counts (train vs dev)", out_dir / "service_counts_topk.png", top_k=args.top_k_services)
    plot_service_topk_dist(tr_p, dv_p, "service mixture (train vs dev)", out_dir / "service_mixture_topk.png", top_k=args.top_k_services)

    # ---- direction prevalence (overall) ----
    plot_label_prevalence_compare(overall_label_rates(df_tr), overall_label_rates(df_dv),
                                  "direction positive rate (overall train vs dev)",
                                  out_dir / "label_prevalence_train_vs_dev.png")

    # ---- service-wise direction positive rates ----
    tr_srv = per_service_label_rates(df_tr)
    dv_srv = per_service_label_rates(df_dv)
    tr_srv.to_csv(out_dir / "service_label_rates_train.csv", index=True)
    dv_srv.to_csv(out_dir / "service_label_rates_dev.csv", index=True)

    plot_service_label_heatmap(tr_srv, "train: service x direction positive rate", out_dir / "service_heatmap_train.png", top_k=args.top_k_services)
    plot_service_label_heatmap(dv_srv, "dev: service x direction positive rate", out_dir / "service_heatmap_dev.png", top_k=args.top_k_services)
    plot_service_label_diff_heatmap(tr_srv, dv_srv, "service x direction rate diff (dev - train)", out_dir / "service_heatmap_dev_minus_train.png", top_k=args.top_k_services)

    # ---- mixture effect diagnostics ----
    cf = counterfactual_overall_rates(tr_p, dv_p, tr_srv, dv_srv)
    cf.to_csv(out_dir / "counterfactual_mixture_effect.csv", index=True)
    plot_counterfactual_table_bar(cf, "counterfactual: dev rates under train service mixture", out_dir / "counterfactual_dev_under_train_mix.png")

    # ---- optional: probe metrics ----
    if args.probe_summary:
        ps = read_probe_summary(Path(args.probe_summary))
        best_recs = extract_best_records(ps)

        probe_out = out_dir / "probe_metrics"
        ensure_dir(probe_out)

        (probe_out / "best_records.json").write_text(json.dumps(best_recs, ensure_ascii=False, indent=2), encoding="utf-8")

        plot_probe_overall(best_recs, "probe overall metrics (best)", probe_out / "overall_micro_macro.png")
        plot_probe_per_label(best_recs, "probe per-label F1 (best)", probe_out)

        # also dump dev_by_service if present
        # (best record may include it under rec["dev_by_service"])
        for k, rec in best_recs.items():
            if "dev_by_service" in rec and isinstance(rec["dev_by_service"], dict):
                # flatten to csv (macro/micro/per_label may exist)
                rows = []
                for svc, m in rec["dev_by_service"].items():
                    row = {"service": svc}
                    if isinstance(m, dict):
                        row.update({
                            "micro_f1": m.get("micro_f1", None),
                            "macro_f1_posonly": m.get("macro_f1_posonly", None),
                            "threshold": m.get("threshold", None),
                            "n_rows": m.get("n_rows", None),
                        })
                        per = m.get("per_label_f1", None)
                        if isinstance(per, list) and len(per) == len(DIRS):
                            for i, d in enumerate(DIRS):
                                row[f"f1_{d}"] = per[i]
                    rows.append(row)
                pd.DataFrame(rows).to_csv(probe_out / f"dev_by_service__{k.replace('/','_')}.csv", index=False)

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
