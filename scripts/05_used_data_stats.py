# scripts/05_used_data_stats.py
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

DIRS = ["who", "what", "when", "where", "why", "how"]

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def parse_int_list(s: str):
    s=s.strip()
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def filter_rows(rows, phases=None, levels=None, k_values=None, services=None):
    ss = set(services) if services else None
    for r in rows:
        ph = int(r.get("phase"))
        lv = int(r.get("level"))
        kv = r.get("k", None)
        kv_int = None if kv is None else int(kv)
        if phases is not None and ph not in phases:
            continue
        if levels is not None and lv not in levels:
            continue
        if k_values is not None:
            # Phase2 のときだけ k を見る（Phase1 は通す）
            if ph == 2 and (kv_int not in k_values):
                continue
        if ss is not None and str(r.get("service","")) not in ss:
            continue
        yield r

def summarize(rows):
    n = 0
    svc_cnt = Counter()
    label_pos = Counter()
    svc_label_pos = defaultdict(Counter)  # svc -> Counter(dir->pos)
    phlv = Counter()

    for r in rows:
        n += 1
        svc = str(r.get("service",""))
        svc_cnt[svc] += 1

        phlv[(int(r.get("phase")), int(r.get("level")))] += 1

        labels = r.get("labels", {})
        for d in DIRS:
            v = int(labels.get(d, 0))
            label_pos[d] += v
            svc_label_pos[svc][d] += v

    # overall label pos rates
    label_rates = {d: (label_pos[d] / n if n else 0.0) for d in DIRS}

    # per service pos rates
    svc_pos_rates = []
    for svc, c in svc_cnt.most_common():
        rates = {d: (svc_label_pos[svc][d] / c if c else 0.0) for d in DIRS}
        svc_pos_rates.append({"service": svc, "count": c, "pos_rates": rates})

    return {
        "n_rows": n,
        "phase_level_counts": [{"phase": p, "level": l, "count": c} for (p,l),c in sorted(phlv.items())],
        "service_distribution": [{"service": s, "count": c, "rate": c/n if n else 0.0} for s,c in svc_cnt.most_common()],
        "label_pos_rates": label_rates,
        "service_label_pos_rates": svc_pos_rates,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, required=True)
    ap.add_argument("--dev", type=str, required=True)
    ap.add_argument("--phases", type=str, default="")
    ap.add_argument("--levels", type=str, default="")
    ap.add_argument("--k_values", type=str, default="")
    ap.add_argument("--services", type=str, default="")  # optional: comma separated
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    phases = parse_int_list(args.phases)
    levels = parse_int_list(args.levels)
    k_values = parse_int_list(args.k_values)
    services = [s.strip() for s in args.services.split(",") if s.strip()] or None

    train_rows = filter_rows(read_jsonl(Path(args.train)), phases, levels, k_values, services)
    dev_rows   = filter_rows(read_jsonl(Path(args.dev)), phases, levels, k_values, services)

    out = {
        "filter": {"phases": phases, "levels": levels, "k_values": k_values, "services": services},
        "train_used": summarize(train_rows),
        "dev_used": summarize(dev_rows),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out}")

if __name__ == "__main__":
    main()
