import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

sys.path.append(os.getcwd())
try:
    from scripts.common import DIRS
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_probe", "scripts/archive/03_train_probe.py")
    train_probe = importlib.util.module_from_spec(spec)
    sys.modules["train_probe"] = train_probe
    spec.loader.exec_module(train_probe)
    micro_macro_f1 = train_probe.micro_macro_f1
    tune_threshold_per_class = train_probe.tune_threshold_per_class
    eval_with_per_class_threshold = train_probe.eval_with_per_class_threshold
except Exception as e:
    print(f"Cannot import requirements: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ProbeModelBase(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            self.lm = AutoModel.from_pretrained(
                model_name,
                output_hidden_states=True,
                torch_dtype=dtype,
                trust_remote_code=True
            )
        except OSError:
            self.lm = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        for p in self.lm.parameters():
            p.requires_grad = False

        self.hidden_size = int(self.lm.config.hidden_size)

    def get_layer_hidden(self, hidden_states, layer_idx: int) -> torch.Tensor:
        idx = layer_idx + 1 if layer_idx >= 0 else len(hidden_states) + layer_idx
        idx = max(0, min(idx, len(hidden_states) - 1))
        return hidden_states[idx]


class EosPoolingProbe(nn.Module):
    def __init__(self, base: ProbeModelBase):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.hidden_size, len(DIRS)).to(dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.base.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = self.base.get_layer_hidden(out.hidden_states, layer_idx)
        lengths = attention_mask.long().sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0)
        bsz = hs.size(0)
        h_last = hs[torch.arange(bsz, device=hs.device), lengths]
        return self.head(h_last.to(torch.float32))


# ---------------------------------------------------------------------------
# Dataset: now preserves Case Grammar metadata
# ---------------------------------------------------------------------------
class PairedDirUncDataset(Dataset):
    """Pairs up filled/missing rows and optionally retains CG metadata."""

    def __init__(self, rows):
        pairs = {}
        for r in rows:
            pid = r["id"].rsplit("::", 1)[0]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][r["condition"]] = r

        self.pairs = []
        for pid, p in pairs.items():
            if "filled" in p and "missing" in p:
                meta = p["missing"].get("metadata", {})
                self.pairs.append({
                    "filled_text":  p["filled"]["text"],
                    "missing_text": p["missing"]["text"],
                    "y_missing":    p["missing"]["labels"],
                    "base_id":      pid,
                    # Case Grammar metadata (empty if not CG dataset)
                    "case_role":        meta.get("case_role", ""),
                    "saturation_score": float(meta.get("saturation_score", -1.0)),
                    "is_saturated":     bool(meta.get("is_saturated", False)),
                    "predicate":        meta.get("predicate", ""),
                    "theme_domain":     meta.get("theme_domain", ""),
                    "dataset_name":    pid.split("::")[0] if "::" in pid else "unknown"
                })

        # Statistics
        stats = defaultdict(int)
        for item in self.pairs:
            stats[item["dataset_name"]] += 1
        print(f"  Total pairs loaded: {len(self.pairs)}")
        for ds, count in sorted(stats.items()):
            print(f"    - {ds:12s}: {count:5d} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        y_vec = torch.tensor(
            [float(item["y_missing"][d]) for d in DIRS], dtype=torch.float32
        )
        return {
            "filled_text":      item["filled_text"],
            "missing_text":     item["missing_text"],
            "y":                y_vec,
            "case_role":        item["case_role"],
            "saturation_score": item["saturation_score"],
            "is_saturated":     item["is_saturated"],
            "dataset_name":     item["dataset_name"],
        }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collate_paired_batch(tokenizer, batch, max_length):
    text_f = [b["filled_text"]  for b in batch]
    text_m = [b["missing_text"] for b in batch]
    y      = torch.stack([b["y"] for b in batch])

    enc_f = tokenizer(text_f, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    enc_m = tokenizer(text_m, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")

    return {
        "f_input_ids":       enc_f["input_ids"],
        "f_attention_mask":  enc_f["attention_mask"],
        "m_input_ids":       enc_m["input_ids"],
        "m_attention_mask":  enc_m["attention_mask"],
        "y":                 y,
        "case_role":        [b["case_role"]        for b in batch],
        "saturation_score": [b["saturation_score"] for b in batch],
        "is_saturated":     [b["is_saturated"]     for b in batch],
        "dataset_name":     [b["dataset_name"]     for b in batch],
    }


# ---------------------------------------------------------------------------
# Case-Grammar aware evaluation
# ---------------------------------------------------------------------------
def _pair_accuracy(y_true, p_pred_m, p_pred_f, thresholds):
    """Standard and strict pair accuracy over all active slots."""
    mask = (y_true == 1)
    total = mask.sum()
    if total == 0:
        return 0.0, 0.0
    thresh_bc = np.broadcast_to(thresholds, y_true.shape)
    correct_std = (p_pred_m[mask] > p_pred_f[mask]).sum()
    correct_str = (
        (p_pred_m[mask] >= thresh_bc[mask]) &
        (p_pred_f[mask] <  thresh_bc[mask])
    ).sum()
    return float(correct_std / total), float(correct_str / total)


def evaluate_cg(
    model, dl, device, layer_idx,
    enable_cg: bool = True,
):
    """
    Full evaluation with:
      - Standard macro/micro F1 (threshold-tuned)
      - Pair accuracy (standard + strict)
      - [CG] Per-case-role pair accuracy
      - [CG] Per-saturation-bin pair accuracy
    """
    model.eval()

    ys, ps_m, ps_f = [], [], []
    roles_list, sat_scores, is_sat_list = [], [], []
    ds_list = []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for batch in dl:
            f_ids  = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            m_ids  = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            y      = batch["y"].to(device)

            logits_f = model(f_ids, f_mask, layer_idx)
            logits_m = model(m_ids, m_mask, layer_idx)

            prob_f = torch.sigmoid(logits_f)
            prob_m = torch.sigmoid(logits_m)

            ys.append(y.cpu().numpy())
            ps_m.append(prob_m.cpu().numpy())
            ps_f.append(prob_f.cpu().numpy())

            if enable_cg:
                roles_list.extend(batch["case_role"])
                sat_scores.extend(batch["saturation_score"])
                is_sat_list.extend(batch["is_saturated"])
                ds_list.extend(batch["dataset_name"])

            y_zero = torch.zeros_like(y)
            loss_f = F.binary_cross_entropy_with_logits(logits_f, y_zero)
            loss_m = F.binary_cross_entropy_with_logits(logits_m, y)
            total_loss += float((loss_f + loss_m).item()) * y.size(0)
            n += y.size(0)

    if not ys:
        return {}

    y_true   = np.concatenate(ys, axis=0)    # (N, C)
    p_pred_m = np.concatenate(ps_m, axis=0)  # (N, C)
    p_pred_f = np.concatenate(ps_f, axis=0)  # (N, C)

    tuned     = tune_threshold_per_class(y_true, p_pred_m, grid=None)
    thresholds = np.array(tuned["thresholds"])  # (C,)
    metrics   = eval_with_per_class_threshold(y_true, p_pred_m, thresholds)
    metrics["threshold_dict"] = tuned.get("threshold_dict", {})


    std_acc, str_acc = _pair_accuracy(y_true, p_pred_m, p_pred_f, thresholds)
    metrics["pair_accuracy_standard"] = std_acc
    metrics["pair_accuracy_strict"]   = str_acc
    metrics["loss"] = total_loss / max(1, n)

    # ----------------------------------------------------------------
    # Case-Grammar analysis
    # ----------------------------------------------------------------
    if enable_cg and roles_list:
        roles_arr = np.array(roles_list)       # (N,)
        sat_arr   = np.array(sat_scores, dtype=float)  # (N,)
        is_sat_arr = np.array(is_sat_list, dtype=bool) # (N,)

        # --- Per case-role pair accuracy ---
        role_acc: dict = {}
        for role in sorted(set(roles_arr)):
            if not role:
                continue
            ridx = np.where(roles_arr == role)[0]
            if len(ridx) == 0:
                continue
            y_r  = y_true[ridx];   pm_r = p_pred_m[ridx];   pf_r = p_pred_f[ridx]
            acc_s, acc_str = _pair_accuracy(y_r, pm_r, pf_r, thresholds)
            # Per-role macro F1
            r_metrics = eval_with_per_class_threshold(y_r, pm_r, thresholds)
            role_acc[role] = {
                "n":                 int(len(ridx)),
                "pair_acc_standard": round(acc_s,   4),
                "pair_acc_strict":   round(acc_str,  4),
                "macro_f1":          round(r_metrics.get("macro_f1_posonly", 0.0), 4),
            }
        metrics["by_case_role"] = role_acc

        # --- Per saturation bin pair accuracy ---
        # Bins: [0,0.2), [0.2,0.4), ... [0.8,1.0], plus a "saturated" bin
        bin_acc: dict = {}
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        for lo, hi in bins:
            bidx = np.where((sat_arr >= lo) & (sat_arr < hi) & (sat_arr >= 0))[0]
            if len(bidx) == 0:
                continue
            y_b = y_true[bidx]; pm_b = p_pred_m[bidx]; pf_b = p_pred_f[bidx]
            acc_s, acc_str = _pair_accuracy(y_b, pm_b, pf_b, thresholds)
            bin_acc[f"sat_{lo:.1f}_{hi:.1f}"] = {
                "n":                 int(len(bidx)),
                "pair_acc_standard": round(acc_s,   4),
                "pair_acc_strict":   round(acc_str,  4),
            }
        # Binary saturation: is_saturated=True (all mandatory roles filled)
        for flag, label in [(True, "sat_full"), (False, "sat_partial")]:
            fidx = np.where(is_sat_arr == flag)[0]
            if len(fidx) == 0:
                continue
            y_f = y_true[fidx]; pm_f = p_pred_m[fidx]; pf_f = p_pred_f[fidx]
            acc_s, acc_str = _pair_accuracy(y_f, pm_f, pf_f, thresholds)
            bin_acc[label] = {
                "n":                 int(len(fidx)),
                "pair_acc_standard": round(acc_s,   4),
                "pair_acc_strict":   round(acc_str,  4),
            }
        metrics["by_saturation"] = bin_acc

        # --- Per dataset pair accuracy ---
        ds_acc: dict = {}
        ds_arr = np.array(ds_list)
        for ds in sorted(set(ds_arr)):
            didx = np.where(ds_arr == ds)[0]
            if len(didx) == 0: continue
            y_d = y_true[didx]; pm_d = p_pred_m[didx]; pf_d = p_pred_f[didx]
            acc_s, acc_str = _pair_accuracy(y_d, pm_d, pf_d, thresholds)
            ds_acc[ds] = {
                "n": int(len(didx)),
                "pair_acc_standard": round(acc_s, 4),
            }
        metrics["by_dataset"] = ds_acc

    return metrics


def _print_cg_metrics(metrics: dict) -> None:
    """Pretty-print the Case Grammar analysis sections."""
    if "by_case_role" in metrics:
        print("\n  [CG] Per-case-role pair accuracy:")
        for role, v in sorted(metrics["by_case_role"].items()):
            print(f"    {role:12s}  n={v['n']:5d}  "
                  f"PairAcc(std)={v['pair_acc_standard']:.3f}  "
                  f"MacroF1={v['macro_f1']:.3f}")

    if "by_saturation" in metrics:
        print("\n  [CG] Per-saturation-bin pair accuracy:")
        for bin_key, v in sorted(metrics["by_saturation"].items()):
            print(f"    {bin_key:20s}  n={v['n']:5d}  "
                  f"PairAcc(std)={v['pair_acc_standard']:.3f}  "
                  f"strict={v['pair_acc_strict']:.3f}")

    if "by_dataset" in metrics:
        print("\n  [CG] Per-dataset pair accuracy:")
        for ds, v in sorted(metrics["by_dataset"].items()):
            print(f"    {ds:12s}  n={v['n']:5d}  "
                  f"PairAcc(std)={v['pair_acc_standard']:.3f}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",    type=str,   default="google/gemma-2-2b-it")
    parser.add_argument("--layer_idx",     type=int,   default=16)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--max_length",    type=int,   default=256)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--margin",        type=float, default=0.2)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--train_data",    type=str,   required=True,
                        help="Comma-separated paths to JSONL (can mix CG and legacy)")
    parser.add_argument("--dev_data",      type=str,   required=True)
    parser.add_argument("--out_dir",       type=str,   default="runs/contrastive_probe")
    parser.add_argument("--no_cg_eval",   action="store_true",
                        help="Disable Case-Grammar sub-analyses (for legacy datasets)")
    parser.add_argument("--resume",       action="store_true", help="Resume from checkpoint")
    parser.add_argument("--start_epoch",  type=int, default=1, help="Epoch to start from")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base  = ProbeModelBase(args.model_name).to(device)
    model = EosPoolingProbe(base).to(device)

    train_rows = []
    for p in args.train_data.split(","):
        train_rows.extend(read_jsonl(Path(p.strip())))

    dev_rows = []
    for p in args.dev_data.split(","):
        dev_rows.extend(read_jsonl(Path(p.strip())))

    train_ds = PairedDirUncDataset(train_rows)
    dev_ds   = PairedDirUncDataset(dev_rows)

    if len(train_ds) == 0 or len(dev_ds) == 0:
        print("Empty dataset!")
        return

    enable_cg = not args.no_cg_eval

    collate_fn = lambda b: collate_paired_batch(tokenizer, b, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  collate_fn=collate_fn)
    dev_dl   = DataLoader(dev_ds,   batch_size=args.batch_size,
                          shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"best_probe_layer{args.layer_idx}.pt"
    best_score = -1.0

    if args.resume:
        if best_path.exists():
            print(f"Resuming from {best_path}")
            model.head.load_state_dict(torch.load(best_path, map_location=device))
        else:
            print(f"Warning: {best_path} not found. Starting from scratch.")

    print(f"Training pairs: {len(train_ds)}, Dev pairs: {len(dev_ds)}")
    print(f"CG analysis: {'enabled' if enable_cg else 'disabled'}")

    for ep in range(args.start_epoch, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for batch in tqdm(train_dl, desc=f"Ep {ep}"):
            f_ids  = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            m_ids  = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            y      = batch["y"].to(device)

            logits_f = model(f_ids, f_mask, args.layer_idx)
            logits_m = model(m_ids, m_mask, args.layer_idx)

            # BCE Loss (absolute state anchoring)
            y_zero     = torch.zeros_like(y)
            loss_bce_f = F.binary_cross_entropy_with_logits(logits_f, y_zero)
            loss_bce_m = F.binary_cross_entropy_with_logits(logits_m, y)

            # Margin Loss (contrastive separation)
            prob_f = torch.sigmoid(logits_f)
            prob_m = torch.sigmoid(logits_m)
            mask_active = (y == 1)
            if mask_active.any():
                diff = prob_m[mask_active] - prob_f[mask_active]
                loss_margin = torch.mean(F.relu(args.margin - diff))
            else:
                loss_margin = torch.tensor(0.0, device=device)

            loss = loss_bce_f + loss_bce_m + args.lambda_margin * loss_margin

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += float(loss.item()) * y.size(0)
            n += y.size(0)

        train_loss = total_loss / max(1, n)

        dev_metrics = evaluate_cg(model, dev_dl, device, args.layer_idx,
                                  enable_cg=enable_cg)

        macro_f1  = dev_metrics.get("macro_f1_posonly", 0.0)
        std_acc   = dev_metrics.get("pair_accuracy_standard", 0.0)
        str_acc   = dev_metrics.get("pair_accuracy_strict",   0.0)
        dev_loss  = dev_metrics.get("loss", 0.0)

        print(f"\nEp {ep} | TrainLoss={train_loss:.4f} | DevLoss={dev_loss:.4f} | "
              f"MacroF1={macro_f1:.4f} | StdAcc={std_acc:.4f} | StrictAcc={str_acc:.4f}")

        if enable_cg:
            _print_cg_metrics(dev_metrics)

        # Model selection: standard pair accuracy + MacroF1
        score = std_acc + macro_f1
        if score > best_score:
            best_score = score
            torch.save(model.head.state_dict(), best_path)

        # Log (serialize nested dicts as JSON strings for log.jsonl)
        log_rec = {
            "epoch":      ep,
            "train_loss": train_loss,
            "loss":       dev_loss,
            "macro_f1":   macro_f1,
            "pair_accuracy_standard": std_acc,
            "pair_accuracy_strict":   str_acc,
            "threshold_dict": dev_metrics.get("threshold_dict", {}),
        }
        if enable_cg:
            log_rec["by_case_role"]  = dev_metrics.get("by_case_role",  {})
            log_rec["by_saturation"] = dev_metrics.get("by_saturation", {})
            log_rec["by_dataset"]   = dev_metrics.get("by_dataset",   {})

        with (out_dir / "log.jsonl").open("a") as f:
            f.write(json.dumps(log_rec) + "\n")

    print(f"\nBest model saved to {best_path}")


if __name__ == "__main__":
    main()
