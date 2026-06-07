import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from collections import defaultdict
from typing import Any

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

try:
    from scripts.common import DIRS
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_probe", "scripts/archive/03_train_probe.py")
    train_probe = importlib.util.module_from_spec(spec)
    sys.modules["train_probe"] = train_probe
    spec.loader.exec_module(train_probe)
    tune_threshold_per_class = train_probe.tune_threshold_per_class
    eval_with_per_class_threshold = train_probe.eval_with_per_class_threshold
except Exception as e:
    print(f"Cannot import requirements: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CachedDataset(Dataset):
    def __init__(self, data):
        self.f_hs = data["f_hs"]
        self.m_hs = data["m_hs"]
        self.y = data["y"]
        self.metadata = data["metadata"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "f_hs": self.f_hs[idx],
            "m_hs": self.m_hs[idx],
            "y": self.y[idx],
            "case_role":        self.metadata[idx].get("case_role", ""),
            "saturation_score": float(self.metadata[idx].get("saturation_score", -1.0)),
            "is_saturated":     bool(self.metadata[idx].get("is_saturated", False)),
            "dataset_name":     self.metadata[idx].get("dataset_name", "unknown")
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LinearQueryProbe(nn.Module):
    def __init__(self, hidden_size: int, target_dtype: torch.dtype):
        super().__init__()
        self.W = nn.Parameter(torch.empty(len(DIRS), hidden_size, dtype=target_dtype))
        self.b = nn.Parameter(torch.zeros(len(DIRS), dtype=target_dtype))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        # hs shape: [batch_size, 7, hidden_size]
        # self.W shape: [7, hidden_size]
        # output shape: [batch_size, 7]
        return (hs * self.W).sum(dim=2) + self.b


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _pair_accuracy(y_true, p_pred_m, p_pred_f, thresholds):
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


def evaluate_cg_cached(model, dl, device, enable_cg: bool = True):
    model.eval()

    ys, ps_m, ps_f = [], [], []
    roles_list, sat_scores, is_sat_list = [], [], []
    ds_list = []
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for batch in dl:
            f_hs = batch["f_hs"].to(device)
            m_hs = batch["m_hs"].to(device)
            y    = batch["y"].to(device)

            logits_f = model(f_hs)
            logits_m = model(m_hs)

            prob_f = torch.sigmoid(logits_f)
            prob_m = torch.sigmoid(logits_m)

            ys.append(y.cpu().numpy())
            ps_m.append(prob_m.float().cpu().numpy())
            ps_f.append(prob_f.float().cpu().numpy())

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

    if enable_cg and roles_list:
        roles_arr = np.array(roles_list)
        sat_arr   = np.array(sat_scores, dtype=float)
        is_sat_arr = np.array(is_sat_list, dtype=bool)

        role_acc = {}
        for role in sorted(set(roles_arr)):
            if not role:
                continue
            ridx = np.where(roles_arr == role)[0]
            if len(ridx) == 0:
                continue
            y_r  = y_true[ridx];   pm_r = p_pred_m[ridx];   pf_r = p_pred_f[ridx]
            acc_s, acc_str = _pair_accuracy(y_r, pm_r, pf_r, thresholds)
            r_metrics = eval_with_per_class_threshold(y_r, pm_r, thresholds)
            role_acc[role] = {
                "n":                 int(len(ridx)),
                "pair_acc_standard": round(acc_s,   4),
                "pair_acc_strict":   round(acc_str,  4),
                "macro_f1":          round(r_metrics.get("macro_f1_posonly", 0.0), 4),
            }
        metrics["by_case_role"] = role_acc

        bin_acc = {}
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

        ds_acc = {}
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
    parser.add_argument("--cache_dir",     type=str,   default="data/cache")
    parser.add_argument("--prefix",        type=str,   required=True, choices=["soft", "strong"])
    parser.add_argument("--layer_idx",     type=int,   required=True)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=5)
    parser.add_argument("--lr",            type=float, default=5e-4)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--margin",        type=float, default=0.2)
    parser.add_argument("--lambda_margin", type=float, default=1.0)
    parser.add_argument("--out_dir",       type=str,   required=True)
    parser.add_argument("--no_cg_eval",    action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load cache files
    cache_path = Path(args.cache_dir)
    train_cache_path = cache_path / f"{args.prefix}_layer{args.layer_idx}_train.pt"
    dev_cache_path   = cache_path / f"{args.prefix}_layer{args.layer_idx}_dev.pt"

    if not train_cache_path.exists() or not dev_cache_path.exists():
        print(f"Error: Cache not found. {train_cache_path} or {dev_cache_path} is missing.")
        sys.exit(1)

    print(f"Loading cache from {train_cache_path} and {dev_cache_path}...")
    train_data = torch.load(train_cache_path, map_location="cpu")
    dev_data   = torch.load(dev_cache_path, map_location="cpu")

    train_ds = CachedDataset(train_data)
    dev_ds   = CachedDataset(dev_data)

    enable_cg = not args.no_cg_eval
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False)

    hidden_size = train_data["f_hs"].size(-1)
    target_dtype = train_data["f_hs"].dtype
    print(f"Loaded cache stats: Train pairs: {len(train_ds)}, Dev pairs: {len(dev_ds)}, Hidden Size: {hidden_size}, Dtype: {target_dtype}")

    model = LinearQueryProbe(hidden_size, target_dtype).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"best_probe_layer{args.layer_idx}.pt"
    best_score = -1.0

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0

        for batch in train_dl:
            f_hs = batch["f_hs"].to(device)
            m_hs = batch["m_hs"].to(device)
            y    = batch["y"].to(device)

            logits_f = model(f_hs)
            logits_m = model(m_hs)

            y_zero     = torch.zeros_like(y)
            loss_bce_f = F.binary_cross_entropy_with_logits(logits_f, y_zero)
            loss_bce_m = F.binary_cross_entropy_with_logits(logits_m, y)

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

        dev_metrics = evaluate_cg_cached(model, dev_dl, device, enable_cg=enable_cg)

        macro_f1  = dev_metrics.get("macro_f1_posonly", 0.0)
        std_acc   = dev_metrics.get("pair_accuracy_standard", 0.0)
        str_acc   = dev_metrics.get("pair_accuracy_strict",   0.0)
        dev_loss  = dev_metrics.get("loss", 0.0)

        print(f"\nEp {ep} | TrainLoss={train_loss:.4f} | DevLoss={dev_loss:.4f} | "
              f"MacroF1={macro_f1:.4f} | StdAcc={std_acc:.4f} | StrictAcc={str_acc:.4f}")

        if enable_cg:
            _print_cg_metrics(dev_metrics)

        score = std_acc + macro_f1
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)

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
