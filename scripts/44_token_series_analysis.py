#!/usr/bin/env python3
import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

# Import configurations and helpers dynamically from 32_train_contrastive_nq_probe
try:
    from scripts.common import DIRS, NATURAL_QUERY_MAP, NATURAL_QUERY_STR
    import importlib.util
    spec = importlib.util.spec_from_file_location("nq_probe", "scripts/32_train_contrastive_nq_probe.py")
    nq_probe = importlib.util.module_from_spec(spec)
    sys.modules["nq_probe"] = nq_probe
    spec.loader.exec_module(nq_probe)
    ProbeModelBase = nq_probe.ProbeModelBase
    PairedDirUncDataset = nq_probe.PairedDirUncDataset
    collate_paired_batch = nq_probe.collate_paired_batch
except Exception as e:
    print(f"Cannot import requirements: {e}")
    sys.exit(1)


def find_predicate_index(input_ids, predicate, tokenizer):
    """Finds the token index where the verb predicate starts."""
    # Try different tokenizations of the predicate word to handle spaces/subwords
    ids_variants = [
        tokenizer.encode(" " + predicate, add_special_tokens=False),
        tokenizer.encode(predicate, add_special_tokens=False),
        tokenizer.encode(" " + predicate.lower(), add_special_tokens=False),
        tokenizer.encode(predicate.lower(), add_special_tokens=False)
    ]
    
    for target_ids in ids_variants:
        n_t = len(target_ids)
        if n_t == 0:
            continue
        # Search for sublist in input_ids
        for i in range(len(input_ids) - n_t + 1):
            if input_ids[i:i+n_t] == target_ids:
                return i
    return None


def run_sliding_probe(base_model, W, b, input_ids, attention_mask, layer_idx):
    """Runs the probe weights (W, b) on all token positions in hidden states."""
    with torch.no_grad():
        out = base_model.lm(input_ids=input_ids, attention_mask=attention_mask)
        hs = base_model.get_layer_hidden(out.hidden_states, layer_idx) # [B, S, D]
        
        # W shape: [len(DIRS), D], b shape: [len(DIRS)]
        # Broadcast probe calculation across all S token positions
        # H: [B, S, 1, D], W_exp: [1, 1, C, D]
        H = hs.unsqueeze(2)
        W_exp = W.unsqueeze(0).unsqueeze(0)
        logits = (H * W_exp).sum(dim=-1) + b # [B, S, len(DIRS)]
        probs = torch.sigmoid(logits)
        return probs # [B, S, len(DIRS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--dev_soft", type=str, default="data/processed/case_grammar/paired_dev_gemini_soft.jsonl")
    parser.add_argument("--dev_strong", type=str, default="data/processed/case_grammar/paired_dev_gemini_strong.jsonl")
    parser.add_argument("--runs_dir", type=str, default="runs/layer_sweep_gemini_nq_aligned")
    parser.add_argument("--out_dir", type=str, default="runs/token_series_analysis")
    parser.add_argument("--layers", type=str, default="0,4,12,16,24")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--min_rel", type=int, default=-5, help="Min relative token position to verb")
    parser.add_argument("--max_rel", type=int, default=15, help="Max relative token position to verb")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(l.strip()) for l in args.layers.split(",")]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model Gemma...")
    base_model = ProbeModelBase(args.model_name).to(device)
    base_model.eval()

    # Monkey-patch PairedDirUncDataset.__getitem__ to pass 'predicate' metadata through
    original_getitem = PairedDirUncDataset.__getitem__
    def patched_getitem(self, idx):
        res = original_getitem(self, idx)
        res["predicate"] = self.aligned_pairs[idx].get("predicate", "")
        return res
    PairedDirUncDataset.__getitem__ = patched_getitem

    # Load datasets
    datasets = {}
    for prefix, path in [("soft", args.dev_soft), ("strong", args.dev_strong)]:
        if not Path(path).exists():
            print(f"Skipping {prefix}: file {path} not found")
            continue
        rows = nq_probe.read_jsonl(Path(path))
        ds = PairedDirUncDataset(rows, tokenizer)
        datasets[prefix] = ds

    # Rel position window sizes
    window_range = list(range(args.min_rel, args.max_rel + 1))
    n_steps = len(window_range)

    # We will analyze: Agent (who -> index 0) and Time (when -> index 2)
    # Mapping role to index in DIRS
    role_to_idx = {role: i for i, role in enumerate(DIRS)}
    target_roles = ["who", "when"] # Agent = who, Time = when

    for omission in datasets.keys():
        print(f"\n==========================================")
        print(f"Analyzing Omission Type: {omission.upper()}")
        print(f"==========================================")
        
        ds = datasets[omission]
        # Dynamically restore predicate from ds.pairs to ds.aligned_pairs
        for i, item in enumerate(ds.pairs):
            ds.aligned_pairs[i]["predicate"] = item["predicate"]

        # Use a custom collation function to add predicate into the batch dictionary
        def custom_collate_fn(b):
            batch_dict = collate_paired_batch(tokenizer, b, 256)
            batch_dict["predicate"] = [x["predicate"] for x in b]
            return batch_dict

        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        for L in layers:
            print(f"\n--- Layer {L} ---")
            # Load trained probe weights W and b
            probe_path = Path(args.runs_dir) / f"{omission}_layer_{L}" / f"best_probe_layer{L}.pt"
            if not probe_path.exists():
                print(f"Probe checkpoint not found at {probe_path}, skipping layer {L}")
                continue
            
            checkpoint = torch.load(probe_path, map_location=device)
            # W shape: [len(DIRS), D], b shape: [len(DIRS)]
            W = checkpoint["W"]
            b = checkpoint["b"]

            # Dict to accumulate probabilities at relative positions
            # Structure: trajectories[role][condition][rel_pos] = list of float probabilities
            trajectories = {
                role: {
                    "filled": {r: [] for r in window_range},
                    "missing": {r: [] for r in window_range}
                } for role in target_roles
            }

            matched_count = 0
            total_count = 0

            for batch in tqdm(dl, desc=f"L{L} {omission}"):
                f_ids = batch["f_input_ids"].to(device)
                f_mask = batch["f_attention_mask"].to(device)
                m_ids = batch["m_input_ids"].to(device)
                m_mask = batch["m_attention_mask"].to(device)

                # Slide probe over all token positions for both filled and missing inputs
                # probs: [B, S, len(DIRS)]
                probs_f = run_sliding_probe(base_model, W, b, f_ids, f_mask, L).float().cpu().numpy()
                probs_m = run_sliding_probe(base_model, W, b, m_ids, m_mask, L).float().cpu().numpy()

                B = len(batch["y"])
                total_count += B

                for bi in range(B):
                    pred = batch["predicate"][bi]
                    case_role = batch["case_role"][bi].lower() # e.g. "agent" -> "who", "time" -> "when"
                    
                    # Convert metadata case role to natural query string
                    # DIRS: ["who", "what", "when", "where", "why", "how", "which"]
                    if case_role == "agent":
                        query_role = "who"
                    elif case_role == "time":
                        query_role = "when"
                    elif case_role == "location":
                        query_role = "where"
                    elif case_role == "theme":
                        query_role = "what"
                    else:
                        continue  # Skip roles we aren't plotting right now to save clutter

                    if query_role not in target_roles:
                        continue

                    # Find verb index in filled and missing texts
                    ids_f = f_ids[bi].tolist()
                    ids_m = m_ids[bi].tolist()
                    
                    t_verb_f = find_predicate_index(ids_f, pred, tokenizer)
                    t_verb_m = find_predicate_index(ids_m, pred, tokenizer)

                    # Only accumulate if predicate is found in both sequences
                    if t_verb_f is not None and t_verb_m is not None:
                        matched_count += 1
                        role_class_idx = role_to_idx[query_role]

                        # Accumulate Filled trajectory
                        for rel in window_range:
                            t_pos = t_verb_f + rel
                            if 0 <= t_pos < len(ids_f) and batch["f_attention_mask"][bi, t_pos] == 1:
                                val = probs_f[bi, t_pos, role_class_idx]
                                trajectories[query_role]["filled"][rel].append(val)

                        # Accumulate Missing trajectory
                        for rel in window_range:
                            t_pos = t_verb_m + rel
                            if 0 <= t_pos < len(ids_m) and batch["m_attention_mask"][bi, t_pos] == 1:
                                val = probs_m[bi, t_pos, role_class_idx]
                                trajectories[query_role]["missing"][rel].append(val)

            print(f"  Predicate matched in {matched_count}/{total_count} pairs ({matched_count/total_count*100:.1f}%)")

            # -----------------------------------------------------------------
            # Plot results for this Layer L and Omission Type
            # -----------------------------------------------------------------
            fig, axes = plt.subplots(1, len(target_roles), figsize=(12, 5), sharey=True)
            if len(target_roles) == 1:
                axes = [axes]

            for ax, role in zip(axes, target_roles):
                # Calculate mean trajectories
                rel_pos_list = []
                mean_f = []
                mean_m = []

                for r in window_range:
                    vals_f = trajectories[role]["filled"][r]
                    vals_m = trajectories[role]["missing"][r]
                    if len(vals_f) > 0 and len(vals_m) > 0:
                        rel_pos_list.append(r)
                        mean_f.append(np.mean(vals_f))
                        mean_m.append(np.mean(vals_m))

                if len(rel_pos_list) > 0:
                    role_label = "Agent (who)" if role == "who" else "Time (when)"
                    ax.plot(rel_pos_list, mean_f, marker='o', color='blue', label='Filled (Baseline)', alpha=0.8)
                    ax.plot(rel_pos_list, mean_m, marker='s', color='red', label='Missing (Uncertain)', alpha=0.8)
                    
                    # Highlight the Verb position at 0
                    ax.axvline(0, color='gray', linestyle='--', label='Verb position', alpha=0.7)
                    
                    ax.set_title(f"Uncertainty: {role_label}")
                    ax.set_xlabel("Relative Token Position to Verb")
                    ax.set_ylabel("Detection Probability")
                    ax.grid(True, linestyle=":", alpha=0.5)
                    ax.set_xticks(window_range)
                    ax.legend()
                    ax.set_ylim(-0.05, 1.05)

            plt.suptitle(f"Layer {L} - Real-time Uncertainty Trajectory ({omission.capitalize()} Omission)")
            plt.tight_layout()
            
            fig_path = out_dir / f"{omission}_layer{L}_trajectory.png"
            plt.savefig(fig_path, dpi=200)
            print(f"  Saved plot to {fig_path}")
            plt.close()


if __name__ == "__main__":
    main()
