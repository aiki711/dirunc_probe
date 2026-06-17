#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path
import numpy as np

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

# Dynamically patch dataset again for the analysis
import importlib.util
spec = importlib.util.spec_from_file_location("analysis_script", "scripts/44_token_series_analysis.py")
analysis_script = importlib.util.module_from_spec(spec)
sys.modules["analysis_script"] = analysis_script
spec.loader.exec_module(analysis_script)

from scripts.common import DIRS
nq_probe = analysis_script.nq_probe
ProbeModelBase = nq_probe.ProbeModelBase
PairedDirUncDataset = nq_probe.PairedDirUncDataset
collate_paired_batch = nq_probe.collate_paired_batch
find_predicate_index = analysis_script.find_predicate_index
run_sliding_probe = analysis_script.run_sliding_probe

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = analysis_script.AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model Gemma...")
    base_model = ProbeModelBase("google/gemma-2-2b-it").to(device)
    base_model.eval()

    # Monkey-patch dataset
    original_getitem = PairedDirUncDataset.__getitem__
    def patched_getitem(self, idx):
        res = original_getitem(self, idx)
        res["predicate"] = self.aligned_pairs[idx].get("predicate", "")
        return res
    PairedDirUncDataset.__getitem__ = patched_getitem

    datasets = {}
    for prefix, path in [("soft", "data/processed/case_grammar/paired_dev_gemini_soft.jsonl"), 
                         ("strong", "data/processed/case_grammar/paired_dev_gemini_strong.jsonl")]:
        if Path(path).exists():
            rows = nq_probe.read_jsonl(Path(path))
            ds = PairedDirUncDataset(rows, tokenizer)
            for i, item in enumerate(ds.pairs):
                ds.aligned_pairs[i]["predicate"] = item["predicate"]
            datasets[prefix] = ds

    target_roles = ["who", "when"]
    role_to_idx = {role: i for i, role in enumerate(DIRS)}
    layers = [0, 4, 12, 24]
    
    # Selected relative positions to query
    test_positions = [-2, 0, 1, 3, 5, 8]

    for omission in datasets.keys():
        print(f"\n==========================================")
        print(f"OMISSION TYPE: {omission.upper()}")
        print(f"==========================================")
        ds = datasets[omission]
        def custom_collate_fn(b):
            batch_dict = collate_paired_batch(tokenizer, b, 256)
            batch_dict["predicate"] = [x["predicate"] for x in b]
            return batch_dict
        dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

        for L in layers:
            probe_path = Path("runs/layer_sweep_gemini_nq_aligned") / f"{omission}_layer_{L}" / f"best_probe_layer{L}.pt"
            if not probe_path.exists():
                continue
            checkpoint = torch.load(probe_path, map_location=device)
            W = checkpoint["W"]
            b = checkpoint["b"]

            # Store trajectories to calculate mean values
            trajectories = {
                role: {
                    "filled": {r: [] for r in test_positions},
                    "missing": {r: [] for r in test_positions}
                } for role in target_roles
            }

            for batch in dl:
                f_ids = batch["f_input_ids"].to(device)
                f_mask = batch["f_attention_mask"].to(device)
                m_ids = batch["m_input_ids"].to(device)
                m_mask = batch["m_attention_mask"].to(device)

                probs_f = run_sliding_probe(base_model, W, b, f_ids, f_mask, L).float().cpu().numpy()
                probs_m = run_sliding_probe(base_model, W, b, m_ids, m_mask, L).float().cpu().numpy()

                for bi in range(len(batch["y"])):
                    pred = batch["predicate"][bi]
                    case_role = batch["case_role"][bi].lower()
                    
                    if case_role == "agent":
                        query_role = "who"
                    elif case_role == "time":
                        query_role = "when"
                    else:
                        continue

                    ids_f = f_ids[bi].tolist()
                    ids_m = m_ids[bi].tolist()
                    
                    t_verb_f = find_predicate_index(ids_f, pred, tokenizer)
                    t_verb_m = find_predicate_index(ids_m, pred, tokenizer)

                    if t_verb_f is not None and t_verb_m is not None:
                        role_class_idx = role_to_idx[query_role]
                        for r in test_positions:
                            # Filled
                            t_pos_f = t_verb_f + r
                            if 0 <= t_pos_f < len(ids_f) and batch["f_attention_mask"][bi, t_pos_f] == 1:
                                trajectories[query_role]["filled"][r].append(probs_f[bi, t_pos_f, role_class_idx])
                            # Missing
                            t_pos_m = t_verb_m + r
                            if 0 <= t_pos_m < len(ids_m) and batch["m_attention_mask"][bi, t_pos_m] == 1:
                                trajectories[query_role]["missing"][r].append(probs_m[bi, t_pos_m, role_class_idx])

            print(f"\n--- Layer {L} ---")
            for role in target_roles:
                role_name = "Agent (who)" if role == "who" else "Time (when)"
                print(f"  Role: {role_name}")
                print(f"    Position | Filled Prob | Missing Prob | Diff (M - F)")
                print(f"    ---------|-------------|--------------|-------------")
                for r in test_positions:
                    vals_f = trajectories[role]["filled"][r]
                    vals_m = trajectories[role]["missing"][r]
                    mean_f = np.mean(vals_f) if len(vals_f) > 0 else 0.0
                    mean_m = np.mean(vals_m) if len(vals_m) > 0 else 0.0
                    print(f"    {r:+8d} | {mean_f:11.4f} | {mean_m:12.4f} | {mean_m - mean_f:+11.4f}")

if __name__ == "__main__":
    main()
