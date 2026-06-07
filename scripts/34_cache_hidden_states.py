import os
import sys
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

from scripts.common import DIRS, NATURAL_QUERY_MAP, NATURAL_QUERY_STR, strip_query_tokens
import importlib.util

# Load dataset and model base dynamically
spec = importlib.util.spec_from_file_location("nq_probe", "scripts/32_train_contrastive_nq_probe.py")
nq_probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nq_probe)
ProbeModelBase = nq_probe.ProbeModelBase
PairedDirUncDataset = nq_probe.PairedDirUncDataset
collate_paired_batch = nq_probe.collate_paired_batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/cache")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,26")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--prefix", type=str, required=True, choices=["soft", "strong"])
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base = ProbeModelBase(args.model_name).to(device)
    base.eval()

    # Load datasets
    train_rows = []
    for p in args.train_data.split(","):
        train_rows.extend(nq_probe.read_jsonl(Path(p.strip())))
    dev_rows = []
    for p in args.dev_data.split(","):
        dev_rows.extend(nq_probe.read_jsonl(Path(p.strip())))

    train_ds = PairedDirUncDataset(train_rows, tokenizer)
    dev_ds = PairedDirUncDataset(dev_rows, tokenizer)

    collate_fn = lambda b: collate_paired_batch(tokenizer, b, 256)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    dev_dl = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # We also need query token sequences for positioning
    query_token_seqs = {}
    for d, qstr in NATURAL_QUERY_MAP.items():
        query_token_seqs[d] = tokenizer.encode(" " + qstr, add_special_tokens=False)

    def extract_and_save(dl, ds, split_name):
        print(f"Extracting hidden states for {split_name}...")
        
        # Initialize storage
        ys = []
        meta_list = []
        
        # We will save one file per layer to avoid memory overflow
        layer_f_hs = {L: [] for L in layers}
        layer_m_hs = {L: [] for L in layers}

        for batch in tqdm(dl, desc=split_name):
            f_ids = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            m_ids = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            y = batch["y"]
            
            ys.append(y)
            
            for bi in range(len(y)):
                meta_list.append({
                    "case_role": batch["case_role"][bi],
                    "saturation_score": batch["saturation_score"][bi],
                    "is_saturated": batch["is_saturated"][bi],
                    "dataset_name": batch["dataset_name"][bi],
                })

            B = f_ids.size(0)
            f_pos = batch["f_positions"].to(device)
            m_pos = batch["m_positions"].to(device)
            batch_indices = torch.arange(B, device=device).unsqueeze(1) # [B, 1]

            # Concatenate filled and missing inputs for a single forward pass
            input_ids = torch.cat([f_ids, m_ids], dim=0)
            attention_mask = torch.cat([f_mask, m_mask], dim=0)

            with torch.no_grad():
                out = base.lm(input_ids=input_ids, attention_mask=attention_mask)

                for L in layers:
                    hs = base.get_layer_hidden(out.hidden_states, L)
                    hs_f = hs[:B]
                    hs_m = hs[B:]
                    
                    # GPU advanced indexing
                    q_hs_f = hs_f[batch_indices, f_pos] # [B, 7, D]
                    q_hs_m = hs_m[batch_indices, m_pos] # [B, 7, D]
                    
                    # Store as lists of batch tensors on GPU
                    layer_f_hs[L].append(q_hs_f)
                    layer_m_hs[L].append(q_hs_m)

        # Concatenate and save per layer
        y_all = torch.cat(ys, dim=0)
        for L in layers:
            f_tensor = torch.cat(layer_f_hs[L], dim=0).cpu()
            m_tensor = torch.cat(layer_m_hs[L], dim=0).cpu()
            
            save_path = out_dir / f"{args.prefix}_layer{L}_{split_name}.pt"
            print(f"Saving {save_path}...")
            torch.save({
                "f_hs": f_tensor,
                "m_hs": m_tensor,
                "y": y_all,
                "metadata": meta_list
            }, save_path)

    extract_and_save(train_dl, train_ds, "train")
    extract_and_save(dev_dl, dev_ds, "dev")

if __name__ == "__main__":
    main()
