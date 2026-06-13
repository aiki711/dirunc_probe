import os
import sys
import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
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

class ComparativeDataset(Dataset):
    def __init__(self, rows, tokenizer, mode="query", align=True):
        pairs = {}
        for r in rows:
            pid = r["id"].rsplit("::", 1)[0]
            if pid not in pairs:
                pairs[pid] = {}
            pairs[pid][r["condition"]] = r

        self.items = []
        for pid, p in pairs.items():
            if "filled" in p and "missing" in p:
                meta = p["missing"].get("metadata", {})
                self.items.append({
                    "filled_text":  p["filled"]["text"],
                    "missing_text": p["missing"]["text"],
                    "y_missing":    p["missing"]["labels"],
                    "base_id":      pid,
                    "case_role":        meta.get("case_role", ""),
                    "saturation_score": float(meta.get("saturation_score", -1.0)),
                    "is_saturated":     bool(meta.get("is_saturated", False)),
                    "dataset_name":    pid.split("::")[0] if "::" in pid else "unknown"
                })

        self.processed = []
        
        filled_bases = [strip_query_tokens(item["filled_text"]).strip() for item in self.items]
        missing_bases = [strip_query_tokens(item["missing_text"]).strip() for item in self.items]
        
        enc_f = tokenizer(filled_bases, add_special_tokens=False)
        enc_m = tokenizer(missing_bases, add_special_tokens=False)
        
        query_token_seqs = {}
        for d, qstr in NATURAL_QUERY_MAP.items():
            query_token_seqs[d] = tokenizer.encode(" " + qstr, add_special_tokens=False)

        # To log some alignment stats
        len_diffs_before = []
        len_diffs_after = []

        for i, item in enumerate(self.items):
            base_f = filled_bases[i]
            base_m = missing_bases[i]
            
            len_f = len(enc_f["input_ids"][i])
            len_m = len(enc_m["input_ids"][i])
            diff = len_f - len_m
            len_diffs_before.append(abs(diff))
            
            # Text is kept natural (no dummy periods added)
            aligned_f = base_f
            aligned_m = base_m

            if mode == "query":
                text_f = aligned_f + NATURAL_QUERY_STR
                text_m = aligned_m + NATURAL_QUERY_STR
            else:
                text_f = aligned_f
                text_m = aligned_m

            ids_f = tokenizer.encode(text_f, add_special_tokens=True)
            ids_m = tokenizer.encode(text_m, add_special_tokens=True)
            len_diffs_after.append(abs(len(ids_f) - len(ids_m)))
            
            if mode == "query":
                pos_f_list = []
                pos_m_list = []
                for d in DIRS:
                    seq = query_token_seqs[d]
                    pos_f = None
                    for j in range(len(ids_f) - len(seq), -1, -1):
                        if ids_f[j:j+len(seq)] == seq:
                            pos_f = j + len(seq) - 1
                            break
                    pos_f_list.append(pos_f if pos_f is not None else len(ids_f) - 1)
                    
                    pos_m = None
                    for j in range(len(ids_m) - len(seq), -1, -1):
                        if ids_m[j:j+len(seq)] == seq:
                            pos_m = j + len(seq) - 1
                            break
                    pos_m_list.append(pos_m if pos_m is not None else len(ids_m) - 1)
                
                # Query alignment positions
                min_q_pos_f = min(pos_f_list)
                min_q_pos_m = min(pos_m_list)
                
                Q_f = len(ids_f) - min_q_pos_f
                Q_m = len(ids_m) - min_q_pos_m
                
                C_f = min_q_pos_f
                C_m = min_q_pos_m
                C_max = max(C_f, C_m)
                
                if align:
                    f_pos_ids = list(range(C_f)) + list(range(C_max, C_max + Q_f))
                    m_pos_ids = list(range(C_m)) + list(range(C_max, C_max + Q_m))
                else:
                    f_pos_ids = list(range(len(ids_f)))
                    m_pos_ids = list(range(len(ids_m)))
            else:
                pos_f_list = [len(ids_f) - 1] * 7
                pos_m_list = [len(ids_m) - 1] * 7
                
                if align:
                    C_f = len(ids_f)
                    C_m = len(ids_m)
                    C_max = max(C_f, C_m)
                    f_pos_ids = list(range(C_f - 1)) + [C_max - 1]
                    m_pos_ids = list(range(C_m - 1)) + [C_max - 1]
                else:
                    f_pos_ids = list(range(len(ids_f)))
                    m_pos_ids = list(range(len(ids_m)))

            self.processed.append({
                "text_f": text_f,
                "text_m": text_m,
                "f_positions": pos_f_list,
                "m_positions": pos_m_list,
                "f_position_ids": f_pos_ids,
                "m_position_ids": m_pos_ids,
                "y": item["y_missing"],
                "case_role": item["case_role"],
                "saturation_score": item["saturation_score"],
                "is_saturated": item["is_saturated"],
                "dataset_name": item["dataset_name"]
            })
            
        print(f"Alignment verification: Mean absolute len diff before={sum(len_diffs_before)/len(len_diffs_before):.3f}, after={sum(len_diffs_after)/len(len_diffs_after):.3f}")
            
    def __len__(self):
        return len(self.processed)
        
    def __getitem__(self, idx):
        item = self.processed[idx]
        y_vec = torch.tensor(
            [float(item["y"][d]) for d in DIRS], dtype=torch.float32
        )
        return {
            "text_f":           item["text_f"],
            "text_m":           item["text_m"],
            "f_positions":      torch.tensor(item["f_positions"], dtype=torch.long),
            "m_positions":      torch.tensor(item["m_positions"], dtype=torch.long),
            "f_position_ids":   torch.tensor(item["f_position_ids"], dtype=torch.long),
            "m_position_ids":   torch.tensor(item["m_position_ids"], dtype=torch.long),
            "y":                y_vec,
            "case_role":        item["case_role"],
            "saturation_score": item["saturation_score"],
            "is_saturated":     item["is_saturated"],
            "dataset_name":     item["dataset_name"],
        }

def collate_comparative_batch(tokenizer, batch, max_length):
    text_f = [b["text_f"] for b in batch]
    text_m = [b["text_m"] for b in batch]
    y      = torch.stack([b["y"] for b in batch])

    B = len(text_f)
    enc = tokenizer(text_f + text_m, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")

    f_ids = enc["input_ids"][:B]
    f_mask = enc["attention_mask"][:B]
    m_ids = enc["input_ids"][B:]
    m_mask = enc["attention_mask"][B:]

    f_pos_ids = []
    for bi in range(B):
        p_ids = list(batch[bi]["f_position_ids"].numpy())
        p_ids = p_ids + [0] * (f_ids.size(1) - len(p_ids))
        f_pos_ids.append(p_ids)
    f_pos_ids_tensor = torch.tensor(f_pos_ids, dtype=torch.long)

    m_pos_ids = []
    for bi in range(B):
        p_ids = list(batch[bi]["m_position_ids"].numpy())
        p_ids = p_ids + [0] * (m_ids.size(1) - len(p_ids))
        m_pos_ids.append(p_ids)
    m_pos_ids_tensor = torch.tensor(m_pos_ids, dtype=torch.long)

    return {
        "f_input_ids":       f_ids,
        "f_attention_mask":  f_mask,
        "m_input_ids":       m_ids,
        "m_attention_mask":  m_mask,
        "f_position_ids":    f_pos_ids_tensor,
        "m_position_ids":    m_pos_ids_tensor,
        "f_positions":       torch.stack([b["f_positions"] for b in batch]),
        "m_positions":       torch.stack([b["m_positions"] for b in batch]),
        "y":                 y,
        "case_role":        [b["case_role"]        for b in batch],
        "saturation_score": [b["saturation_score"] for b in batch],
        "is_saturated":     [b["is_saturated"]     for b in batch],
        "dataset_name":     [b["dataset_name"]     for b in batch],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--dev_data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="data/cache")
    parser.add_argument("--layers", type=str, default="0,4,8,12,16,20,24,26")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--mode", type=str, required=True, choices=["query", "final_token"])
    parser.add_argument("--align", action="store_true")
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--split", type=str, default="both", choices=["train", "dev", "both"])
    args = parser.parse_args()

    layers = [int(l.strip()) for l in args.layers.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "right"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    base = ProbeModelBase(args.model_name).to(device)
    base.eval()

    collate_fn = lambda b: collate_comparative_batch(tokenizer, b, 256)
    train_dl, dev_dl = None, None
    train_ds, dev_ds = None, None

    if args.split in ["train", "both"]:
        train_rows = []
        for p in args.train_data.split(","):
            train_rows.extend(nq_probe.read_jsonl(Path(p.strip())))
        train_ds = ComparativeDataset(train_rows, tokenizer, mode=args.mode, align=args.align)
        train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.split in ["dev", "both"]:
        dev_rows = []
        for p in args.dev_data.split(","):
            dev_rows.extend(nq_probe.read_jsonl(Path(p.strip())))
        dev_ds = ComparativeDataset(dev_rows, tokenizer, mode=args.mode, align=args.align)
        dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def extract_and_save(dl, ds, split_name):
        print(f"Extracting hidden states for {split_name} (Mode: {args.mode}, Align: {args.align})...")
        
        ys = []
        meta_list = []
        
        layer_f_hs = {L: [] for L in layers}
        layer_m_hs = {L: [] for L in layers}

        for batch in tqdm(dl, desc=split_name):
            f_ids = batch["f_input_ids"].to(device)
            f_mask = batch["f_attention_mask"].to(device)
            f_pos_ids = batch["f_position_ids"].to(device)
            m_ids = batch["m_input_ids"].to(device)
            m_mask = batch["m_attention_mask"].to(device)
            m_pos_ids = batch["m_position_ids"].to(device)
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

            input_ids = torch.cat([f_ids, m_ids], dim=0)
            attention_mask = torch.cat([f_mask, m_mask], dim=0)
            position_ids = torch.cat([f_pos_ids, m_pos_ids], dim=0)

            with torch.no_grad():
                out = base.lm(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

                for L in layers:
                    hs = base.get_layer_hidden(out.hidden_states, L)
                    hs_f = hs[:B]
                    hs_m = hs[B:]
                    
                    q_hs_f = hs_f[batch_indices, f_pos] # [B, 7, D]
                    q_hs_m = hs_m[batch_indices, m_pos] # [B, 7, D]
                    
                    layer_f_hs[L].append(q_hs_f)
                    layer_m_hs[L].append(q_hs_m)

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

    if args.split in ["train", "both"]:
        extract_and_save(train_dl, train_ds, "train")
    if args.split in ["dev", "both"]:
        extract_and_save(dev_dl, dev_ds, "dev")

if __name__ == "__main__":
    main()
