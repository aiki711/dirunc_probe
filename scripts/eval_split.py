import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
import importlib.util
import sys

# Dynamic Import from 03_train_probe.py
spec = importlib.util.spec_from_file_location("train_probe", "scripts/03_train_probe.py")
train_probe = importlib.util.module_from_spec(spec)
sys.modules["train_probe"] = train_probe
spec.loader.exec_module(train_probe)

# Access definitions
ProbeModelBase = train_probe.ProbeModelBase
QueryHead = train_probe.QueryHead
read_jsonl = train_probe.read_jsonl
extract_activations = train_probe.extract_activations
collate_batch = train_probe.collate_batch
DIRS = train_probe.DIRS
QUERY_TOKENS = train_probe.QUERY_TOKENS
QUERY_LABEL_STR = train_probe.QUERY_LABEL_STR

def evaluate_split(model, dl, device):
    """
    Evaluate QueryHead model on cached features.
    Similar to evaluate_cached in 03, but returns raw preds for confusion matrix.
    """
    model.eval()
    all_y_true = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dl:
            x = batch[0].to(device)
            y = batch[1].to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            
            all_y_true.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    if not all_y_true:
        return np.array([]), np.array([])
        
    y_true = np.concatenate(all_y_true, axis=0)
    probs = np.concatenate(all_probs, axis=0)
    return y_true, probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained model (e.g. runs/mixed/layer_X)")
    parser.add_argument("--layer_idx", type=int, required=True, help="Layer index to use")
    parser.add_argument("--model_base", type=str, default="distilroberta-base", help="Base model name")
    parser.add_argument("--qasrl_test", type=str, default="data/processed/qasrl/dirunc/test.jsonl")
    parser.add_argument("--multiwoz_test", type=str, default="data/processed/multiwoz/dirunc/test.jsonl")
    parser.add_argument("--out_dir", type=str, default="results/eval_split")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load trained head
    model_path = Path(args.model_dir) / f"best_query_layer{args.layer_idx}.pt"
    print(f"Loading head from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"additional_special_tokens": train_probe.SPECIAL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
        
    base_tmp = AutoModel.from_pretrained(args.model_base)
    base_tmp.resize_token_embeddings(len(tokenizer))
    hidden_size = base_tmp.config.hidden_size
    del base_tmp

    head = QueryHead(hidden_size, len(DIRS), torch.float32).to(device)
    head.load_state_dict(torch.load(model_path, map_location=device))
    head.eval()

    print("Initializing base model for feature extraction...")
    base_model = ProbeModelBase(
        args.model_base,
        vocab_size=len(tokenizer),
        train_token_ids=[tokenizer.convert_tokens_to_ids(t) for t in train_probe.SPECIAL_TOKENS],
        pretrained_model=None
    ).to(device)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "QA-SRL": args.qasrl_test,
        "MultiWOZ": args.multiwoz_test
    }
    
    results = {}
    
    from torch.utils.data import DataLoader
    
    for name, path_str in datasets.items():
        print(f"\nProcessing {name} ({path_str})...")
        p = Path(path_str)
        if not p.exists():
            print(f"Skipping {name}, file not found.")
            continue
            
        rows = read_jsonl(p)
        ds_raw = train_probe.JsonlDirUncDataset(rows, train_probe.FilterSpec())
        dl_raw = DataLoader(ds_raw, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=lambda b: collate_batch(tokenizer, b, max_length=256))
        
        print(f"  Extracting activations (Layer {args.layer_idx})...")
        X_dict, Y, _ = extract_activations(dl_raw, base_model, [args.layer_idx], "query", device, tokenizer)
        X = X_dict[args.layer_idx]
        
        ds_cached = TensorDataset(X, Y)
        dl_cached = DataLoader(ds_cached, batch_size=args.batch_size, shuffle=False)
        
        y_true, probs = evaluate_split(head, dl_cached, device)
        y_pred = (probs > 0.1).astype(int)  # Use optimal threshold from training
        
        has_pos = y_true.sum(axis=1) > 0
        acc_dict = {}
        if has_pos.sum() > 0:
            # DEBUG: Print first 5 labels to verify correctness
            print(f"  Debug: First 5 True Labels:\n{y_true[:5]}")
            print(f"  Debug: First 5 Pred Probs:\n{probs[:5]}")
            
            f1s = []
            for i, d in enumerate(DIRS):
                tp = ((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum()
                fp = ((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum()
                fn = ((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum()
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                f1s.append(f1)
                acc_dict[d] = f1
                print(f"  {d}: F1={f1:.4f}")
            
            macro = np.mean(f1s)
            results[name] = {"macro_f1": macro, "per_label": acc_dict}
            
            # Confusion Matrix
            single_mask = y_true.sum(axis=1) == 1
            if single_mask.sum() > 0:
                y_true_s = y_true[single_mask].argmax(axis=1)
                y_pred_s = probs[single_mask].argmax(axis=1)
                
                cm = confusion_matrix(y_true_s, y_pred_s, labels=range(len(DIRS)))
                
                # Plot using Matplotlib directly
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)
                ax.set(xticks=np.arange(cm.shape[1]),
                       yticks=np.arange(cm.shape[0]),
                       xticklabels=DIRS, yticklabels=DIRS,
                       title=f'Confusion Matrix - {name} (Layer {args.layer_idx})',
                       ylabel='True Label',
                       xlabel='Predicted Label')

                plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                         rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                fmt = 'd'
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], fmt),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")
                fig.tight_layout()
                plt.savefig(out_path / f"cm_{name}_layer{args.layer_idx}.png")
                print(f"  Saved confusion matrix to {out_path / f'cm_{name}_layer{args.layer_idx}.png'}")

    with (out_path / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
