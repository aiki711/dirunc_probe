
import sys
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add scripts dir to path to import from 03_train_probe
sys.path.append(str(Path(__file__).parent))

# Dynamic import for module with digits in name
import importlib.util
spec = importlib.util.spec_from_file_location("train_probe_03", Path(__file__).parent / "03_train_probe.py")
mod_03 = importlib.util.module_from_spec(spec)
sys.modules["train_probe_03"] = mod_03  # Required for dataclasses/pickle to work
spec.loader.exec_module(mod_03)

ProbeModelBase = mod_03.ProbeModelBase
QueryHead = mod_03.QueryHead
BaselineHead = mod_03.BaselineHead
MultiLayerQueryHead = getattr(mod_03, "MultiLayerQueryHead", None) # if exists
SPECIAL_TOKENS = mod_03.SPECIAL_TOKENS
DIRS = mod_03.DIRS
extract_activations = mod_03.extract_activations
collate_batch = mod_03.collate_batch
AutoTokenizer = mod_03.AutoTokenizer

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def plot_cosine_sim(weight_matrix: torch.Tensor, out_path: Path):
    # weight: (NumLabels, Hidden)
    # Cosine sim: (NumLabels, NumLabels)
    # Normalize rows
    norms = weight_matrix.norm(p=2, dim=1, keepdim=True) + 1e-8
    normalized = weight_matrix / norms
    sim = torch.mm(normalized, normalized.t()).detach().float().cpu().numpy()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(sim, annot=True, fmt=".2f", xticklabels=DIRS, yticklabels=DIRS, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.title("Cosine Similarity of Probe Weights")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Analysis] Saved cosine heatmap to {out_path}")

def analyze_gap(
    model_name: str,
    layer_idx: any, # int or list of int
    probe_state: dict,
    mode: str,
    dev_rows: list,
    out_dir: Path,
    device: torch.device
):
    print(f"[Analysis] Analyzing Contrastive Gap for mode={mode} layer={layer_idx}...")
    
    # 1. Setup Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS}) # Ensure consistency
    
    base_model = ProbeModelBase(
        model_name,
        vocab_size=len(tokenizer), # Resize if needed
        train_token_ids=[], # No training
    ).to(device)
    base_model.lm.eval()
    
    # Setup Probe Head
    hidden_size = base_model.hidden_size
    dtype = base_model.lm.dtype
    
    if mode == "baseline":
        head = BaselineHead(hidden_size, len(DIRS), dtype).to(device)
    elif mode == "query":
        head = QueryHead(hidden_size, len(DIRS), dtype).to(device)
    elif mode == "multilayer":
        num_fusion_layers = len(layer_idx)
        head = MultiLayerQueryHead(hidden_size, num_fusion_layers, len(DIRS), dtype).to(device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    head.load_state_dict(probe_state)
    head.eval()
    
    # 2. Extract Activations
    # Filter rows to only those with pair_id
    pair_rows = [r for r in dev_rows if "pair_id" in r]
    if not pair_rows:
        print("[Analysis] No rows with 'pair_id' found. Skipping Gap analysis.")
        return

    # Use simple batching
    bsz = 16
    dataset = []
    for r in pair_rows:
        # Dummy y, not needed for prediction but structure requires it
        dataset.append({"text": r["text"], "y": torch.zeros(len(DIRS)), "meta": r})
        
    dl = DataLoader(dataset, batch_size=bsz, collate_fn=lambda b: collate_batch(tokenizer, b, 256), shuffle=False)
    
    # We use extract_activations logic but we need to map back to rows
    # Actually, simpler loop since we just need one layer
    
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dl, desc="Inference"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            out = base_model.lm(input_ids=input_ids, attention_mask=attn_mask)
            
            if mode == "multilayer":
                # layers list extraction
                layer_list = layer_idx
                layer_states = []
                for L in layer_list:
                    hs_l = base_model.get_layer_hidden(out.hidden_states, L)
                    # For query model, we need vectors for each direction
                    feats = []
                    for i in range(input_ids.size(0)):
                        valid_mask = attn_mask[i] > 0
                        valid_ids = input_ids[i][valid_mask].tolist()
                        valid_hs = hs_l[i][valid_mask]
                        found_vecs = []
                        for d in DIRS:
                            tid = token_id_map[d]
                            pos = -1
                            for j in range(len(valid_ids)-1, -1, -1):
                                if valid_ids[j] == tid:
                                    pos = j
                                    break
                            if pos != -1:
                                found_vecs.append(valid_hs[pos])
                            else:
                                found_vecs.append(valid_hs[-1])
                        feats.append(torch.stack(found_vecs))
                    layer_states.append(torch.stack(feats))
                # features: (B, num_layers, 6, H)
                features = torch.stack(layer_states, dim=1)
            else:
                # single layer extraction
                hs = base_model.get_layer_hidden(out.hidden_states, layer_idx)
                if mode == "baseline":
                    lengths = attn_mask.long().sum(dim=1) - 1
                    features = hs[torch.arange(input_ids.size(0)), lengths]
                else: # query
                    feats = []
                    for i in range(input_ids.size(0)):
                        valid_mask = attn_mask[i] > 0
                        valid_ids = input_ids[i][valid_mask].tolist()
                        valid_hs = hs[i][valid_mask]
                        found_vecs = []
                        for d in DIRS:
                            tid = token_id_map[d]
                            pos = -1
                            for j in range(len(valid_ids)-1, -1, -1):
                                if valid_ids[j] == tid:
                                    pos = j
                                    break
                            if pos != -1:
                                found_vecs.append(valid_hs[pos])
                            else:
                                found_vecs.append(valid_hs[-1])
                        feats.append(torch.stack(found_vecs))
                    features = torch.stack(feats) # (B, 6, H)

            logits = head(features)
            probs = torch.sigmoid(logits) # (B, 6)
            all_probs.append(probs.cpu())
            
    all_probs = torch.cat(all_probs, dim=0).numpy() # (N, 6)
    
    # 3. Calculate Gap
    # Group by pair_id
    pair_map = defaultdict(dict)
    for i, r in enumerate(pair_rows):
        pid = r["pair_id"]
        cond = r.get("condition", "unknown") # 'resolved' or 'unresolved'
        pair_map[pid][cond] = all_probs[i]
        pair_map[pid]["target_dir"] = r.get("target_dir")
    
    gaps = defaultdict(list) # dir -> list of gaps
    
    for pid, data in pair_map.items():
        if "resolved" in data and "unresolved" in data:
            p_res = data["resolved"]
            p_unr = data["unresolved"]
            # Gap = Unresolved - Resolved 
            # (We expect Unresolved to have HIGHER uncertainty/prob=1 for missing slot)
            # Wait, 1 = missing slot.
            # So Unresolved (missing slot) should have Prob ~ 1.
            # Resolved (present slot) should have Prob ~ 0.
            # So Gap = P_unr - P_res should be Positive (ideally close to 1).
            
            diff = p_unr - p_res
            
            # We care specifically about the TARGET direction for this pair
            tgt = data.get("target_dir")
            if tgt in DIRS:
                idx = DIRS.index(tgt)
                gaps[tgt].append(diff[idx])
                gaps["ALL"].append(diff[idx])
    
    # 4. Plot
    avg_gaps = {d: np.mean(vals) for d, vals in gaps.items()}
    print("[Analysis] Average Contrastive Gaps (P_unr - P_res):")
    for d, v in avg_gaps.items():
        print(f"  {d}: {v:.4f}")
        
    plt.figure(figsize=(10, 5))
    x_labels = sorted(avg_gaps.keys())
    y_vals = [avg_gaps[k] for k in x_labels]
    
    sns.barplot(x=x_labels, y=y_vals, palette="viridis")
    plt.title(f"Average Contrastive Gap (Layer {layer_idx}, Mode {mode})\nGap = P(Missing) - P(Present)")
    plt.ylabel("Gap Probability")
    plt.ylim(-0.1, 1.1)
    for i, v in enumerate(y_vals):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
    out_file = out_dir / f"gap_analysis_{mode}_layer{layer_idx}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[Analysis] Saved Gap plot to {out_file}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", type=str, required=True, help="Path to summary.json from training")
    ap.add_argument("--out_dir", type=str, required=True, help="Folder to save plots")
    ap.add_argument("--dev_jsonl", type=str, required=True, help="Dev data with pairs")
    ap.add_argument("--analyze_mode", type=str, default="best", choices=["best", "all"])
    args = ap.parse_args()
    
    summary_path = Path(args.summary_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with summary_path.open("r") as f:
        summary = json.load(f)
        
    model_name = summary.get("model_name", "google/gemma-2-2b-it")
    best_key = summary.get("best_overall_key")
    
    # Identify target configs
    targets = []
    if args.analyze_mode == "best" and best_key:
        targets.append(best_key)
    else:
        # Collect all
        for k in summary.keys():
            if k.startswith("baseline/layer_") or k.startswith("query/layer_"):
                targets.append(k)
    
    print(f"[Analysis] Targets: {targets}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Analysis] Using device: {device}")
    
    dev_rows = load_jsonl(Path(args.dev_jsonl))
    
    for key in targets:
        # key example: "query/layer_25" or "multilayer"
        if key == "multilayer":
            mode = "multilayer"
            layer_idx = summary.get("multilayer", {}).get("layers", [10, 15, 20, 25])
        elif "/" in key:
            mode, layer_str = key.split("/")
            layer_idx = int(layer_str.replace("layer_", ""))
        else:
            print(f"[Analysis] Skipping non-layer key: {key}")
            continue
        
        info = summary[key]
        best_path = info.get("best_path")
        if not best_path:
            best_path = info.get("best", {}).get("best_path")
            
        if not best_path:
            print(f"[Warning] No best_path found for {key}, skipping.")
            continue
            
        # Fix path absolute/relative
        # best_path in summary might be relative to the original project root
        # We try to resolve it relative to the summary.json directory first
        bp = summary_path.parent / Path(best_path).name 
        if not bp.exists():
            # Try deeper search in subdirs
            potential_paths = list(summary_path.parent.glob(f"**/{Path(best_path).name}"))
            if potential_paths:
                bp = potential_paths[0]
            else:
                bp = Path(best_path)
        
        if not bp.exists():
             print(f"[Error] Checkpoint not found: {best_path} (resolved to {bp})")
             continue
             
        print(f"Loading {bp}...")
        state = torch.load(bp, map_location=device)
        
        # 1. Weights / Cosine Sim
        if mode == "multilayer":
            # State for MultiLayerQueryHead: layer_weights.weights, W, b
            if "layer_weights.weights" in state:
                w_vec = torch.softmax(state["layer_weights.weights"], dim=0).detach().cpu().numpy()
                print(f"[Analysis] Layer Weights (normalized): {w_vec}")
            if "W" in state:
                plot_cosine_sim(state["W"], out_dir / f"cosine_sim_{mode}_fused.png")
        elif mode == "query":
            if "W" in state:
                plot_cosine_sim(state["W"], out_dir / f"cosine_sim_{mode}_layer{layer_idx}.png")
        elif mode == "baseline":
            if "linear.weight" in state:
                plot_cosine_sim(state["linear.weight"], out_dir / f"cosine_sim_{mode}_layer{layer_idx}.png")
                
        # 2. Contrastive Gap
        analyze_gap(model_name, layer_idx, state, mode, dev_rows, out_dir, device)

if __name__ == "__main__":
    main()
