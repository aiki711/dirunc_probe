#!/usr/bin/env python3
"""
Extract and sweep ALL layers (0 to 26) of Gemma-2-2b-it on the binary-balanced test split.
Runs Gemma inference on the fly on a representative training subset and the dev set,
extracting activations for all 27 layers to compute and plot the F1 score transition.
"""
import json, random, os, sys, warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import importlib.util

warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

CACHE_DIR = Path("data/cache")
OUT_DIR   = Path("runs/identify_verify_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
ALL_SLOTS = ["who", "what", "when", "where", "how"]
ALL_CLASSES = ["who", "what", "when", "where", "how", "None"]
ROLE_TO_DIR = {
    "Agent":"who","Theme":"what","Location":"where",
    "Source":"where","Goal":"where","Time":"when","Manner":"how",
}
DIRS = ["who", "what", "when", "where", "why", "how", "which"]
NATURAL_QUERY_MAP = {
    "who": "who did it",
    "what": "what was done",
    "when": "when did it happen",
    "where": "where did it happen",
    "how": "how was it done",
    "why": "why did it happen",
    "which": "which one was it",
}

class DummyZeroClassifier:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0
        return out

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_s32():
    spec = importlib.util.spec_from_file_location("s32",
        "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load data splits ───────────────────────────────────────────────
    print("Loading data splits...")
    dev_rows   = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    train_rows = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    s32 = load_s32()
    dev_pairs   = s32.PairedDirUncDataset(dev_rows).pairs
    train_pairs = s32.PairedDirUncDataset(train_rows).pairs

    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")

    # Sample a representative subset of train pairs for speed (2000 pairs = 4000 sentences)
    random.seed(42)
    train_subset_pairs = random.sample(train_pairs, min(len(train_pairs), 2000))
    print(f"Using {len(train_subset_pairs)} training pairs for fast on-the-fly probing.")

    # Reconstruct the exact same binary-balanced test items (seed 42)
    test_slot_missing = {s: [] for s in ALL_SLOTS}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    neg_sampled_idx = random.sample(test_filled_idx, 150)
    neg_items = [(idx, "filled", "None") for idx in neg_sampled_idx]

    who_pool = test_slot_missing["who"]
    who_sampled = random.sample(who_pool, min(len(who_pool), 30))
    pos_items = [(idx, "missing", "who") for idx in who_sampled]
    
    rem_to_sample = 150 - len(pos_items)
    slots_left = ["what", "when", "where", "how"]
    per_slot = rem_to_sample // len(slots_left)
    for s in slots_left:
        pool = test_slot_missing[s]
        sampled = random.sample(pool, per_slot)
        pos_items.extend([(idx, "missing", s) for idx in sampled])
    
    if len(pos_items) < 150:
        needed = 150 - len(pos_items)
        extra = random.sample(test_slot_missing["when"], needed)
        pos_items.extend([(idx, "missing", "when") for idx in extra])

    eval_items = neg_items + pos_items
    random.shuffle(eval_items)

    eval_y_verify = [1 if cond == "missing" else 0 for _, cond, _ in eval_items]
    eval_y_verify = np.array(eval_y_verify)

    # ── 2. Load Gemma model and tokenizer ─────────────────────────────────
    print("\nLoading google/gemma-2-2b-it...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b-it", torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Reconstruct query token tokenization
    query_token_seqs = {}
    for d, qstr in NATURAL_QUERY_MAP.items():
        query_token_seqs[d] = tokenizer.encode(" " + qstr, add_special_tokens=False)

    # ── 3. Helper to extract activations for ALL layers ───────────────────
    # We write a batch processor to extract query token hidden states for all layers (0-26)
    def extract_activations(pairs_list):
        """Extracts [N, 27, 7, D] hidden states for both filled and missing versions."""
        f_hs_all = []
        m_hs_all = []
        
        batch_size = 16
        for start_idx in tqdm(range(0, len(pairs_list), batch_size), desc="Extracting activations"):
            batch = pairs_list[start_idx : start_idx + batch_size]
            B = len(batch)
            
            # Format inputs
            f_texts = [p.get("filled_text", p.get("text", "")) for p in batch]
            m_texts = [p.get("missing_text", p.get("text", "")) for p in batch]
            
            f_enc = tokenizer(f_texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
            m_enc = tokenizer(m_texts, truncation=True, padding='max_length', max_length=128, return_tensors='pt').to(device)
            
            # Calculate alignment positions for query tokens
            # We align at the end of the query phrase
            # For soft aligned, it aligns with the last token of the stripped query
            def get_positions(enc_input_ids):
                positions = []
                for bi in range(B):
                    ids = enc_input_ids[bi].tolist()
                    row_pos = []
                    for slot in DIRS:
                        # Find position of query token phrase
                        q_seq = query_token_seqs[slot]
                        pos = -1
                        # Search backward
                        for k in range(len(ids) - len(q_seq), -1, -1):
                            if ids[k : k + len(q_seq)] == q_seq:
                                pos = k + len(q_seq) - 1
                                break
                        if pos == -1:
                            # fallback to last non-padding token
                            try:
                                pos = ids.index(tokenizer.pad_token_id) - 1
                            except ValueError:
                                pos = len(ids) - 1
                        row_pos.append(pos)
                    positions.append(row_pos)
                return torch.tensor(positions, device=device)

            f_pos = get_positions(f_enc["input_ids"])
            m_pos = get_positions(m_enc["input_ids"])
            
            # Forward pass
            input_ids = torch.cat([f_enc["input_ids"], m_enc["input_ids"]], dim=0)
            attention_mask = torch.cat([f_enc["attention_mask"], m_enc["attention_mask"]], dim=0)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # out.hidden_states is a tuple of 27 tensors, each [2*B, SeqLen, D]
                # We extract for all 27 layers
                batch_f_layers = []
                batch_m_layers = []
                batch_indices = torch.arange(B, device=device).unsqueeze(1)
                
                for L in range(27):
                    hs = out.hidden_states[L]
                    hs_f = hs[:B]
                    hs_m = hs[B:]
                    
                    q_hs_f = hs_f[batch_indices, f_pos].cpu().float().numpy() # [B, 7, D]
                    q_hs_m = hs_m[batch_indices, m_pos].cpu().float().numpy() # [B, 7, D]
                    
                    batch_f_layers.append(q_hs_f)
                    batch_m_layers.append(q_hs_m)
                
                # Stack layers: batch_f_layers is 27 list items of [B, 7, D]
                # We stack to [B, 27, 7, D]
                f_hs_all.append(np.stack(batch_f_layers, axis=1))
                m_hs_all.append(np.stack(batch_m_layers, axis=1))
                
        return np.concatenate(f_hs_all, axis=0), np.concatenate(m_hs_all, axis=0)

    print("\n--- Extracting hidden states for Training Subset ---")
    train_f_hs, train_m_hs = extract_activations(train_subset_pairs)
    print(f"  Train hidden states shape: {train_f_hs.shape}") # [N_train, 27, 7, D]

    print("\n--- Extracting hidden states for Dev Set ---")
    dev_f_hs, dev_m_hs = extract_activations(dev_pairs)
    print(f"  Dev hidden states shape: {dev_f_hs.shape}") # [N_dev, 27, 7, D]

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Reconstruct labels for training subset
    N_train = train_f_hs.shape[0]
    train_y = np.zeros((N_train, 7))
    for i, pair in enumerate(train_subset_pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            d = DIRS.index(ROLE_TO_DIR[role])
            train_y[i, d] = 1.0

    # ── 4. Sweep all 27 layers ────────────────────────────────────────────
    sweep_results = []
    print("\n--- Sweeping all 27 layers ---")
    for layer in range(27):
        # ── A. Multinomial Probing Classifier ──
        X_multi_list = []
        y_multi_list = []
        for i in range(N_train):
            pair = train_subset_pairs[i]
            role = pair.get("case_role", "")
            if role in ROLE_TO_DIR:
                slot = ROLE_TO_DIR[role]
                X_multi_list.append(train_m_hs[i, layer].mean(axis=0))
                y_multi_list.append(slot)
                X_multi_list.append(train_f_hs[i, layer].mean(axis=0))
                y_multi_list.append("None")
        
        X_multi = np.array(X_multi_list)
        y_multi = np.array(y_multi_list)

        pca = PCA(n_components=128, random_state=42) # 128 components is faster & enough
        X_pca = pca.fit_transform(X_multi)
        
        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X_pca, y_multi)
        
        multi_clf = LogisticRegression(max_iter=300, C=1.0, random_state=42, class_weight='balanced')
        multi_clf.fit(X_lda, y_multi)

        pred_multi = []
        for idx, cond, _ in eval_items:
            hs = dev_m_hs[idx, layer] if cond == "missing" else dev_f_hs[idx, layer]
            hs_avg = hs.mean(axis=0, keepdims=True)
            hs_pca = pca.transform(hs_avg)
            hs_lda = lda.transform(hs_pca)
            pred_class = multi_clf.predict(hs_lda)[0]
            pred_multi.append(0 if pred_class == "None" else 1)

        multi_acc = accuracy_score(eval_y_verify, pred_multi)
        multi_f1  = f1_score(eval_y_verify, pred_multi)

        # ── B. Binary OR Probing Classifier ──
        probes = []
        for d in range(7):
            X = np.concatenate([train_f_hs[:, layer, d, :], train_m_hs[:, layer, d, :]], axis=0)
            y = np.concatenate([np.zeros(N_train), train_y[:, d]], axis=0)
            if len(np.unique(y)) <= 1:
                clf = DummyZeroClassifier()
            else:
                clf = LogisticRegression(solver='liblinear', tol=1e-2, max_iter=200, C=1.0, random_state=42, class_weight='balanced')
                clf.fit(X, y)
            probes.append(clf)

        # Calibrate thresholds on CAL split
        cal_slot_missing, cal_filled_idx = {s: [] for s in DIRS}, []
        for i in cal_indices:
            if i >= len(dev_pairs): continue
            pair = dev_pairs[i]
            role = pair.get("case_role", "")
            if role in ROLE_TO_DIR:
                cal_slot_missing[ROLE_TO_DIR[role]].append(i)
                cal_filled_idx.append(i)

        thresholds = []
        for d, slot in enumerate(DIRS):
            pos_idxs = cal_slot_missing.get(slot, [])
            if not pos_idxs:
                thresholds.append(0.5)
                continue
            n_cal = min(30, len(pos_idxs)) # 30 is enough for CAL speed
            cal_pos = random.sample(pos_idxs, n_cal)
            cal_neg = random.sample(cal_filled_idx, n_cal)
            X_cal = np.concatenate([dev_m_hs[cal_pos, layer, d, :], dev_f_hs[cal_neg, layer, d, :]], axis=0)
            y_cal = np.array([1]*n_cal + [0]*n_cal)
            probs_cal = probes[d].predict_proba(X_cal)[:, 1]
            best_t, best_f1 = 0.5, -1.0
            for t in np.linspace(0.01, 0.99, 50):
                f = f1_score(y_cal, (probs_cal >= t).astype(int), zero_division=0)
                if f > best_f1:
                    best_f1, best_t = f, t
            thresholds.append(best_t)

        pred_or = []
        for idx, cond, _ in eval_items:
            hs = dev_m_hs[idx, layer] if cond == "missing" else dev_f_hs[idx, layer]
            any_fire = False
            for d, slot in enumerate(DIRS):
                if slot in ALL_SLOTS:
                    p = probes[d].predict_proba(hs[d:d+1, :])[0, 1]
                    if p >= thresholds[d]:
                        any_fire = True
                        break
            pred_or.append(1 if any_fire else 0)

        or_acc = accuracy_score(eval_y_verify, pred_or)
        or_f1  = f1_score(eval_y_verify, pred_or)

        print(f"  Layer {layer:2d} -> Multinomial F1: {multi_f1*100:5.2f}% | Binary OR F1: {or_f1*100:5.2f}%")
        sweep_results.append({
            "layer": layer,
            "multinomial_acc": float(multi_acc),
            "multinomial_f1": float(multi_f1),
            "or_acc": float(or_acc),
            "or_f1": float(or_f1)
        })

    # Save to JSON
    out_json = OUT_DIR / "all_layers_balanced_sweep_results.json"
    out_json.write_text(json.dumps(sweep_results, indent=2))
    print(f"\nSaved results to {out_json}")

    # ── Plotting ──────────────────────────────────────────────────────────
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.figure(figsize=(10, 5.8), dpi=150)
    
    layers_plot = [r["layer"] for r in sweep_results]
    m_f1s       = [r["multinomial_f1"] * 100 for r in sweep_results]
    m_accs      = [r["multinomial_acc"] * 100 for r in sweep_results]
    or_f1s      = [r["or_f1"] * 100 for r in sweep_results]

    plt.plot(layers_plot, m_f1s, marker='o', markersize=4, color='#00897B', linewidth=2.5, 
             label="Multinomial Probing F1 (Omission)")
    plt.plot(layers_plot, m_accs, marker='s', markersize=4, color='#00796B', linewidth=1.8, linestyle=":",
             label="Multinomial Probing Accuracy")
    plt.plot(layers_plot, or_f1s, marker='x', markersize=4, color='#E53935', linewidth=2.0, linestyle="--",
             label="Binary OR Probing F1 (Omission)")

    plt.xlabel("Gemma-2-2b-it Layer Index (0 to 26)", fontsize=11, fontweight='bold')
    plt.ylabel("Score (%)", fontsize=11, fontweight='bold')
    plt.title("Probing Performance across Transformer Layers (All 27 Layers)\n(Binary Balanced Test Split)", 
              fontsize=12, fontweight='bold', pad=12)
    
    plt.xticks(np.arange(0, 27, 2))
    plt.ylim(40, 100)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plot_path = OUT_DIR / "layer_sweep_all_layers_balanced.png"
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
