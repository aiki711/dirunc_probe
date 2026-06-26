#!/usr/bin/env python3
"""
Re-run probing evaluation with corrected labels (case_role-based).

For role identification: uses PCA(256) + LDA pipeline.
LDA maximizes between-class scatter / within-class scatter,
explicitly separating different slot type representations.

Prompting baselines are kept from existing result files.
"""
import os, sys, json, random, warnings
import torch, numpy as np
warnings.simplefilter('ignore')
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from scripts.common import DIRS
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Constants ──────────────────────────────────────────────────────────────
CASE_ROLES  = ["Agent","Theme","Location","Source","Goal","Time","Manner"]
ALL_CLASSES = ["who","what","when","where","how","None"]

ROLE_TO_DIR = {
    "Agent":    "who",
    "Theme":    "what",
    "Location": "where",
    "Source":   "where",
    "Goal":     "where",
    "Time":     "when",
    "Manner":   "how",
}

class DummyZeroClassifier:
    def fit(self, X, y): pass
    def predict(self, X): return np.zeros(X.shape[0], dtype=int)
    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2), dtype=np.float32)
        out[:, 0] = 1.0
        return out

def load_s32():
    spec = importlib.util.spec_from_file_location(
        "s32", "scripts/32_train_contrastive_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def build_caserole_y(pairs):
    """[N, 7] binary matrix from case_role."""
    y = np.zeros((len(pairs), 7), dtype=np.float32)
    for i, pair in enumerate(pairs):
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            d = DIRS.index(ROLE_TO_DIR[role])
            y[i, d] = 1.0
    return y

def calibrate_threshold_per_slot(y_true_1d, probs_1d):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y_true_1d, (probs_1d >= t).astype(int), pos_label=1, zero_division=0)
        if f > best_f:
            best_f, best_t = f, t
    return best_t

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    CACHE_DIR  = Path("data/cache")
    PREFIX     = "final_token_aligned_soft"
    LAYER      = 26
    EVAL_SIZE  = 300
    OUT_DIR    = Path("runs/identify_verify_comparison")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s32 = load_s32()
    PairedDirUncDataset = s32.PairedDirUncDataset

    # ── 1. Load caches ────────────────────────────────────────────────────
    print(f"Loading layer {LAYER} caches...")
    train_cache = torch.load(CACHE_DIR / f"{PREFIX}_layer{LAYER}_train.pt", map_location="cpu")
    dev_cache   = torch.load(CACHE_DIR / f"{PREFIX}_layer{LAYER}_dev.pt",   map_location="cpu")

    train_f_hs = train_cache["f_hs"].float().numpy()
    train_m_hs = train_cache["m_hs"].float().numpy()
    dev_f_hs   = dev_cache["f_hs"].float().numpy()
    dev_m_hs   = dev_cache["m_hs"].float().numpy()
    N_train    = train_f_hs.shape[0]

    # ── 2. Build datasets ─────────────────────────────────────────────────
    print("Building train pairs...")
    train_rows  = read_jsonl("data/processed/case_grammar/natural_train.jsonl")
    train_ds    = PairedDirUncDataset(train_rows)
    train_pairs = train_ds.pairs[:N_train]

    print("Building dev pairs...")
    dev_rows  = read_jsonl("data/processed/case_grammar/natural_dev.jsonl")
    dev_ds    = PairedDirUncDataset(dev_rows)
    dev_pairs = dev_ds.pairs

    if len(dev_pairs) != dev_cache["f_hs"].shape[0]:
        dev_pairs = dev_pairs[:dev_cache["f_hs"].shape[0]]

    # ── 2b. Load cal/test split indices ───────────────────────────────────
    cal_indices  = np.load(CACHE_DIR / "dev_cal_indices.npy")
    test_indices = np.load(CACHE_DIR / "dev_test_indices.npy")
    print(f"  Cal:  {len(cal_indices)} pairs | Test: {len(test_indices)} pairs")

    # ── 3. case_role-based training labels ────────────────────────────────
    print("Building case_role-based training labels...")
    train_y_cr = build_caserole_y(train_pairs)

    # ── 4. Train 7 binary probes (sufficiency detection) ──────────────────
    print("Training 7 binary probes with class_weight='balanced'...")
    probes = []
    for d in range(7):
        X = np.concatenate([train_f_hs[:, d, :], train_m_hs[:, d, :]], axis=0)
        y = np.concatenate([np.zeros(N_train), train_y_cr[:, d]], axis=0)
        if len(np.unique(y)) <= 1:
            clf = DummyZeroClassifier()
        else:
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                                     class_weight='balanced')
        clf.fit(X, y)
        probes.append(clf)
        print(f"  [{DIRS[d]:6s}] pos={int(train_y_cr[:, d].sum())}")

    # ── 4b. Train 6-class multinomial probe for role identification ────────
    # Uses mean of 7-slot hidden states; trained as direct 6-class problem
    print("Training 6-class multinomial probe (role identification)...")
    role_labels_train = [ROLE_TO_DIR.get(p.get('case_role',''), 'None')
                         for p in train_pairs]
    X_multi = np.concatenate([
        train_f_hs.mean(axis=1),   # [N_train, D] filled → 'None'
        train_m_hs.mean(axis=1),   # [N_train, D] missing → mapped role
    ], axis=0)
    y_multi = np.array(['None'] * N_train + role_labels_train)
    # PCA: D=2304 → 256 (retains >95% variance typically)
    print("  Step 1: PCA (256 components) for dimensionality reduction...")
    pca = PCA(n_components=256, random_state=42)
    X_multi_pca = pca.fit_transform(X_multi)

    # LDA: Maximises between-class/within-class scatter
    # Maps 256-D to at most (n_classes-1)=5 discriminant dimensions
    print("  Step 2: LDA — maximising inter-class / intra-class separation...")
    lda = LinearDiscriminantAnalysis()
    X_multi_lda = lda.fit_transform(X_multi_pca, y_multi)

    # Final classifier in LDA space
    print("  Step 3: LogisticRegression in LDA space...")
    multi_clf = LogisticRegression(max_iter=500, C=1.0, random_state=42,
                                   class_weight='balanced')
    multi_clf.fit(X_multi_lda, y_multi)
    print(f"  Classes: {list(multi_clf.classes_)}")

    # ── 5. Calibrate thresholds on CAL split only ─────────────────────────
    print("Calibrating per-slot thresholds on CAL split (not test)...")
    cal_slot_missing = {s: [] for s in DIRS}
    cal_filled_idx   = []
    for i in cal_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            cal_slot_missing[ROLE_TO_DIR[role]].append(i)
            cal_filled_idx.append(i)

    thresholds = []
    random.seed(0)
    for d, slot in enumerate(DIRS):
        pos = cal_slot_missing.get(slot, [])
        if not pos:
            thresholds.append(0.5)
            continue
        n_cal = min(50, len(pos))
        cal_pos = random.sample(pos, n_cal)
        cal_neg = random.sample(cal_filled_idx, n_cal)
        X_cal = np.concatenate([
            dev_m_hs[cal_pos, d, :],
            dev_f_hs[cal_neg, d, :]
        ], axis=0)
        y_cal = np.array([1]*n_cal + [0]*n_cal, dtype=np.float32)
        probs_cal = probes[d].predict_proba(X_cal)[:, 1]
        t = calibrate_threshold_per_slot(y_cal, probs_cal)
        thresholds.append(t)
        print(f"  [{slot:6s}] threshold={t:.3f}")

    # ── 6. Build balanced eval set from TEST split only ───────────────────
    print(f"Building balanced {EVAL_SIZE}-sample eval set (TEST split)...")
    test_slot_missing = {s: [] for s in DIRS}
    test_filled_idx   = []
    for i in test_indices:
        pair = dev_pairs[i]
        role = pair.get("case_role", "")
        if role in ROLE_TO_DIR:
            test_slot_missing[ROLE_TO_DIR[role]].append(i)
            test_filled_idx.append(i)

    class_groups = {c: [] for c in ALL_CLASSES}
    for slot, idxs in test_slot_missing.items():
        if slot in ALL_CLASSES:
            for i in idxs:
                class_groups[slot].append((i, "missing"))
    for i in test_filled_idx:
        class_groups["None"].append((i, "filled"))

    num_per_class = max(1, EVAL_SIZE // 6)
    random.seed(42)
    sampled_items = []
    for c in ALL_CLASSES:
        idxs = class_groups[c]
        sampled_items.extend(random.sample(idxs, min(len(idxs), num_per_class)))
    print(f"  Sampled {len(sampled_items)} items")

    # ── 7. Ground truth ───────────────────────────────────────────────────
    y_true_role = []
    y_true_suff_str = []
    for idx, cond in sampled_items:
        pair = dev_pairs[idx]
        if cond == "filled":
            y_true_role.append("None")
            y_true_suff_str.append("Sufficient")
        else:
            y_true_role.append(ROLE_TO_DIR[pair["case_role"]])
            y_true_suff_str.append("Insufficient")

    # ── 8. Probe predictions ──────────────────────────────────────────────
    # Role identification: multinomial classifier on mean hidden state
    # Sufficiency: any binary probe fires above threshold
    probe_pred_role = []
    probe_pred_suff = []
    for idx, cond in sampled_items:
        hs_7   = dev_f_hs[idx] if cond == 'filled' else dev_m_hs[idx]
        hs_avg = hs_7.mean(axis=0, keepdims=True)   # [1, D]
        hs_pca = pca.transform(hs_avg)               # [1, 256]
        hs_lda = lda.transform(hs_pca)               # [1, n_components]

        # ─ Role identification via LDA projection + classifier ─
        pred_role = multi_clf.predict(hs_lda)[0]     # 'who'/'what'/.../'None'
        probe_pred_role.append(pred_role)

        # ─ Sufficiency from multinomial prediction ─
        probe_pred_suff.append('Sufficient' if pred_role == 'None' else 'Insufficient')

    # ── 9. Compute metrics ─────────────────────────────────────────────────
    acc_probe_role = accuracy_score(y_true_role, probe_pred_role)
    f1_probe_role  = f1_score(y_true_role, probe_pred_role, average="macro", zero_division=0)
    acc_probe_suff = accuracy_score(y_true_suff_str, probe_pred_suff)
    f1_probe_suff  = f1_score(y_true_suff_str, probe_pred_suff,
                               pos_label="Insufficient", zero_division=0)

    print("\n====== Updated Probing Results ======")
    print(f"Identify Accuracy (6-class): {acc_probe_role*100:.2f}%")
    print(f"Identify F1 (Macro):         {f1_probe_role*100:.2f}%")
    print(f"Verify Accuracy:             {acc_probe_suff*100:.2f}%")
    print(f"Verify F1 (Omission):        {f1_probe_suff*100:.2f}%")

    # ── 10. Load existing prompting results ───────────────────────────────
    existing = json.loads((OUT_DIR / "comparison_report.json").read_text())
    prompting = existing["prompting"]
    multilabel = json.loads((OUT_DIR / "multilabel_prompting_results.json").read_text())
    onestep_suff_f1 = multilabel["onestep"]["verify_f1_omission"]
    onestep_suff_acc = multilabel["onestep"]["verify_accuracy"]
    twostep_suff_f1  = multilabel["twostep"]["verify_f1_omission"]
    twostep_suff_acc = multilabel["twostep"]["verify_accuracy"]
    twostep_id_f1    = multilabel["twostep"]["identify_f1_macro_after_verify"]

    # ── 11. Save updated report ───────────────────────────────────────────
    updated = {
        "eval_size": len(sampled_items),
        "layer": LAYER,
        "label_source": "case_role (corrected)",
        "probing": {
            "identify_accuracy":       float(acc_probe_role),
            "identify_f1_macro":       float(f1_probe_role),
            "sufficiency_accuracy":    float(acc_probe_suff),
            "sufficiency_f1_omission": float(f1_probe_suff),
        },
        "prompting": prompting,
    }
    (OUT_DIR / "comparison_report_v2.json").write_text(json.dumps(updated, indent=2))
    print(f"\nSaved updated report to {OUT_DIR / 'comparison_report_v2.json'}")

    # ── 12. Regenerate comprehensive_comparison.png ───────────────────────
    print("Regenerating comprehensive_comparison.png ...")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    fig.suptitle("Probing vs. Prompting Baselines (Gemma-2-2b-it, Layer 26)",
                 fontsize=14, fontweight='bold', y=1.02)

    # Colors
    c_probe    = '#00897B'   # Teal
    c_onestep  = '#FB8C00'   # Amber
    c_twostep  = '#E53935'   # Red

    # ── Left: Uncertainty Detection (Sufficiency / Verify) ────────────────
    methods_suff = ['Probing\n(Ours)', 'One-step\nPrompting', 'Identify-then\nVerify']
    acc_suff  = [acc_probe_suff,  onestep_suff_acc,  twostep_suff_acc]
    f1_suff   = [f1_probe_suff,   onestep_suff_f1,   twostep_suff_f1]
    x1 = np.arange(len(methods_suff))
    w  = 0.35
    b1 = ax1.bar(x1 - w/2, [v*100 for v in acc_suff], w,
                 color=[c_probe, c_onestep, c_twostep], alpha=0.85, label='Accuracy')
    b2 = ax1.bar(x1 + w/2, [v*100 for v in f1_suff],  w,
                 color=[c_probe, c_onestep, c_twostep], alpha=0.5,  label='F1 (Omission)',
                 edgecolor=[c_probe, c_onestep, c_twostep], linewidth=1.5, linestyle='--')
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                     f'{h:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_title('Uncertainty Detection\n(Sufficient vs. Insufficient)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=10)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(methods_suff, fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=8, loc='upper right')

    # ── Right: Per-slot Binary F1 (Case Role Completeness) ────────────────
    slot_json_path = OUT_DIR / "slot_metrics_test.json"
    if slot_json_path.exists():
        slot_data  = json.loads(slot_json_path.read_text())
        slot_names = slot_data["slots"]       # ["who","when","how","what","where"]
        slot_f1    = slot_data["f1"]          # F1 per slot (%)
    else:
        slot_names = ["who", "when", "how", "what", "where"]
        slot_f1    = [0.0] * 5
        print(f"  WARNING: {slot_json_path} not found; using zeros.")

    x2  = np.arange(len(slot_names))
    w2  = 0.55
    colors2 = [('#B0BEC5' if s == 'who' else c_probe) for s in slot_names]

    b3 = ax2.bar(x2, slot_f1, w2, color=colors2, alpha=0.85,
                 edgecolor='white', linewidth=0.8)
    for bar, f in zip(b3, slot_f1):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{f:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Chance-level reference line
    ax2.axhline(50.0, color='gray', linestyle='--', linewidth=1.2, label='Chance level (50%)')

    # Dagger for 'who'
    if 'who' in slot_names:
        wi = slot_names.index('who')
        ax2.text(wi, slot_f1[wi] + 6, '†', ha='center', fontsize=12, color='#B0BEC5')

    ax2.set_title('Case Role Completeness Detection\n(Binary per-slot: missing vs. filled)',
                  fontsize=11, fontweight='bold')
    ax2.set_ylabel('F1-Score (%)', fontsize=10)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([('who†' if s == 'who' else s) for s in slot_names], fontsize=10)
    ax2.set_ylim(0, 115)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.text(0.02, 0.03,
             '† who (Agent): dev の 1.9% のみ, 信頼性低\n'
             '  Cal: dev 50% (層別) | Test: dev 残り 50%',
             transform=ax2.transAxes, fontsize=6.5, color='gray', va='bottom')

    # Global legend for left panel colours
    patches = [
        mpatches.Patch(color=c_probe,   label='Probing (Ours)'),
        mpatches.Patch(color=c_onestep, label='One-step Prompting'),
        mpatches.Patch(color=c_twostep, label='Identify-then-Verify'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    out_png = OUT_DIR / "comprehensive_comparison.png"
    plt.savefig(out_png, bbox_inches='tight')
    print(f"Saved to {out_png}")


if __name__ == "__main__":
    main()
