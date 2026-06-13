import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path("runs/layer_sweep_gemini_nq_aligned")
OUT_DIR = Path("runs")
LAYERS = [0, 4, 8, 12, 16, 20, 24, 26]
OMISSIONS = ["soft", "strong"]

def load_best_epoch(omission: str, layer: int):
    log_file = RUNS_DIR / f"{omission}_layer_{layer}" / "log.jsonl"
    if not log_file.exists():
        return None
    
    epochs = []
    with log_file.open("r") as f:
        for line in f:
            if line.strip():
                epochs.append(json.loads(line))
                
    if not epochs:
        return None
        
    # We select the best epoch based on macro_f1 + pair_accuracy_standard (same as in bash script summary)
    best_idx = np.argmax([ep.get("macro_f1", 0.0) + ep.get("pair_accuracy_standard", 0.0) for ep in epochs])
    return epochs[best_idx]

def main():
    data = {om: {} for om in OMISSIONS}
    
    # Load all data
    for om in OMISSIONS:
        for layer in LAYERS:
            best_ep = load_best_epoch(om, layer)
            if best_ep:
                data[om][layer] = best_ep
                
    # ----------------------------------------------------
    # Analysis 1: Soft vs Strong Macro F1 Plot
    # ----------------------------------------------------
    available_layers = [l for l in LAYERS if l in data["soft"] and l in data["strong"]]
    soft_f1 = [data["soft"][l]["macro_f1"] for l in available_layers]
    strong_f1 = [data["strong"][l]["macro_f1"] for l in available_layers]
    
    plt.figure(figsize=(8, 5))
    plt.plot(available_layers, soft_f1, marker='o', linestyle='-', color='blue', label='Soft Omission (something)')
    plt.plot(available_layers, strong_f1, marker='s', linestyle='--', color='red', label='Strong Omission (deletion)')
    plt.title("Probing Macro F1: Soft vs Strong Omission across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Macro F1")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(available_layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hypothesis2_f1_comparison.png", dpi=200)
    plt.close()
    
    # Diff plot
    diff_f1 = [str_f - sft_f for str_f, sft_f in zip(strong_f1, soft_f1)]
    plt.figure(figsize=(8, 4))
    plt.bar(available_layers, diff_f1, color='purple', alpha=0.7, width=2.0)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.title("F1 Performance Gap (Strong - Soft) across Layers")
    plt.xlabel("Layer")
    plt.ylabel("Delta Macro F1 (Strong - Soft)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(available_layers)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hypothesis2_f1_gap.png", dpi=200)
    plt.close()
    
    # ----------------------------------------------------
    # Analysis 2: Case Role (who/where vs why/how etc.)
    # ----------------------------------------------------
    # Check roles
    sample_ep = data["soft"][available_layers[-1]]
    case_roles = sorted(sample_ep.get("by_case_role", {}).keys())
    
    # Let's plot case role accuracies across layers for Soft
    plt.figure(figsize=(10, 6))
    for role in case_roles:
        y_vals = []
        for l in available_layers:
            role_data = data["soft"][l].get("by_case_role", {}).get(role, {})
            # Use standard or strict pair accuracy, or macro_f1. Let's use macro_f1 or pair accuracy.
            # Standard pair accuracy or macro_f1 is fine. Let's plot standard pair accuracy.
            y_vals.append(role_data.get("pair_acc_standard", 0.0))
        plt.plot(available_layers, y_vals, marker='o', label=role)
    plt.title("Soft Omission: Standard Pair Accuracy by Case Role")
    plt.xlabel("Layer")
    plt.ylabel("Standard Pair Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(available_layers)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hypothesis1_case_roles_soft.png", dpi=200)
    plt.close()

    # Same for Strong
    plt.figure(figsize=(10, 6))
    for role in case_roles:
        y_vals = []
        for l in available_layers:
            role_data = data["strong"][l].get("by_case_role", {}).get(role, {})
            y_vals.append(role_data.get("pair_acc_standard", 0.0))
        plt.plot(available_layers, y_vals, marker='s', label=role)
    plt.title("Strong Omission: Standard Pair Accuracy by Case Role")
    plt.xlabel("Layer")
    plt.ylabel("Standard Pair Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(available_layers)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hypothesis1_case_roles_strong.png", dpi=200)
    plt.close()
    
    # ----------------------------------------------------
    # Analysis 3: Saturation bins (Argument necessity)
    # ----------------------------------------------------
    sample_sat = data["soft"][available_layers[-1]].get("by_saturation", {})
    sat_bins = sorted([k for k in sample_sat.keys() if k.startswith("sat_0") or k.startswith("sat_1")])
    
    plt.figure(figsize=(10, 6))
    for bin_key in sat_bins:
        y_vals = []
        for l in available_layers:
            sat_data = data["soft"][l].get("by_saturation", {}).get(bin_key, {})
            y_vals.append(sat_data.get("pair_acc_standard", 0.0))
        plt.plot(available_layers, y_vals, marker='o', label=bin_key)
    plt.title("Soft Omission: Standard Pair Accuracy by Saturation Score (Necessity)")
    plt.xlabel("Layer")
    plt.ylabel("Standard Pair Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(available_layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hypothesis1_saturation_soft.png", dpi=200)
    plt.close()

    # Same for Strong
    plt.figure(figsize=(10, 6))
    for bin_key in sat_bins:
        y_vals = []
        for l in available_layers:
            sat_data = data["strong"][l].get("by_saturation", {}).get(bin_key, {})
            y_vals.append(sat_data.get("pair_acc_standard", 0.0))
        plt.plot(available_layers, y_vals, marker='s', label=bin_key)
    plt.title("Strong Omission: Standard Pair Accuracy by Saturation Score (Necessity)")
    plt.xlabel("Layer")
    plt.ylabel("Standard Pair Accuracy")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(available_layers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hypothesis1_saturation_strong.png", dpi=200)
    plt.close()

    # Log text summary
    print("--- F1 Score Table ---")
    print("| Layer | Soft Macro F1 | Strong Macro F1 | Gap (Strong - Soft) |")
    print("|-------|---------------|-----------------|---------------------|")
    for l, sf, st, df in zip(available_layers, soft_f1, strong_f1, diff_f1):
        print(f"| {l:5d} | {sf:.4f}        | {st:.4f}          | {df:+.4f}             |")
        
    print("\n--- Case Role Accuracies Table (Soft Omission) ---")
    header = "| Layer | " + " | ".join(case_roles) + " |"
    print(header)
    print("|---| " + " | ".join(["---"] * len(case_roles)) + " |")
    for l in available_layers:
        row = f"| {l:5d} | "
        vals = []
        for role in case_roles:
            val = data["soft"][l].get("by_case_role", {}).get(role, {}).get("pair_acc_standard", 0.0)
            vals.append(f"{val:.4f}")
        row += " | ".join(vals) + " |"
        print(row)

    print("\n--- Case Role Accuracies Table (Strong Omission) ---")
    print(header)
    print("|---| " + " | ".join(["---"] * len(case_roles)) + " |")
    for l in available_layers:
        row = f"| {l:5d} | "
        vals = []
        for role in case_roles:
            val = data["strong"][l].get("by_case_role", {}).get(role, {}).get("pair_acc_standard", 0.0)
            vals.append(f"{val:.4f}")
        row += " | ".join(vals) + " |"
        print(row)

    print("\n--- Saturation Bins Table (Soft Omission) ---")
    header = "| Layer | " + " | ".join(sat_bins) + " |"
    print(header)
    print("|---| " + " | ".join(["---"] * len(sat_bins)) + " |")
    for l in available_layers:
        row = f"| {l:5d} | "
        vals = []
        for bin_key in sat_bins:
            val = data["soft"][l].get("by_saturation", {}).get(bin_key, {}).get("pair_acc_standard", 0.0)
            vals.append(f"{val:.4f}")
        row += " | ".join(vals) + " |"
        print(row)

if __name__ == "__main__":
    main()
