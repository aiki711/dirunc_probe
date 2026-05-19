import json
from pathlib import Path

soft_log = Path("runs/contrastive_probe_gemini_soft_L16/log.jsonl")
strong_log = Path("runs/contrastive_probe_gemini_strong_L16/log.jsonl")

def read_metrics(log_file):
    metrics = {}
    if not log_file.exists():
        return metrics
    with log_file.open("r") as f:
        for line in f:
            try:
                data = json.loads(line)
                ep = data.get("epoch")
                if ep is not None:
                    metrics[ep] = {
                        "macro_f1": data.get("macro_f1", 0),
                        "pair_acc_std": data.get("pair_accuracy_standard", 0),
                        "pair_acc_str": data.get("pair_accuracy_strict", 0),
                    }
            except Exception:
                pass
    return metrics

soft_data = read_metrics(soft_log)
strong_data = read_metrics(strong_log)

epochs = sorted(set(list(soft_data.keys()) + list(strong_data.keys())))

if not epochs:
    print("No data available yet.")
else:
    print(f"{'Epoch':<5} | {'[Soft] Pair Acc (Std)':<21} | {'[Strong] Pair Acc (Std)':<21} | {'[Soft] Macro F1':<15} | {'[Strong] Macro F1':<15}")
    print("-" * 90)
    for ep in epochs:
        s_std = f"{soft_data.get(ep, {}).get('pair_acc_std', 0):.4f}" if ep in soft_data else "-"
        st_std = f"{strong_data.get(ep, {}).get('pair_acc_std', 0):.4f}" if ep in strong_data else "-"
        s_f1 = f"{soft_data.get(ep, {}).get('macro_f1', 0):.4f}" if ep in soft_data else "-"
        st_f1 = f"{strong_data.get(ep, {}).get('macro_f1', 0):.4f}" if ep in strong_data else "-"
        print(f"{ep:<5} | {s_std:<21} | {st_std:<21} | {s_f1:<15} | {st_f1:<15}")

