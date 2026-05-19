import json
from pathlib import Path

log_file = Path("runs/contrastive_probe_gemini_soft_L16/log.jsonl")
if not log_file.exists():
    print("Log file not found.")
else:
    print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Dev Loss':<10} | {'Macro F1':<10} | {'Pair Acc (Std)':<15} | {'Pair Acc (Strict)':<15}")
    print("-" * 75)
    with log_file.open("r") as f:
        for line in f:
            try:
                data = json.loads(line)
                print(f"{data.get('epoch', '-'):<5} | "
                      f"{data.get('train_loss', 0):<10.4f} | "
                      f"{data.get('loss', 0):<10.4f} | "
                      f"{data.get('macro_f1', 0):<10.4f} | "
                      f"{data.get('pair_accuracy_standard', 0):<15.4f} | "
                      f"{data.get('pair_accuracy_strict', 0):<15.4f}")
            except Exception as e:
                pass
