# scratch/summarize_all_mechanical.py
import json
from pathlib import Path

runs_dir = Path('/home/admin/work/s2550009/dirunc_probe/runs')
layers = [0, 4, 8, 12, 16, 20, 24, 26]

configs = {
    '7 Query Tokens (Aligned)': 'layer_sweep_gemini_nq_aligned',
    '7 Query Tokens (Unaligned)': 'layer_sweep_gemini_nq_unaligned',
    'Final Token (Aligned)': 'layer_sweep_gemini_final_token_aligned',
    'Final Token (Unaligned)': 'layer_sweep_gemini_final_token_unaligned'
}

def get_best_performance(log_path):
    if not log_path.exists():
        return None
    best_score = -1.0
    best_record = None
    with log_path.open('r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # best_score is standard_acc + macro_f1
            score = rec.get('pair_accuracy_standard', 0.0) + rec.get('macro_f1', 0.0)
            if score > best_score:
                best_score = score
                best_record = rec
    return best_record

print("# 4-Way Comparison for MECHANICAL Omission Type")
print()
print("| Layer | Config | Best Epoch | Macro F1 | Std Pair Acc | Strict Pair Acc |")
print("| :---: | :--- | :---: | :---: | :---: | :---: |")

for L in layers:
    for name, folder in configs.items():
        log_path = runs_dir / folder / f'mechanical_layer_{L}' / 'log.jsonl'
        best = get_best_performance(log_path)
        if best:
            print(f"| {L:2d} | {name:27s} | {best['epoch']:2d} | {best['macro_f1']:.4f} | {best['pair_accuracy_standard']:.4f} | {best['pair_accuracy_strict']:.4f} |")
        else:
            print(f"| {L:2d} | {name:27s} | N/A | N/A | N/A | N/A |")
