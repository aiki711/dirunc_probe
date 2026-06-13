# scratch/summarize_nq_sweep.py
import json
from pathlib import Path

out_base = Path('/home/admin/work/s2550009/dirunc_probe/runs/layer_sweep_gemini_nq_aligned')
layers = [0, 4, 8, 12, 16, 20, 24, 26]
omission_types = ['soft', 'strong', 'mechanical']

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

print("# NQ (Natural Query) Layer Sweep Results summary\n")

for o_type in omission_types:
    print(f"## Omission Type: {o_type.upper()}")
    print('| Layer | Best Epoch | Macro F1 | Std Pair Acc | Strict Pair Acc |')
    print('| :---: | :---: | :---: | :---: | :---: |')
    for L in layers:
        log_path = out_base / f'{o_type}_layer_{L}' / 'log.jsonl'
        best = get_best_performance(log_path)
        if best:
            print(f'| {L:2d} | {best["epoch"]:2d} | {best["macro_f1"]:.4f} | {best["pair_accuracy_standard"]:.4f} | {best["pair_accuracy_strict"]:.4f} |')
        else:
            print(f'| {L:2d} | N/A | N/A | N/A | N/A |')
    print()
