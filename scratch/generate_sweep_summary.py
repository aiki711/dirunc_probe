import json
from pathlib import Path

layers = [0, 4, 8, 12, 16, 20, 24, 26]

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
            except Exception:
                continue
            # best_score is standard_acc + macro_f1
            score = rec.get('pair_accuracy_standard', 0.0) + rec.get('macro_f1', 0.0)
            if score > best_score:
                best_score = score
                best_record = rec
    return best_record

print("### 1. Rule-based / Mechanical Ablation Sweep (runs/layer_sweep_gemma2)")
print('| Layer | Best Epoch | Macro F1 | Std Pair Acc | Strict Pair Acc |')
print('| :--- | :---: | :---: | :---: | :---: |')
out_base_gemma2 = Path('runs/layer_sweep_gemma2')
for L in layers:
    log_path = out_base_gemma2 / f'layer_{L}' / 'log.jsonl'
    best = get_best_performance(log_path)
    if best:
        print(f'| {L:2d} | {best["epoch"]:2d} | {best["macro_f1"]:.4f} | {best["pair_accuracy_standard"]:.4f} | {best["pair_accuracy_strict"]:.4f} |')
    else:
        print(f'| {L:2d} | N/A | N/A | N/A | N/A |')

print("\n### 2. Gemini Soft/Strong Omission Sweep (runs/layer_sweep_gemini)")
print('| Layer | Omission Type | Best Epoch | Macro F1 | Std Pair Acc | Strict Pair Acc |')
print('| :--- | :--- | :---: | :---: | :---: | :---: |')
out_base_gemini = Path('runs/layer_sweep_gemini')
for L in layers:
    for o_type in ['soft', 'strong']:
        log_path = out_base_gemini / f'{o_type}_layer_{L}' / 'log.jsonl'
        best = get_best_performance(log_path)
        if best:
            print(f'| {L:2d} | {o_type:6s} | {best["epoch"]:2d} | {best["macro_f1"]:.4f} | {best["pair_accuracy_standard"]:.4f} | {best["pair_accuracy_strict"]:.4f} |')
        else:
            print(f'| {L:2d} | {o_type:6s} | N/A | N/A | N/A | N/A |')
