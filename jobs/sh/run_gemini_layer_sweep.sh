#!/bin/bash
# jobs/sh/run_gemini_layer_sweep.sh
# Soft / Strong Omission 両データに対するレイヤースイープ実行と集計スクリプト
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- 設定 ---
MODEL="google/gemma-2-2b-it"
LAYERS=(0 4 8 12 16 20 24 26)
EPOCHS=5
BATCH_SIZE=16
LR=5e-4

TRAIN_SOFT="data/processed/case_grammar/paired_train_gemini_soft.jsonl"
DEV_SOFT="data/processed/case_grammar/paired_dev_gemini_soft.jsonl"

TRAIN_STRONG="data/processed/case_grammar/paired_train_gemini_strong.jsonl"
DEV_STRONG="data/processed/case_grammar/paired_dev_gemini_strong.jsonl"

OUT_BASE="runs/layer_sweep_gemini"
LOG_DIR="logs/layer_sweep"
mkdir -p "${OUT_BASE}" "${LOG_DIR}"

echo "=================================================="
echo "Starting Layer Sweep for Gemini Soft / Strong Probe"
echo "Model: ${MODEL}"
echo "Layers to sweep: ${LAYERS[*]}"
echo "=================================================="

# 各レイヤーについてループ実行
for L in "${LAYERS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Processing Layer $L / ${LAYERS[-1]}"
    echo "--------------------------------------------------"

    # --- Soft Omission ---
    echo "[Soft] Training layer $L..."
    python3 scripts/32_train_contrastive_probe.py \
        --model_name "${MODEL}" \
        --layer_idx "$L" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --train_data "${TRAIN_SOFT}" \
        --dev_data "${DEV_SOFT}" \
        --out_dir "${OUT_BASE}/soft_layer_${L}" \
        2>&1 | tee "${LOG_DIR}/soft_layer_${L}.log"

    # --- Strong Omission ---
    echo "[Strong] Training layer $L..."
    python3 scripts/32_train_contrastive_probe.py \
        --model_name "${MODEL}" \
        --layer_idx "$L" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --train_data "${TRAIN_STRONG}" \
        --dev_data "${DEV_STRONG}" \
        --out_dir "${OUT_BASE}/strong_layer_${L}" \
        2>&1 | tee "${LOG_DIR}/strong_layer_${L}.log"
done

echo ""
echo "=================================================="
echo "Layer Sweep Completed! Generating Summary..."
echo "=================================================="

python3 -c "
import json
from pathlib import Path

out_base = Path('${OUT_BASE}')
layers = [${LAYERS[@]// /, }]

def get_best_performance(log_path):
    if not log_path.exists():
        return None
    best_score = -1.0
    best_record = None
    with log_path.open('r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # best_score is standard_acc + macro_f1
            score = rec.get('pair_accuracy_standard', 0.0) + rec.get('macro_f1', 0.0)
            if score > best_score:
                best_score = score
                best_record = rec
    return best_record

print('| Layer | Omission Type | Best Epoch | Macro F1 | Std Pair Acc | Strict Pair Acc |')
print('| :--- | :--- | :---: | :---: | :---: | :---: |')

for L in layers:
    for o_type in ['soft', 'strong']:
        log_path = out_base / f'{o_type}_layer_{L}' / 'log.jsonl'
        best = get_best_performance(log_path)
        if best:
            print(f'| {L:2d} | {o_type:6s} | {best[\"epoch\"]:2d} | {best[\"macro_f1\"]:.4f} | {best[\"pair_accuracy_standard\"]:.4f} | {best[\"pair_accuracy_strict\"]:.4f} |')
        else:
            print(f'| {L:2d} | {o_type:6s} | N/A | N/A | N/A | N/A |')
"

echo "=================================================="
