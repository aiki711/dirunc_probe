#!/bin/bash
# jobs/sh/run_gemini_nq_layer_sweep_cached.sh
# Soft / Strong Omission 両データに対する自然言語クエリ(NQ)の高速キャッシュ版レイヤースイープ実行と集計スクリプト
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- 設定 ---
LAYERS=(0 4 8 12 16 20 24 26)
EPOCHS=5
BATCH_SIZE=16
LR=5e-4

OUT_BASE="runs/layer_sweep_gemini_nq_aligned"
LOG_DIR="logs/layer_sweep_nq_aligned"
mkdir -p "${OUT_BASE}" "${LOG_DIR}"

echo "=================================================="
echo "Starting Accelerated Natural Query Layer Sweep (Cached)"
echo "Layers to sweep: ${LAYERS[*]}"
echo "=================================================="

# 各レイヤーについてループ実行
for L in "${LAYERS[@]}"; do
    echo ""
    echo "--------------------------------------------------"
    echo "Processing Layer $L / ${LAYERS[-1]}"
    echo "--------------------------------------------------"

    # --- Soft Omission ---
    SOFT_LOG="${OUT_BASE}/soft_layer_${L}/log.jsonl"
    if [ -f "${SOFT_LOG}" ] && [ "$(wc -l < "${SOFT_LOG}")" -ge "${EPOCHS}" ]; then
        echo "[Soft] Layer $L already trained. Skipping."
    else
        echo "[Soft] Training cached layer $L..."
        python3 scripts/32b_train_cached_probe.py \
            --prefix "soft" \
            --layer_idx "$L" \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --lr "${LR}" \
            --out_dir "${OUT_BASE}/soft_layer_${L}" \
            2>&1 | tee "${LOG_DIR}/soft_layer_${L}_cached.log"
    fi

    # --- Strong Omission ---
    STRONG_LOG="${OUT_BASE}/strong_layer_${L}/log.jsonl"
    if [ -f "${STRONG_LOG}" ] && [ "$(wc -l < "${STRONG_LOG}")" -ge "${EPOCHS}" ]; then
        echo "[Strong] Layer $L already trained. Skipping."
    else
        echo "[Strong] Training cached layer $L..."
        python3 scripts/32b_train_cached_probe.py \
            --prefix "strong" \
            --layer_idx "$L" \
            --batch_size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --lr "${LR}" \
            --out_dir "${OUT_BASE}/strong_layer_${L}" \
            2>&1 | tee "${LOG_DIR}/strong_layer_${L}_cached.log"
    fi
done

echo ""
echo "=================================================="
echo "Accelerated Layer Sweep Completed! Generating Summary..."
echo "=================================================="

python3 -c "
import json
from pathlib import Path

out_base = Path('${OUT_BASE}')
layers = [int(x) for x in \"${LAYERS[*]}\".split()]

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
