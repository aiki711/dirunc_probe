#!/bin/bash
# jobs/sh/run_gemini_4way_sweep.sh
# 4つの比較実験（Query vs Final Token, Aligned vs Unaligned）を自動実行し集計するスクリプト
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# 仮想環境の有効化
echo "Activating virtual environment..."
source dirunc_probe/bin/activate

# --- 設定 ---
LAYERS=(0 4 8 12 16 20 24 26)
EPOCHS=5
BATCH_SIZE=16
LR=5e-4

TRAIN_SOFT="data/processed/case_grammar/paired_train_gemini_soft.jsonl"
DEV_SOFT="data/processed/case_grammar/paired_dev_gemini_soft.jsonl"
TRAIN_STRONG="data/processed/case_grammar/paired_train_gemini_strong.jsonl"
DEV_STRONG="data/processed/case_grammar/paired_dev_gemini_strong.jsonl"

CACHE_DIR="data/cache"
mkdir -p "${CACHE_DIR}"

echo "=================================================="
# Phase 1: Caching representations
echo "Phase 1: Generating cached hidden states..."
echo "=================================================="

# Helper function to cache only if needed
cache_if_missing() {
    local prefix="$1"
    local split="$2"
    local data_path="$3"
    local mode="$4"
    local align_flag="$5"  # either "--align" or ""
    
    local test_file="${CACHE_DIR}/${prefix}_layer26_${split}.pt"
    if [ -f "${test_file}" ]; then
        echo "Cache ${prefix} (${split}) already exists. Skipping."
    else
        echo "Caching ${prefix} (${split})..."
        if [ -n "${align_flag}" ]; then
            python3 scripts/34_cache_all_configurations.py \
                --train_data "${data_path}" --dev_data "${data_path}" \
                --mode "${mode}" "${align_flag}" --prefix "${prefix}" \
                --split "${split}" --layers "0,4,8,12,16,20,24,26"
        else
            python3 scripts/34_cache_all_configurations.py \
                --train_data "${data_path}" --dev_data "${data_path}" \
                --mode "${mode}" --prefix "${prefix}" \
                --split "${split}" --layers "0,4,8,12,16,20,24,26"
        fi
    fi
}

# 1. NQ (7 query tokens) Unaligned
cache_if_missing "nq_unaligned_soft" "train" "${TRAIN_SOFT}" "query" ""
cache_if_missing "nq_unaligned_soft" "dev" "${DEV_SOFT}" "query" ""
cache_if_missing "nq_unaligned_strong" "train" "${TRAIN_STRONG}" "query" ""
cache_if_missing "nq_unaligned_strong" "dev" "${DEV_STRONG}" "query" ""

# 2. Final Token Aligned
cache_if_missing "final_token_aligned_soft" "train" "${TRAIN_SOFT}" "final_token" "--align"
cache_if_missing "final_token_aligned_soft" "dev" "${DEV_SOFT}" "final_token" "--align"
cache_if_missing "final_token_aligned_strong" "train" "${TRAIN_STRONG}" "final_token" "--align"
cache_if_missing "final_token_aligned_strong" "dev" "${DEV_STRONG}" "final_token" "--align"

# 3. Final Token Unaligned
cache_if_missing "final_token_unaligned_soft" "train" "${TRAIN_SOFT}" "final_token" ""
cache_if_missing "final_token_unaligned_soft" "dev" "${DEV_SOFT}" "final_token" ""
cache_if_missing "final_token_unaligned_strong" "train" "${TRAIN_STRONG}" "final_token" ""
cache_if_missing "final_token_unaligned_strong" "dev" "${DEV_STRONG}" "final_token" ""


echo ""
echo "=================================================="
# Phase 2: Sweep Training
echo "Phase 2: Running Probe Training Sweeps..."
echo "=================================================="

run_sweep() {
    local label="$1"
    local prefix_base="$2"
    local out_base="$3"
    
    echo "--------------------------------------------------"
    echo "Running sweep: ${label}"
    echo "--------------------------------------------------"
    mkdir -p "${out_base}"
    
    for L in "${LAYERS[@]}"; do
        # Soft Omission
        local soft_log="${out_base}/soft_layer_${L}/log.jsonl"
        if [ -f "${soft_log}" ] && [ "$(wc -l < "${soft_log}")" -ge "${EPOCHS}" ]; then
            echo "[Soft] Layer $L already trained. Skipping."
        else
            echo "[Soft] Training Layer $L..."
            python3 scripts/32b_train_cached_probe.py \
                --prefix "${prefix_base}_soft" \
                --layer_idx "$L" \
                --batch_size "${BATCH_SIZE}" \
                --epochs "${EPOCHS}" \
                --lr "${LR}" \
                --out_dir "${out_base}/soft_layer_${L}" >/dev/null
        fi

        # Strong Omission
        local strong_log="${out_base}/strong_layer_${L}/log.jsonl"
        if [ -f "${strong_log}" ] && [ "$(wc -l < "${strong_log}")" -ge "${EPOCHS}" ]; then
            echo "[Strong] Layer $L already trained. Skipping."
        else
            echo "[Strong] Training Layer $L..."
            python3 scripts/32b_train_cached_probe.py \
                --prefix "${prefix_base}_strong" \
                --layer_idx "$L" \
                --batch_size "${BATCH_SIZE}" \
                --epochs "${EPOCHS}" \
                --lr "${LR}" \
                --out_dir "${out_base}/strong_layer_${L}" >/dev/null
        fi
    done
}

# 1. 7 Query Tokens Unaligned
run_sweep "7 Query Tokens (Unaligned)" "nq_unaligned" "runs/layer_sweep_gemini_nq_unaligned"

# 2. Final Token Aligned
run_sweep "Final Token (Aligned)" "final_token_aligned" "runs/layer_sweep_gemini_final_token_aligned"

# 3. Final Token Unaligned
run_sweep "Final Token (Unaligned)" "final_token_unaligned" "runs/layer_sweep_gemini_final_token_unaligned"

echo ""
echo "=================================================="
# Phase 3: Result Compilation & Tabulation
echo "Phase 3: Aggregating comparative results..."
echo "=================================================="

python3 -c "
import json
from pathlib import Path

layers = [0, 4, 8, 12, 16, 20, 24, 26]
omission_types = ['soft', 'strong']

runs = {
    '7 Query Tokens (Aligned)': Path('runs/layer_sweep_gemini_nq_aligned'),
    '7 Query Tokens (Unaligned)': Path('runs/layer_sweep_gemini_nq_unaligned'),
    'Final Token (Aligned)': Path('runs/layer_sweep_gemini_final_token_aligned'),
    'Final Token (Unaligned)': Path('runs/layer_sweep_gemini_final_token_unaligned'),
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
            rec = json.loads(line)
            score = rec.get('pair_accuracy_standard', 0.0) + rec.get('macro_f1', 0.0)
            if score > best_score:
                best_score = score
                best_record = rec
    return best_record

print('# Summary of 4-Way Comparison')
for o_type in omission_types:
    print(f'\n## Omission Type: {o_type.upper()}')
    print('| Layer | Config | Best Epoch | Macro F1 | Std Pair Acc | Strict Pair Acc |')
    print('| :---: | :--- | :---: | :---: | :---: | :---: |')
    for L in layers:
        for name, run_path in runs.items():
            log_path = run_path / f'{o_type}_layer_{L}' / 'log.jsonl'
            best = get_best_performance(log_path)
            if best:
                print(f'| {L:2d} | {name:26s} | {best[\"epoch\"]:2d} | {best[\"macro_f1\"]:.4f} | {best[\"pair_accuracy_standard\"]:.4f} | {best[\"pair_accuracy_strict\"]:.4f} |')
            else:
                print(f'| {L:2d} | {name:26s} | N/A | N/A | N/A | N/A |')
" > runs/comparison_summary.md

cat runs/comparison_summary.md
echo "=================================================="
echo "All sweeps completed. Compiled output is saved at runs/comparison_summary.md"
echo "=================================================="
