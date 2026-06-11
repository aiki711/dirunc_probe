#!/bin/bash
# jobs/sh/run_gemini_mechanical_4way_sweep.sh
# 機械的削除データに対して4つの比較実験を自動実行するスクリプト
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Activating virtual environment..."
source dirunc_probe/bin/activate

# --- 設定 ---
LAYERS=(0 4 8 12 16 20 24 26)
EPOCHS=5
BATCH_SIZE=16
LR=5e-4

TRAIN_MECH="data/processed/case_grammar/paired_train_gemini_mechanical.jsonl"
DEV_MECH="data/processed/case_grammar/paired_dev_gemini_mechanical.jsonl"

CACHE_DIR="data/cache"
mkdir -p "${CACHE_DIR}"

echo "=================================================="
# Phase 1: Caching representations
echo "Phase 1: Generating cached hidden states for mechanical..."
echo "=================================================="

cache_if_missing() {
    local prefix="$1"
    local split="$2"
    local mode="$3"
    local align_flag="$4"
    
    local test_file="${CACHE_DIR}/${prefix}_layer26_${split}.pt"
    if [ -f "${test_file}" ]; then
        echo "Cache ${prefix} (${split}) already exists. Skipping."
    else
        echo "Caching ${prefix} (${split})..."
        if [ -n "${align_flag}" ]; then
            python3 scripts/34_cache_all_configurations.py \
                --train_data "${TRAIN_MECH}" --dev_data "${DEV_MECH}" \
                --mode "${mode}" "${align_flag}" --prefix "${prefix}" \
                --split "${split}" --layers "0,4,8,12,16,20,24,26"
        else
            python3 scripts/34_cache_all_configurations.py \
                --train_data "${TRAIN_MECH}" --dev_data "${DEV_MECH}" \
                --mode "${mode}" --prefix "${prefix}" \
                --split "${split}" --layers "0,4,8,12,16,20,24,26"
        fi
    fi
}

# 1. NQ (7 query tokens) Aligned
cache_if_missing "mechanical" "train" "query" "--align"
cache_if_missing "mechanical" "dev" "query" "--align"

# 2. NQ (7 query tokens) Unaligned
cache_if_missing "nq_unaligned_mechanical" "train" "query" ""
cache_if_missing "nq_unaligned_mechanical" "dev" "query" ""

# 3. Final Token Aligned
cache_if_missing "final_token_aligned_mechanical" "train" "final_token" "--align"
cache_if_missing "final_token_aligned_mechanical" "dev" "final_token" "--align"

# 4. Final Token Unaligned
cache_if_missing "final_token_unaligned_mechanical" "train" "final_token" ""
cache_if_missing "final_token_unaligned_mechanical" "dev" "final_token" ""

echo ""
echo "=================================================="
# Phase 2: Sweep Training
echo "Phase 2: Running Probe Training Sweeps..."
echo "=================================================="

run_sweep() {
    local label="$1"
    local prefix="$2"
    local out_base="$3"
    
    echo "--------------------------------------------------"
    echo "Running sweep: ${label}"
    echo "--------------------------------------------------"
    
    for L in "${LAYERS[@]}"; do
        local log_file="${out_base}/mechanical_layer_${L}/log.jsonl"
        if [ -f "${log_file}" ] && [ "$(wc -l < "${log_file}")" -ge "${EPOCHS}" ]; then
            echo "[Mechanical] Layer $L already trained. Skipping."
        else
            echo "[Mechanical] Training Layer $L..."
            python3 scripts/32b_train_cached_probe.py \
                --prefix "${prefix}" \
                --layer_idx "$L" \
                --batch_size "${BATCH_SIZE}" \
                --epochs "${EPOCHS}" \
                --lr "${LR}" \
                --out_dir "${out_base}/mechanical_layer_${L}" >/dev/null
        fi
    done
}

run_sweep "7 Query Tokens (Aligned)" "mechanical" "runs/layer_sweep_gemini_nq_aligned"
run_sweep "7 Query Tokens (Unaligned)" "nq_unaligned_mechanical" "runs/layer_sweep_gemini_nq_unaligned"
run_sweep "Final Token (Aligned)" "final_token_aligned_mechanical" "runs/layer_sweep_gemini_final_token_aligned"
run_sweep "Final Token (Unaligned)" "final_token_unaligned_mechanical" "runs/layer_sweep_gemini_final_token_unaligned"

echo "=================================================="
echo "Mechanical sweeps completed!"
echo "=================================================="
