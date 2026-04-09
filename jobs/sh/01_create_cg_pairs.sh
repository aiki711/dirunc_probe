#!/bin/bash
# jobs/sh/01_create_cg_pairs.sh
# Case Grammar 最小ペアデータの生成
# 実行方法: bash jobs/sh/01_create_cg_pairs.sh
# 前提: プロジェクトルート (dirunc_probe/) で実行すること

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "========================================="
echo "Step 1: Case Grammar Pair Generation"
echo "Start: $(date)"
echo "Dir  : $(pwd)"
echo "========================================="

# --- 環境確認 ---
echo "[CHECK] Python: $(python3 --version)"
echo "[CHECK] CUDA  : $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'torch not found')"

# --- 必要なディレクトリの確認 ---
MISSING=0
for p in \
    "data/raw/sgd/train" \
    "data/raw/multiwoz/data.json" \
    "data/processed/sgd/required_slots_by_service_intent.json" \
    "data/processed/sgd/slot_meta_by_service_slot.json"
do
    if [ ! -e "${p}" ]; then
        echo "[WARN] Missing: ${p}"
        MISSING=1
    else
        echo "[OK]   Found  : ${p}"
    fi
done

# QA-SRL はオプション
if [ ! -e "temp_qasrl/qasrl-v2/orig/dev.jsonl.gz" ]; then
    echo "[WARN] QA-SRL not found (will be skipped automatically)"
fi

if [ "${MISSING}" -eq 1 ]; then
    echo ""
    echo "[ERROR] 必要なデータが見つかりません。"
    echo "        以下のコマンドで元環境からコピーしてください:"
    echo "          rsync -av s2550009@<元サーバ>:/home/s2550009/dirunc_probe/data/ ./data/"
    echo "          rsync -av s2550009@<元サーバ>:/home/s2550009/dirunc_probe/temp_qasrl/ ./temp_qasrl/"
    exit 1
fi

# --- 出力ディレクトリ作成 ---
mkdir -p data/processed/case_grammar logs

# --- 実行 ---
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 scripts/33_create_case_grammar_pairs.py 2>&1 | tee logs/cg_data_gen.log

echo "========================================="
echo "End: $(date)"
echo "[OUTPUT] data/processed/case_grammar/"
ls -lh data/processed/case_grammar/
echo "========================================="
