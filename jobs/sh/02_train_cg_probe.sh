#!/bin/bash
# jobs/sh/02_train_cg_probe.sh
# Case Grammar 対照プローブの学習
# 実行方法: bash jobs/sh/02_train_cg_probe.sh [オプション]
#
# オプション例:
#   MODEL=meta-llama/Llama-2-7b-hf LAYER=16 bash jobs/sh/02_train_cg_probe.sh
#   EPOCHS=3 LR=1e-3 bash jobs/sh/02_train_cg_probe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# --- ハイパーパラメータ（環境変数で上書き可） ---
MODEL="${MODEL:-google/gemma-2-2b-it}"
LAYER="${LAYER:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-5e-4}"
MARGIN="${MARGIN:-0.2}"
LAMBDA_MARGIN="${LAMBDA_MARGIN:-1.0}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-runs/cg_probe}"

# --- データパス ---
TRAIN_DATA="data/processed/case_grammar/cg_train.jsonl"
DEV_DATA="data/processed/case_grammar/cg_dev.jsonl"

echo "========================================="
echo "Step 2: CG Contrastive Probe Training"
echo "Start : $(date)"
echo "Model : ${MODEL}"
echo "Layer : ${LAYER}"
echo "Train : ${TRAIN_DATA}"
echo "Dev   : ${DEV_DATA}"
echo "OutDir: ${OUT_DIR}"
echo "========================================="

# --- GPU 確認 ---
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[GPU] $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
else
    echo "[WARN] GPU not available. Training will use CPU (very slow)."
fi

# --- データ存在確認 ---
MISSING=0
for p in "${TRAIN_DATA}" "${DEV_DATA}"; do
    if [ ! -f "${p}" ]; then
        echo "[ERROR] 見つかりません: ${p}"
        echo "        先に 01_create_cg_pairs.sh を実行してください。"
        echo "        または元環境からコピー:"
        echo "          rsync -av s2550009@<元サーバ>:/home/s2550009/dirunc_probe/data/processed/case_grammar/ ./data/processed/case_grammar/"
        MISSING=1
    else
        SIZE=$(wc -l < "${p}")
        echo "[OK]   ${p} (${SIZE} lines)"
    fi
done

if [ "${MISSING}" -eq 1 ]; then
    exit 1
fi

# --- 出力ディレクトリ ---
mkdir -p "${OUT_DIR}" logs
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# --- 学習実行 ---
python3 scripts/32_train_contrastive_probe.py \
    --model_name      "${MODEL}" \
    --layer_idx       "${LAYER}" \
    --batch_size      "${BATCH_SIZE}" \
    --max_length      "${MAX_LENGTH}" \
    --epochs          "${EPOCHS}" \
    --lr              "${LR}" \
    --margin          "${MARGIN}" \
    --lambda_margin   "${LAMBDA_MARGIN}" \
    --seed            "${SEED}" \
    --train_data      "${TRAIN_DATA}" \
    --dev_data        "${DEV_DATA}" \
    --out_dir         "${OUT_DIR}" \
    2>&1 | tee logs/cg_probe_train.log

echo "========================================="
echo "End: $(date)"
echo ""
echo "[RESULTS]"
echo "  Best model : ${OUT_DIR}/best_probe_layer${LAYER}.pt"
echo "  Log        : ${OUT_DIR}/log.jsonl"
echo ""

# --- 最終エポックの結果を表示 ---
if [ -f "${OUT_DIR}/log.jsonl" ]; then
    echo "  最終エポック結果:"
    tail -n 1 "${OUT_DIR}/log.jsonl" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f\"    MacroF1     : {d.get('macro_f1', 0):.4f}\")
print(f\"    Pair Acc    : {d.get('pair_accuracy_standard', 0):.4f}\")
print(f\"    Pair Acc(S) : {d.get('pair_accuracy_strict', 0):.4f}\")
if 'by_case_role' in d:
    print('    --- by_case_role ---')
    for role, v in sorted(d['by_case_role'].items()):
        print(f\"      {role:12s}: Acc={v['pair_acc_standard']:.3f}  F1={v['macro_f1']:.3f}  (n={v['n']})\")
if 'by_saturation' in d:
    print('    --- by_saturation ---')
    for k, v in sorted(d['by_saturation'].items()):
        print(f\"      {k:22s}: Acc={v['pair_acc_standard']:.3f}  (n={v['n']})\")
"
fi

echo "========================================="
