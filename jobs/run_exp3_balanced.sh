#!/bin/bash
# 実験3（マルチレイヤー＋クラスごと閾値最適化）をBalancedデータセットで実行するスクリプト
# Output: runs/balanced/experiment3_multilayer

set -e

# プロジェクトルートへ移動
cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

# ========== 設定 ==========
MODEL_NAME="google/gemma-2-2b-it" 
DATA_DIR="data/processed/mixed/dirunc"     # Train data source
EVAL_DATA_DIR="data/balanced"              # Dev data source
DEV_FILE="dev_balanced.jsonl"              # Dev data filename

OUT_DIR="runs/balanced/experiment3_multilayer"

EPOCHS=3
BATCH_SIZE=8
LR=5e-4
MAX_LENGTH=256
SEED=42
MODE="query"

echo "========================================="
echo "実験3（Balanced評価）開始"
echo "========================================="
echo "モデル: ${MODEL_NAME}"
echo "出力: ${OUT_DIR}"
echo ""

# ディレクトリ作成
mkdir -p "${OUT_DIR}"
mkdir -p log

python scripts/03_train_probe.py \
  --model_name "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --eval_data_dir "${EVAL_DATA_DIR}" \
  --dev_file "${DEV_FILE}" \
  --out_dir "${OUT_DIR}" \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --max_length ${MAX_LENGTH} \
  --mode ${MODE} \
  --multilayer \
  --fusion_layers "10,15,20,25" \
  --strip_query_in_baseline \
  --no_tqdm \
  --seed ${SEED} 2>&1 | tee log/experiment3_balanced_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "実験3（Balanced評価）完了"
echo "========================================="
