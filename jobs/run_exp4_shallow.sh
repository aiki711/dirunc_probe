#!/bin/bash
# Exp4: Static Probe 浅い層 (2, 4, 6, 8) 追加実験用スクリプト

set -e

# =========================================
# 実行パラメータの設定
# =========================================
source dirunc_probe/bin/activate

MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"
OUT_DIR="runs/balanced/experiment4_static"
EPOCHS=3
BATCH_SIZE=8  # (OOM対策: 16 -> 8 に変更済)
LR="5e-5"
MAX_LENGTH=128
MODE="query"
SEED=42

# 対象レイヤーを "2,4,6,8" に変更
LAYERS="2,4,6,8,10,12,14,16,18,20,22,24"

# ログディレクトリの作成
mkdir -p log

echo "========================================="
echo "6W1H 静的評価実験 開始 (Shallow Layers)"
echo "========================================="
echo "モデル: ${MODEL_NAME}"
echo "対象層: ${LAYERS}"
echo "出力先: ${OUT_DIR}"
echo "ログ: log/exp4_shallow_$(date +%Y%m%d_%H%M%S).log"
echo ""

# 各レイヤーごとに独立してプローブを学習（--multilayer フラグを使用しない）
# これにより、各層ごとの最良エポック・予測精度を記録する
for LAYER in ${LAYERS}; do
  echo ">>> Layer ${LAYER} の学習を開始します..."
  python scripts/03_train_probe.py \
    --model_name ${MODEL_NAME} \
    --data_dir ${DATA_DIR} \
    --out_dir ${OUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --max_length ${MAX_LENGTH} \
    --mode ${MODE} \
    --layer "${LAYER}" \
    --strip_query_in_baseline \
    --save_dir ${OUT_DIR}/cache \
    --no_tqdm \
    --seed ${SEED} 2>&1 | tee log/exp4_layer${LAYER}_$(date +%Y%m%d_%H%M%S).log
done

echo ""
echo "========================================="
echo "6W1H 静的評価実験 完了"
echo "========================================="
