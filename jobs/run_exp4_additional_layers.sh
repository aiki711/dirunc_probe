#!/bin/bash
# 実験4 追加レイヤー個別評価 (Layer 10, 15, 20)
# すでに Layer 25 の結果があることを前提に、不足している層のみを実行する。

set -e

cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"
OUT_DIR="runs/exp4_static"

EPOCHS=3
BATCH_SIZE=2
LR=5e-4
MAX_LENGTH=256
SEED=42
MODE="query"

mkdir -p log

layers=(10 15 20)

for layer in "${layers[@]}"; do
    echo "========================================="
    echo "実験4: Layer ${layer} 個別評価開始"
    echo "========================================="
    
    python scripts/03_train_probe.py \
      --model_name ${MODEL_NAME} \
      --data_dir ${DATA_DIR} \
      --out_dir ${OUT_DIR} \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --max_length ${MAX_LENGTH} \
      --mode ${MODE} \
      --layer ${layer} \
      --save_dir ${OUT_DIR}/cache \
      --no_tqdm \
      --seed ${SEED} 2>&1 | tee log/exp4_layer${layer}_$(date +%Y%m%d_%H%M%S).log

    echo "Layer ${layer} 完了"
    echo ""
done

echo "========================================="
echo "追加レイヤー（10, 15, 20）の全評価が完了しました"
echo "========================================="
