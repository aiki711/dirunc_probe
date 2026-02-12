#!/bin/bash
# マルチレイヤープローブの学習スクリプト

cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"
EPOCHS=3
BATCH_SIZE=8
LR=5e-4
MAX_LENGTH=256
SEED=42

echo "========================================="
echo "マルチレイヤー特徴結合実験"
echo "========================================="
echo "融合レイヤー: 10, 15, 20, 25"
echo "出力: runs/mixed_llm_gemma2b_multilayer"
echo ""
echo "注意: この実験では、融合するレイヤーを個別に学習してから"
echo "      結果を統合する方法を使用します。"
echo ""

# レイヤーごとに学習（既に完了している場合はスキップ）
FUSION_LAYERS=(10 15 20 25)

for LAYER in "${FUSION_LAYERS[@]}"; do
    LAYER_DIR="runs/mixed_llm_gemma2b_multilayer/query/layer_${LAYER}"
    if [ -f "${LAYER_DIR}/best_query_layer${LAYER}.pt" ]; then
        echo "[Skip] Layer ${LAYER} already trained"
    else
        echo "[Training] Layer ${LAYER}..."
        python scripts/03_train_probe.py \
          --model_name ${MODEL_NAME} \
          --data_dir ${DATA_DIR} \
          --out_dir runs/mixed_llm_gemma2b_multilayer \
          --epochs ${EPOCHS} \
          --batch_size ${BATCH_SIZE} \
          --lr ${LR} \
          --max_length ${MAX_LENGTH} \
          --mode query \
          --layer ${LAYER} \
          --no_tqdm \
          --seed ${SEED} 2>&1 | tee -a log/train_multilayer_$(date +%Y%m%d_%H%M%S).log
    fi
done

echo ""
echo "(simulated multi-layer) 各レイヤーの学習完了！"
echo "結果: runs/mixed_llm_gemma2b_multilayer/summary.json"
echo ""
echo "注意: 実際のマルチレイヤー特徴結合（学習可能な重み）は"
echo "      03_train_probe.pyの修正が完了した後に実行してください。"
