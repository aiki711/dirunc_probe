#!/bin/bash
# 実験2（クラスごと閾値最適化）をBalancedデータセットで実行するスクリプト

set -e

# プロジェクトルートへ移動
cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

# 設定
MODEL_NAME="google/gemma-2-2b-it"
DATA_JSONL="data/processed/mixed/dirunc/dev_balanced.jsonl"
OUT_BASE="runs/balanced/experiment2_perclass"
IMBALANCED_RUNS="runs/imbalanced/experiment2_perclass_v1" # v1がついている可能性が高いので要確認

# レイヤー一覧 (summary.jsonより: 5, 10, 15, 20, 25)
LAYERS=(5 10 15 20 25)

echo "========================================="
echo "実験2（Balanced評価）開始"
echo "========================================="


    
# experiment2_perclass ディレクトリを探す
EXP_DIR=$(find runs/imbalanced -maxdepth 1 -name "experiment2_perclass*" | head -n 1)

if [ -z "$EXP_DIR" ]; then
    echo "Error: Experiment 2 directory not found in runs/imbalanced!"
    exit 1
fi
echo "Experiment Directory found: ${EXP_DIR}"

for layer in "${LAYERS[@]}"; do
    echo ""
    echo "Processing Layer ${layer}..."
    
    # その中の best_query_layer{layer}.pt を探す
    MODEL_PATH=$(find "$EXP_DIR" -name "best_query_layer${layer}.pt" | head -n 1)
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: Model for layer ${layer} not found!"
        continue
    fi
    
    echo "Found model: ${MODEL_PATH}"
    
    OUT_DIR="${OUT_BASE}/query/layer_${layer}"
    mkdir -p "$OUT_DIR"
    
    python scripts/03b_eval_probe.py \
      --model_name "${MODEL_NAME}" \
      --data_jsonl "${DATA_JSONL}" \
      --model_path "${MODEL_PATH}" \
      --out_dir "${OUT_DIR}" \
      --layer_idx "${layer}" \
      --mode "query" \
      --summary_json "${EXP_DIR}/summary.json" \
      --batch_size 32
      
    echo "Layer ${layer} finished."
done

echo ""
echo "========================================="
echo "実験2（Balanced評価）完了"
echo "========================================="
