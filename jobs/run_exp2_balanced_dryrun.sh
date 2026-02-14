#!/bin/bash
# 実験2（クラスごと閾値最適化）をBalancedデータセットで実行するスクリプト（ドライラン）

set -e

# プロジェクトルートへ移動
cd "$(dirname "$0")/.." || exit 1
# source dirunc_probe/bin/activate # ドライランなので不要

# 設定
MODEL_NAME="google/gemma-2-2b-it"
DATA_JSONL="data/processed/mixed/dirunc/dev_balanced.jsonl"
OUT_BASE="runs/balanced/experiment2_perclass"

# レイヤー一覧 (summary.jsonより: 5, 10, 15, 20, 25)
LAYERS=(5 10 15 20 25)

echo "========================================="
echo "実験2（Balanced評価）DRY RUN"
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
    else 
        echo "Found model: ${MODEL_PATH}"
    fi
    
    OUT_DIR="${OUT_BASE}/query/layer_${layer}"
    
    echo "Command to run:"
    echo "  python scripts/03b_eval_probe.py \\"
    echo "    --model_name \"${MODEL_NAME}\" \\"
    echo "    --data_jsonl \"${DATA_JSONL}\" \\"
    echo "    --model_path \"${MODEL_PATH}\" \\"
    echo "    --out_dir \"${OUT_DIR}\" \\"
    echo "    --layer_idx \"${layer}\" \\"
    echo "    --mode \"query\" \\"
    echo "    --summary_json \"${EXP_DIR}/summary.json\" \\"
    echo "    --batch_size 32"
      
    echo "Layer ${layer} (simulated) finished."
done
