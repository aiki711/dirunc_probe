#!/bin/bash
# プローブモデル実験の統合実行スクリプト
# 
# 3つの実験タイプを実行可能：
# 1. ベースライン（単一閾値）
# 2. クラスごと閾値最適化
# 3. マルチレイヤー＋クラスごと閾値最適化

set -e  # エラーで停止

cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

# ========== 設定 ==========
MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"
EPOCHS=3
BATCH_SIZE=8
LR=5e-4
MAX_LENGTH=256
SEED=42
MODE="query"  # baselineまたはqueryまたはboth

# 実験タイプ（コマンドライン引数で指定）
EXPERIMENT_TYPE="${1:-all}"  # baseline, perclass, multilayer, all

echo "========================================="
echo "プローブモデル実験実行"
echo "========================================="
echo "実験タイプ: ${EXPERIMENT_TYPE}"
echo "モデル: ${MODEL_NAME}"
echo "データ: ${DATA_DIR}"
echo ""

# ========== 実験1: ベースライン（単一閾値） ==========
if [ "${EXPERIMENT_TYPE}" = "baseline" ] || [ "${EXPERIMENT_TYPE}" = "all" ]; then
    echo ""
    echo "========================================="
    echo "実験1: ベースライン（単一閾値）"
    echo "========================================="
    echo "出力: runs/experiment1_baseline"
    echo ""
    
    python scripts/03_train_probe.py \
      --model_name ${MODEL_NAME} \
      --data_dir ${DATA_DIR} \
      --out_dir runs/experiment1_baseline \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --max_length ${MAX_LENGTH} \
      --mode ${MODE} \
      --sweep \
      --strip_query_in_baseline \
      --no_tqdm \
      --seed ${SEED} 2>&1 | tee log/experiment1_baseline_$(date +%Y%m%d_%H%M%S).log
    
    echo ""
    echo "✓ 実験1完了"
fi

# ========== 実験2: クラスごと閾値最適化 ==========
if [ "${EXPERIMENT_TYPE}" = "perclass" ] || [ "${EXPERIMENT_TYPE}" = "all" ]; then
    echo ""
    echo "========================================="
    echo "実験2: クラスごと閾値最適化"
    echo "========================================="
    echo "出力: runs/experiment2_perclass"
    echo ""
    
    python scripts/03_train_probe.py \
      --model_name ${MODEL_NAME} \
      --data_dir ${DATA_DIR} \
      --out_dir runs/experiment2_perclass \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --max_length ${MAX_LENGTH} \
      --mode ${MODE} \
      --sweep \
      --strip_query_in_baseline \
      --no_tqdm \
      --seed ${SEED} 2>&1 | tee log/experiment2_perclass_$(date +%Y%m%d_%H%M%S).log
    
    echo ""
    echo "✓ 実験2完了"
    echo "注意: 実験1と実験2は同じコードですが、train_probe_from_cache内で"
    echo "      per-class閾値最適化が自動的に実行されます。"
fi

# ========== 実験3: マルチレイヤー＋クラスごと閾値最適化 ==========
if [ "${EXPERIMENT_TYPE}" = "multilayer" ] || [ "${EXPERIMENT_TYPE}" = "all" ]; then
    echo ""
    echo "========================================="
    echo "実験3: マルチレイヤー＋クラスごと閾値最適化"
    echo "========================================="
    echo "融合レイヤー: 10, 15, 20, 25"
    echo "出力: runs/experiment3_multilayer"
    echo ""
    
    python scripts/03_train_probe.py \
      --model_name ${MODEL_NAME} \
      --data_dir ${DATA_DIR} \
      --out_dir runs/experiment3_multilayer \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr ${LR} \
      --max_length ${MAX_LENGTH} \
      --mode ${MODE} \
      --multilayer \
      --fusion_layers "10,15,20,25" \
      --strip_query_in_baseline \
      --no_tqdm \
      --seed ${SEED} 2>&1 | tee log/experiment3_multilayer_$(date +%Y%m%d_%H%M%S).log
    
    echo ""
    echo "✓ 実験3完了"
fi

# ========== 完了 ==========
echo ""
echo "========================================="
echo "全実験完了！"
echo "========================================="
echo ""
echo "結果の場所:"
if [ "${EXPERIMENT_TYPE}" = "baseline" ] || [ "${EXPERIMENT_TYPE}" = "all" ]; then
    echo "  実験1（ベースライン）:     runs/experiment1_baseline/summary.json"
fi
if [ "${EXPERIMENT_TYPE}" = "perclass" ] || [ "${EXPERIMENT_TYPE}" = "all" ]; then
    echo "  実験2（クラス閾値）:        runs/experiment2_perclass/summary.json"
fi
if [ "${EXPERIMENT_TYPE}" = "multilayer" ] || [ "${EXPERIMENT_TYPE}" = "all" ]; then
    echo "  実験3（マルチレイヤー）:    runs/experiment3_multilayer/summary.json"
fi
echo ""
echo "比較レポートの生成:"
echo "  python scripts/compare_results.py \\"
echo "    --baseline runs/experiment1_baseline \\"
echo "    --perclass runs/experiment2_perclass \\"
echo "    --multilayer runs/experiment3_multilayer"
echo ""
