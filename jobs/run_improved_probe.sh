#!/bin/bash
# 改善版プローブモデルのトレーニングスクリプト
# クラスごと閾値最適化とマルチレイヤー特徴結合を実行

set -e  # エラーで停止

# 仮想環境を有効化
source dirunc_probe/bin/activate

MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"
EPOCHS=3
BATCH_SIZE=8
LR=5e-4
MAX_LENGTH=256
SEED=42

echo "==================================="
echo "改善版プローブモデルのトレーニング"
echo "==================================="
echo ""
echo "ベースライン結果: runs/mixed_llm_gemma2b/"
echo ""

# 1. クラスごと閾値最適化版（単一レイヤー）
echo "----------------------------------------"
echo "1. クラスごと閾値最適化版の学習開始"
echo "   出力ディレクトリ: runs/mixed_llm_gemma2b_perclass"
echo "----------------------------------------"
python scripts/03_train_probe.py \
  --model_name ${MODEL_NAME} \
  --data_dir ${DATA_DIR} \
  --out_dir runs/mixed_llm_gemma2b_perclass \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --max_length ${MAX_LENGTH} \
  --mode query \
  --sweep \
  --no_tqdm \
  --seed ${SEED} 2>&1 | tee log/train_perclass_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "クラスごと閾値最適化版の学習完了！"
echo "結果の保存先: runs/mixed_llm_gemma2b_perclass/summary.json"
echo ""

# 2. マルチレイヤー版
echo "----------------------------------------"
echo "2. マルチレイヤー版の学習開始"
echo "   使用レイヤー: 10, 15, 20, 25"
echo "   出力ディレクトリ: runs/mixed_llm_gemma2b_multilayer"
echo "----------------------------------------"

# マルチレイヤーモードを追加したスクリプトを作成します
# とりあえずスキップして、まずは単一レイヤー版で検証
echo "※マルチレイヤー版は実装後に実行予定"
echo ""

# 3. 結果の比較
echo "----------------------------------------"
echo "3. 結果の比較"
echo "----------------------------------------"
echo ""
echo "ベースライン vs クラスごと閾値最適化の比較:"
python scripts/compare_results.py \
  --baseline runs/mixed_llm_gemma2b \
  --perclass runs/mixed_llm_gemma2b_perclass \
  --output comparison_perclass.md

echo ""
echo "==================================="
echo "全ての実験が完了しました！"
echo "==================================="
echo ""
echo "結果のサマリ:"
echo "  - ベースライン: runs/mixed_llm_gemma2b/summary.json"
echo "  - クラスごと閾値: runs/mixed_llm_gemma2b_perclass/summary.json"
echo "  - 比較レポート: comparison_perclass.md"
echo ""
