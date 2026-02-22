#!/bin/bash
# 実験4（Step 3-1: 6W1H静的評価）
# 統合データセット（train/dev）で学習し、testデータ（静的評価）で性能を検証する。
# 出力先: runs/exp4_static

set -e

# プロジェクトルートへ移動
cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

# ========== 設定 ==========
MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"
OUT_DIR="runs/exp4_static"

EPOCHS=3
BATCH_SIZE=2       # OOM対策として極小化
LR=5e-4
MAX_LENGTH=256
SEED=42
MODE="query"

# ログディレクトリの確保
mkdir -p log
mkdir -p $OUT_DIR

echo "========================================="
echo "6W1H 静的評価実験 開始 (Step 3-1)"
echo "========================================="
echo "モデル: ${MODEL_NAME}"
echo "出力先: ${OUT_DIR}"
echo "ログ: log/exp4_static_$(date +%Y%m%d_%H%M%S).log"
echo ""

# OOM対策: キャッシュの強制クリア
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

# 実行
# OOM回避のため --save_dir でディスクキャッシュ処理を有効化する設定を付与
python scripts/03_train_probe.py \
  --model_name ${MODEL_NAME} \
  --data_dir ${DATA_DIR} \
  --out_dir ${OUT_DIR} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --max_length ${MAX_LENGTH} \
  --mode ${MODE} \
  --multilayer \
  --fusion_layers "10,15,20,25" \
  --strip_query_in_baseline \
  --no_tqdm \
  --seed ${SEED} 2>&1 | tee log/exp4_static_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "6W1H 静的評価実験 完了"
echo "========================================="
