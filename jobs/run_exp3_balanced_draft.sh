#!/bin/bash
# 実験3（マルチレイヤー＋クラスごと閾値最適化）をBalancedデータセットで実行するスクリプト
# Output: runs/balanced/experiment3_multilayer

set -e

# プロジェクトルートへ移動
cd "$(dirname "$0")/.." || exit 1
source dirunc_probe/bin/activate

# ========== 設定 ==========
MODEL_NAME="google/gemma-2-2b-it"
DATA_DIR="data/processed/mixed/dirunc"  # Source of data
OUT_DIR="runs/balanced/experiment3_multilayer"

# Balance specific check
# Note: The script uses DATA_DIR to find train/dev.jsonl. 
# We need to make sure we are using the balanced dev set if that's the intention.
# scripts/03_train_probe.py arguments:
# --data_dir: source of train.jsonl
# --eval_data_dir: source of dev.jsonl (optional, if different)
# --dev_file: filename (default dev.jsonl)

# For experiment 2, we used `data/balanced/dev_balanced.jsonl`.
# Let's check `jobs/run_exp2_balanced.sh` again. 
# run_exp2_balanced.sh used `scripts/03b_eval_probe.py` which takes --data_jsonl.
# `scripts/03_train_probe.py` uses --data_dir and --dev_file.

# We created `data/balanced/dev_balanced.jsonl` (symlink).
# So we can set --eval_data_dir data/balanced --dev_file dev_balanced.jsonl
# OR just --dev_file dev_balanced.jsonl if we point data_dir to data/balanced (but train.jsonl might be missing there).

# Let's check where train.jsonl is.
# DATA_DIR="data/processed/mixed/dirunc" has train.jsonl.
# We want to use that for training.
# For validation, does the user want balanced dev?
# "experiment 3 ... balanced" implies balanced evaluation. 

# Let's assume we use:
# --data_dir data/processed/mixed/dirunc
# --eval_data_dir data/balanced
# --dev_file dev_balanced.jsonl

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

python scripts/03_train_probe.py \
  --model_name ${MODEL_NAME} \
  --data_dir ${DATA_DIR} \
  --eval_data_dir "data/balanced" \
  --dev_file "dev_balanced.jsonl" \
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
  --seed ${SEED} 2>&1 | tee log/experiment3_balanced_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================="
echo "実験3（Balanced評価）完了"
echo "========================================="
