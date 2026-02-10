#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Activate virtualenv
if [ -f dirunc_probe/bin/activate ]; then
    source dirunc_probe/bin/activate
elif [ -f ../dirunc_probe/bin/activate ]; then
    source ../dirunc_probe/bin/activate
fi

cd "$(dirname "$0")/.."
mkdir -p log runs

echo "=== LLM PROBE TRAINING START ==="
date
hostname
nvidia-smi || true

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Model Selection
MODEL_GEMMA="google/gemma-2-2b-it"
MODEL_LLAMA="meta-llama/Meta-Llama-3-8B"

# Use Gemma-2-2B
MODEL_NAME=$MODEL_GEMMA

# Mixed Dataset (QA-SRL + MultiWOZ)
DATA_DIR="data/processed/mixed/dirunc"
OUT_DIR="runs/mixed_llm_gemma2b"

echo "Model: ${MODEL_NAME}"
echo "Data: ${DATA_DIR}"
echo "OutDir: ${OUT_DIR}"

# Training with Mixed Dataset
# Note: Using lower learning rate for LLMs to avoid instability
python scripts/03_train_probe.py \
  --model_name "${MODEL_NAME}" \
  --data_dir "${DATA_DIR}" \
  --out_dir "${OUT_DIR}" \
  --epochs 3 \
  --batch_size 4 \
  --extract_batch_size 8 \
  --lr 5e-5 \
  --max_length 256 \
  --mode query \
  --layer -1 \
  --sweep \
  --no_tqdm \
  --tqdm_mininterval 30

echo "=== Training Complete ==="

# Source-Specific Evaluation
echo "=== Evaluating on QA-SRL and MultiWOZ Test Sets ==="
BEST_LAYER=$(python -c "import json; d=json.load(open('${OUT_DIR}/summary.json')); print(max([(v['best']['macro_f1_posonly'], k) for k,v in d.items() if 'query/layer' in k])[1].split('/')[-1].replace('layer_',''))")

echo "Best Layer: ${BEST_LAYER}"

python scripts/eval_split.py \
  --model_dir "${OUT_DIR}/query/layer_${BEST_LAYER}" \
  --layer_idx "${BEST_LAYER}" \
  --model_base "${MODEL_NAME}" \
  --qasrl_test data/processed/qasrl/dirunc/test.jsonl \
  --multiwoz_test data/processed/multiwoz/dirunc/test.jsonl \
  --out_dir "${OUT_DIR}/eval_layer${BEST_LAYER}"

echo "=== LLM PROBE TRAINING END ==="
date
