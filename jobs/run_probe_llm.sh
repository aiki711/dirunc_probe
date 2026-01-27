#!/bin/bash
#PBS -N dirunc_llm
#PBS -q GPU-1
#PBS -o log/dirunc_llm.o%j
#PBS -e log/dirunc_llm.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe

set -euo pipefail
export PYTHONUNBUFFERED=1

# Activate virtualenv if exists
# Activate virtualenv if exists
if [ -f ../persona_vectors/persona_steering/bin/activate ]; then
    source ../persona_vectors/persona_steering/bin/activate
fi

cd "${PBS_O_WORKDIR:-$PWD}"
mkdir -p log runs

echo "=== JOB START ==="
date
hostname
nvidia-smi || true

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Models
MODEL_GEMMA="google/gemma-2-2b-it"
MODEL_LLAMA="meta-llama/Meta-Llama-3-8B"

# Select Model
MODEL_NAME=$MODEL_GEMMA
#MODEL_NAME=$MODEL_LLAMA

# Output Dir
RUN_NAME="runs/run_gemma_2b_contrastive"
OUT_DIR="${RUN_NAME}"

echo "Model: ${MODEL_NAME}"
echo "OutDir: ${OUT_DIR}"

# Log file setup
LOG_FILE="log/dirunc_llm_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# Training
# Note: Using lower learning rate (e.g. 5e-5) for LLMs to avoid Instability/NaN.
# Using small batch size due to VRAM constraints.
python scripts/03_train_probe.py \
  --model_name "${MODEL_NAME}" \
  --data_dir data/processed/sgd/dirunc_contrastive_downsampled \
  --out_dir "${OUT_DIR}" \
  --epochs 2 \
  --batch_size 4 \
  --lr 5e-5 \
  --max_length 512 \
  --layer -1 \
  --sweep \
  --strip_query_in_baseline \
  --no_tqdm \
  --tqdm_mininterval 30 2>&1 | tee "${LOG_FILE}"

# Visualization
python scripts/06_viz_dirunc.py \
  --train_jsonl data/processed/sgd/dirunc_contrastive_downsampled/train.jsonl \
  --dev_jsonl   data/processed/sgd/dirunc_contrastive_downsampled/dev.jsonl \
  --summary_json "${OUT_DIR}/summary.json" \
  --out_dir "${OUT_DIR}/viz_main" \
  --cooc_norm rate

# Analysis (Cosine Sim & Gap)
python scripts/07_analyze_probe.py \
  --summary_json "${OUT_DIR}/summary.json" \
  --out_dir "${OUT_DIR}/analysis" \
  --dev_jsonl data/processed/sgd/dirunc_contrastive_downsampled/dev.jsonl \
  --analyze_mode best

echo "=== JOB END ==="
date
