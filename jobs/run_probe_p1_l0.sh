#!/bin/bash
#PBS -N dirunc_p1_l0
#PBS -q GPU-1
#PBS -o log/dirunc_p1_l0.o%j
#PBS -e log/dirunc_p1_l0.e%j
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=08:00:00
#PBS -j oe

set -euo pipefail

cd "${PBS_O_WORKDIR:-$PWD}"
mkdir -p log runs

echo "=== JOB START ==="
date
hostname
nvidia-smi || true

# ---- (optional) if CUDA torch is not installed, this won't use GPU ----
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

#python scripts/02_make_dirunc_dataset.py \
#  --splits train,dev \
#  --max_dialogues_per_split 0 \
#  --debug_dir_stats

python scripts/03_train_probe.py \
  --model_name distilroberta-base \
  --phases 1 --levels 0 \
  --mode both \
  --batch_size 16 --epochs 2 --max_length 512 \
  --strip_query_in_baseline \
  --refine_pm2 \
  --tqdm_mininterval 10 \
  --out_dir runs/dirunc_p1_l0 \
  --no_tqdm \
  --eval_services Flights_3,RideSharing_1

#python scripts/04_stats_dirunc.py \
#  --data_dir data/processed/sgd/dirunc \
#  --slot_meta data/processed/sgd/slot_meta_by_service_slot.json \
#  --splits train,dev \
#  --out_dir runs/dirunc_stats \
#  --examples_per_slot 2 \
#  --text_max_len 180 \
#  --topk_services 30 \
#  --topk_service_intents 50

#python scripts/05_viz_generated.py \
#  --data_dir data/processed/sgd/dirunc \
#  --train_file train.jsonl \
#  --dev_file dev.jsonl \
#  --out_dir runs/dirunc_viz_plus \
#  --top_k_services 30 \
#  --probe_summary runs/dirunc_p1_l0/summary.json

python scripts/06_viz_dirunc.py \
  --train_jsonl data/processed/sgd/dirunc/train.jsonl \
  --dev_jsonl   data/processed/sgd/dirunc/dev.jsonl \
  --summary_json runs/dirunc_p1_l0/summary.json \
  --out_dir runs/dirunc_viz_main \
  --phases 1 --levels 0 \
  --cooc_norm rate

echo "=== JOB END ==="
date
