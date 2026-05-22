#!/bin/bash
# jobs/sh/run_gemini_train_both.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Starting training for Gemini Soft Data..."
OUT_DIR="runs/cg_probe_gemini_soft" \
TRAIN_DATA="data/processed/case_grammar/paired_train_gemini_soft.jsonl" \
DEV_DATA="data/processed/case_grammar/paired_dev_gemini_soft.jsonl" \
LAYER=16 \
bash jobs/sh/02_train_cg_probe.sh

echo "--------------------------------------------------"
echo "Starting training for Gemini Strong Data..."
OUT_DIR="runs/cg_probe_gemini_strong" \
TRAIN_DATA="data/processed/case_grammar/paired_train_gemini_strong.jsonl" \
DEV_DATA="data/processed/case_grammar/paired_dev_gemini_strong.jsonl" \
LAYER=16 \
bash jobs/sh/02_train_cg_probe.sh

echo "All training completed!"
