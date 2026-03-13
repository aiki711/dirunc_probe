#!/bin/bash

# Experiment 7: Neuron Analysis execution script
set -euo pipefail

# 1. プロジェクトルートへ移動
cd "$(dirname "$0")/../.."
echo "Current directory: $(pwd)"

# 2. 仮想環境の有効化
echo "Activating virtual environment..."
source dirunc_probe/bin/activate

mkdir -p runs/balanced/experiment7_neurons

echo "=== Experiment 7 Start ==="
date

# 7a. Analyze probe weights to find consistent neurons
echo "Running 07a_analyze_probe_weights.py..."
python scripts/07a_analyze_probe_weights.py \
    --model_dir runs/balanced/experiment6_single_token \
    --out_json runs/balanced/experiment7_neurons/neurons_report.json

# 7b. Create shift pairs (A/B sentences) for verification
echo "Running 07b_create_shift_pairs.py..."
python scripts/07b_create_shift_pairs.py

# 7c. Verify neuron shift (Statistical test)
echo "Running 07c_verify_neuron_shift.py..."
python scripts/07c_verify_neuron_shift.py

echo "=== Experiment 7 End ==="
date
