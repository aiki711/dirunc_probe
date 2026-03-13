#!/bin/bash

# Experiment 8: Causal Analysis (Ablation and Patching)
set -euo pipefail

# 1. プロジェクトルートへ移動
cd "$(dirname "$0")/../.."
echo "Current directory: $(pwd)"

# 2. 仮想環境の有効化
echo "Activating virtual environment..."
source dirunc_probe/bin/activate

mkdir -p runs/balanced/experiment8_causal

echo "=== Experiment 8 Start ==="
date

# 8a. Ablation test (Knock-out)
echo "Running 08a_ablation_test.py..."
python scripts/08a_ablation_test.py

# 8b. Patching test (Intervention)
echo "Running 08b_patching_test.py..."
python scripts/08b_patching_test.py

# 8c. Mechanism analysis (How it works)
echo "Running 08c_how_mechanism.py..."
python scripts/08c_how_mechanism.py

echo "=== Experiment 8 End ==="
date
