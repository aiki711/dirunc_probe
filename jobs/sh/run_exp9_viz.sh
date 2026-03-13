#!/bin/bash

# Experiment 9: Evolution Visualization
set -euo pipefail

# 1. プロジェクトルートへ移動
cd "$(dirname "$0")/../.."
echo "Current directory: $(pwd)"

# 2. 仮想環境の有効化
echo "Activating virtual environment..."
source dirunc_probe/bin/activate

mkdir -p runs/balanced/experiment9_sentence runs/balanced/experiment9_word

echo "=== Experiment 9 Start ==="
date

# 9a. Sentence evolution extraction
echo "Running 09a_sentence_evolution.py..."
python scripts/09a_sentence_evolution.py

# 9b. Plot sentence evolution
echo "Running 09b_plot_sentence_evolution.py..."
python scripts/09b_plot_sentence_evolution.py

# 9c. Word evolution extraction
echo "Running 09c_word_evolution.py..."
python scripts/09c_word_evolution.py

# 9d. Plot word evolution
echo "Running 09d_plot_word_evolution.py..."
python scripts/09d_plot_word_evolution.py

echo "=== Experiment 9 End ==="
date
