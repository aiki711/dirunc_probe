#!/bin/bash

# Gemma-2-2b-it Case Grammar Layer Sweep
# Sampling layers: 0, 4, 8, 12, 16, 20, 24, 26

LAYERS=(16 20 24 26)
MODEL="google/gemma-2-2b-it"
TRAIN_DATA="data/processed/case_grammar/paired_train_gemini.jsonl"
DEV_DATA="data/processed/case_grammar/paired_dev_gemini.jsonl"
OUT_BASE="runs/layer_sweep_gemma2"

mkdir -p $OUT_BASE

for L in "${LAYERS[@]}"; do
    echo "===================================================="
    echo "Processing Layer: $L"
    echo "===================================================="
    
    python scripts/32_train_contrastive_probe.py \
        --model_name "$MODEL" \
        --train_data "$TRAIN_DATA" \
        --dev_data "$DEV_DATA" \
        --layer_idx $L \
        --out_dir "$OUT_BASE/layer_$L" \
        --epochs 5 \
        --batch_size 16 \
        --lr 5e-4
        
    if [ $? -ne 0 ]; then
        echo "Error at layer $L. Exiting."
        exit 1
    fi
done

echo "All layers processed successfully."
