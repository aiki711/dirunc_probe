#!/bin/bash
# ---------------------------------------------------------------------------
# Experiment 6: Leave-One-Domain-Out (LODO) & Zero-Input Bias Mitigation
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="google/gemma-2-2b-it"
DATA_JSONL="data/processed/mixed/dirunc/train.jsonl"
EVAL_DATA_JSONL="data/processed/mixed/dirunc/test.jsonl"
OUT_DIR="runs/balanced/experiment6_lodo"
LOG_DIR="${OUT_DIR}/logs"
LAYER_IDX=8 # Set to 8 based on recent experiments, but can be changed

mkdir -p "$LOG_DIR"

# Top 10 domains by sample count in train.jsonl
DOMAINS=(
  "multiwoz_hotel"
  "multiwoz_restaurant"
  "sgd_Events_2"
  "multiwoz_attraction"
  "sgd_Restaurants_1"
  "multiwoz_train"
  "sgd_Events_1"
  "sgd_Media_1"
  "sgd_RideSharing_2"
  "sgd_Banks_1"
)

echo "Starting Experiment 6 LODO Cross-Validation"
echo "Model: $MODEL"
echo "Layer: $LAYER_IDX"
echo "Domains to evaluate as hold-out: ${DOMAINS[*]}"
echo "--------------------------------------------------------"

for DOMAIN in "${DOMAINS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting LODO for hold-out domain: $DOMAIN"
    
    # 1. Train holding out the domain
    TRAIN_LOG="${LOG_DIR}/${DOMAIN}_train.log"
    echo "  -> Training... (Log: $TRAIN_LOG)"
    python3 scripts/06a_train_probe_lodo.py \
        --model_name "$MODEL" \
        --data_jsonl "$DATA_JSONL" \
        --test_domain "$DOMAIN" \
        --save_dir "$OUT_DIR" \
        --layer_idx "$LAYER_IDX" \
        --batch_size 16 \
        --num_epochs 3 \
        --neg_sample_prob 0.20 \
        > "$TRAIN_LOG" 2>&1
        
    MODEL_PATH="${OUT_DIR}/lodo_query_layer${LAYER_IDX}.pt"
    
    # 2. Evaluate on the held-out domain
    EVAL_LOG="${LOG_DIR}/${DOMAIN}_eval.log"
    EVAL_JSON="${OUT_DIR}/${DOMAIN}_eval.json"
    echo "  -> Evaluating... (Log: $EVAL_LOG)"
    python3 scripts/06b_eval_probe_lodo.py \
        --model_name "$MODEL" \
        --data_jsonl "$EVAL_DATA_JSONL" \
        --test_domain "$DOMAIN" \
        --model_path "$MODEL_PATH" \
        --layer_idx "$LAYER_IDX" \
        --batch_size 16 \
        --out_json "$EVAL_JSON" \
        > "$EVAL_LOG" 2>&1
        
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Finished domain: $DOMAIN"
    echo "--------------------------------------------------------"
done

# 3. Aggregate Results
SUMMARY_JSON="${OUT_DIR}/summary_lodo_exp6.json"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Aggregating results to $SUMMARY_JSON..."
python3 scripts/06c_aggregate_lodo.py \
    --results_dir "$OUT_DIR" \
    --out_summary "$SUMMARY_JSON"

echo "Experiment 6 completed successfully!"
