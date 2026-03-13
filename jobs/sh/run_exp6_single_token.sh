# 1. プロジェクトルートへ移動（jobs/sh から実行される想定）
cd "$(dirname "$0")/../.."
echo "Current directory: $(pwd)"

# 2. 仮想環境の有効化
echo "Activating virtual environment..."
source dirunc_probe/bin/activate

# ディレクトリの作成
mkdir -p log runs/balanced/experiment6_single_token

echo "=== JOB START (SINGLE TOKEN) ==="
date
hostname
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi || true
fi

# 2. 環境設定
export PYTHONPATH=${PYTHONPATH:-}:.
if [ -f .hf_token ]; then
    export HF_TOKEN=$(cat .hf_token)
fi

# Pythonバージョンの確認
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 実験パラメータ
DATA_JSONL="data/processed/mixed/dirunc/all.jsonl"
OUT_DIR="runs/balanced/experiment6_single_token"
MODEL="google/gemma-2-2b-it"
LAYER_IDX=8

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

echo "Starting Experiment 6 LODO (Single Tokenization)"
echo "Model: $MODEL"
echo "Layer: $LAYER_IDX"
echo "--------------------------------------------------------"

# 3. メインループ
for DOMAIN in "${DOMAINS[@]}"; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Hold-out domain: $DOMAIN"
    EVAL_JSON="${OUT_DIR}/${DOMAIN}_eval.json"
    MODEL_PATH="${OUT_DIR}/lodo_query_layer${LAYER_IDX}_${DOMAIN}.pt"
    
    if [ -f "$EVAL_JSON" ]; then
        echo "  -> Skip: $EVAL_JSON already exists."
        continue
    fi

    # a. Train holding out the domain
    if [ -f "$MODEL_PATH" ]; then
        echo "  -> Skip: $MODEL_PATH already exists."
    else
        echo "  -> Training..."
        python scripts/06a_train_probe_lodo.py \
            --model_name "$MODEL" \
            --data_jsonl "$DATA_JSONL" \
            --test_domain "$DOMAIN" \
            --save_dir "$OUT_DIR" \
            --layer_idx "$LAYER_IDX" \
            --batch_size 16 \
            --num_epochs 3 \
            --neg_sample_prob 0.20 2>&1 | tee -a "$OUT_DIR/realtime.log"
    fi
        
    # b. Evaluate on the held-out domain
    echo "  -> Evaluating..."
    python scripts/06b_eval_probe_lodo.py \
        --model_name "$MODEL" \
        --data_jsonl "$DATA_JSONL" \
        --test_domain "$DOMAIN" \
        --model_path "$MODEL_PATH" \
        --layer_idx "$LAYER_IDX" \
        --batch_size 16 \
        --out_json "$EVAL_JSON"
        
    echo "--------------------------------------------------------"
done

# 4. Aggregate Results
SUMMARY_JSON="${OUT_DIR}/summary_lodo_exp6.json"
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Aggregating results to $SUMMARY_JSON..."
python scripts/06c_aggregate_lodo.py \
    --results_dir "$OUT_DIR" \
    --out_summary "$SUMMARY_JSON"

echo "=== JOB END ==="
date
