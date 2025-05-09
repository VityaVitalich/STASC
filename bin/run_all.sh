#!/bin/bash

# ---- Setup ----
export WANDB_API_KEY=''
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  # Uncomment if needed

MODEL_CONFIG='configs/model/qwen_2.5_1.5b.yaml'
DATA_CONFIG='configs/data_config/hotpot.yaml'
FT_CONFIG='configs/fine_tuning/fine_tune.yaml'
ACCELERATE_CONFIG='configs/fine_tuning/accelerate_config.yaml'

# ---- Run all STASC variants ----
SCRIPT_NAME="stasc.py"
STASC_CONFIG_DIR="configs/algo/stasc_versions"

echo "=== Running STASC variants ==="
for ALGO_CONFIG in "$STASC_CONFIG_DIR"/stasc_*.yaml; do
    echo "[RUN] $SCRIPT_NAME with $ALGO_CONFIG"

    python $SCRIPT_NAME \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --algo_config $ALGO_CONFIG \
        --ft_config $FT_CONFIG \
        --accelerate_config_path $ACCELERATE_CONFIG
done

# ---- Mapping of algorithm name to script ----
declare -A ALGO_SCRIPTS
ALGO_SCRIPTS["critic"]="stasc.py"
ALGO_SCRIPTS["self_refine"]="self_refine.py"
ALGO_SCRIPTS["baseline_rag"]="baseline.py"
ALGO_SCRIPTS["baseline_no_cot"]="baseline.py"
ALGO_SCRIPTS["baseline_cot"]="baseline.py"
ALGO_SCRIPTS["debate_finalize"]="debate.py"
ALGO_SCRIPTS["debate_common"]="debate.py"
ALGO_SCRIPTS["cove"]="cove.py"

# ---- List of non-stasc algorithms to run ----
ALGO_LIST=(
  "critic"
  "self_refine"
  "baseline_rag"
  "baseline_no_cot"
  "baseline_cot"
  "debate_finalize"
  "debate_common"
  "cove"
)

# ---- Run all non-STASC algorithms ----
echo "=== Running Baseline and Other Algorithms ==="
for ALGO_NAME in "${ALGO_LIST[@]}"; do
    ALGO_CONFIG="configs/algo/${ALGO_NAME}.yaml"
    SCRIPT_NAME=${ALGO_SCRIPTS[$ALGO_NAME]}

    echo "[RUN] $SCRIPT_NAME with $ALGO_CONFIG"

    CMD="python $SCRIPT_NAME \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --algo_config $ALGO_CONFIG"

    # Add fine-tuning config if using stasc.py
    if [[ "$SCRIPT_NAME" == "stasc.py" ]]; then
        CMD+=" --ft_config $FT_CONFIG --accelerate_config_path $ACCELERATE_CONFIG"
    fi

    eval $CMD
done
