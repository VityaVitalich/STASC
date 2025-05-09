#!/bin/bash

# ---- Setup ----
export WANDB_API_KEY=''
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  # Uncomment if needed

MODEL_CONFIG='configs/model/qwen_2.5_1.5b.yaml'
DATA_CONFIG='configs/data_config/hotpot.yaml'
FT_CONFIG='configs/fine_tuning/fine_tune.yaml'
ACCELERATE_CONFIG='configs/fine_tuning/accelerate_config.yaml'

SCRIPT_NAME="stasc.py"
CONFIG_DIR="configs/algo/stasc_versions"

# ---- Execution loop over all stasc_*.yaml files ----
for ALGO_CONFIG in "$CONFIG_DIR"/stasc_*.yaml; do
    echo "Running $SCRIPT_NAME with $ALGO_CONFIG"

    CMD="python $SCRIPT_NAME \
        --data_config $DATA_CONFIG \
        --model_config $MODEL_CONFIG \
        --algo_config $ALGO_CONFIG \
        --ft_config $FT_CONFIG \
        --accelerate_config_path $ACCELERATE_CONFIG"

    eval $CMD
done
