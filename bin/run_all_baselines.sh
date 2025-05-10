#!/bin/sh

# ---- Setup ----
export WANDB_API_KEY=''
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1  # Uncomment if needed

MODEL_CONFIG='configs/model/qwen_2.5_1.5b.yaml'
DATA_CONFIG='configs/data_config/hotpot.yaml'
FT_CONFIG='configs/fine_tuning/fine_tune.yaml'
ACCELERATE_CONFIG='configs/fine_tuning/accelerate_config.yaml'

# ---- List of algorithms to run ----
ALGO_LIST="critic self_refine baseline_rag baseline_no_cot baseline_cot debate_finalize debate_common cove sft"

# ---- Script mapping function ----
get_script_name() {
  case "$1" in
    critic) echo "stasc.py" ;;
    self_refine) echo "self_refine.py" ;;
    baseline_rag|baseline_no_cot|baseline_cot) echo "baseline.py" ;;
    debate_finalize|debate_common) echo "debate.py" ;;
    cove) echo "cove.py" ;;
    sft) echo "sft_baseline.py" ;;
    *) echo "unknown" ;;
  esac
}

# ---- Run all non-STASC algorithms ----
echo "=== Running Baseline and Other Algorithms ==="
for ALGO_NAME in $ALGO_LIST; do
  ALGO_CONFIG="configs/algo/${ALGO_NAME}.yaml"
  SCRIPT_NAME=$(get_script_name "$ALGO_NAME")

  if [ "$SCRIPT_NAME" = "unknown" ]; then
    echo "[WARN] Unknown script for $ALGO_NAME, skipping"
    continue
  fi

  echo "[RUN] $SCRIPT_NAME with $ALGO_CONFIG"

  CMD="python $SCRIPT_NAME \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --algo_config $ALGO_CONFIG"

    case "$SCRIPT_NAME" in
    sft_baseline.py|stasc.py)
        CMD="$CMD --ft_config $FT_CONFIG --accelerate_config_path $ACCELERATE_CONFIG"
        ;;
    esac


  sh -c "$CMD"
done
