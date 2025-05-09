export WANDB_API_KEY=''
# need for qwen models, having pos embedders < max model len
# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

export ALGO_CONFIG='configs/algo/sft.yaml'
export MODEL_CONFIG='configs/model/qwen_2.5_1.5b.yaml'
export DATA_CONFIG='configs/data_config/hotpot.yaml'

python sft_baseline.py \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --algo_config $ALGO_CONFIG \
    --ft_config configs/fine_tuning/fine_tune.yaml \
    --accelerate_config_path configs/fine_tuning/accelerate_config.yaml 


