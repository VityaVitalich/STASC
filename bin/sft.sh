export WANDB_API_KEY=<>
export ACC_CONFIG='configs/hotpot/accelerate_config.yaml'
export CONFIG_PATH='configs/hotpot/sft.yaml'
export FT_CONFIG='configs/hotpot/fine_tune.yaml'

python sft_baseline.py \
    --config $CONFIG_PATH \
    --ft_config $FT_CONFIG \
    --accelerate_config_path $ACC_CONFIG

