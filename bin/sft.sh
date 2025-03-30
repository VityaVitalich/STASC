export WANDB_API_KEY=''
export ACC_CONFIG='configs/accelerate_config.yaml'
export CONFIG_PATH='configs/sft.yaml'

accelerate launch --config_file $ACC_CONFIG fine_tune.py --config_path $CONFIG_PATH

