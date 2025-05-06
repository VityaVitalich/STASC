export CONFIG_PATH='configs/critic.yaml'

python stasc.py --config $CONFIG_PATH \
    --ft_config configs/fine_tune.yaml \
    --accelerate_config_path configs/accelerate_config.yaml 
