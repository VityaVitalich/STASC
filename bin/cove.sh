export ALGO_CONFIG='configs/algo/cove.yaml'
export MODEL_CONFIG='configs/model/qwen_2.5_1.5b.yaml'
export DATA_CONFIG='configs/data_config/hotpot.yaml'

python cove.py \
    --data_config $DATA_CONFIG \
    --model_config $MODEL_CONFIG \
    --algo_config $ALGO_CONFIG