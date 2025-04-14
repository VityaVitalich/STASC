import sys
import os


sys.path.append('../')
sys.path.append('./')

import argparse
from pathlib import Path
import datasets
from datasets import Dataset, DatasetDict
from functools import partial
from utils.generation_utils import generate_for_dataset, store_generation_results, load_config
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import RewardEvaluator
from vllm import LLM, SamplingParams
from utils.utils import KM

from transformers import AutoTokenizer
from prompts import get_prompt_builder
import os
import yaml
import subprocess
import threading
import torch
import logging



def setup_logger(run_name: str, log_file="star.log"):
    """
    Sets up a logger named "star_logger_{run_name}" that writes both 
    to the console and to `log_file`.
    """
    logger_name = f"self_refine_logger_{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        # 1) Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        # 2) File handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

def perform_generation(data, model, prompt_func, sampling_params, id_key, output_col):
    """
    Perform (rationale) generation or (rationalization) generation for the dataset.
    Store the generation results in the dataset under 'output_col'.
    """
    generation_results = generate_for_dataset(
        model=model,
        data=data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=id_key
    )
    return store_generation_results(data, generation_results, result_col=output_col, id_col=id_key)



def main():
    parser = argparse.ArgumentParser(description="Run the baseline generation")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    logger = setup_logger(config['run_name'], log_file=f"logs/detailed/{config['run_name']}.log")

    # Load dataset
    dataset = datasets.load_from_disk(str(config['data_path']))
    _, test_data = dataset["train"], dataset["test"]


    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], cache_dir=config['cache_dir'])

    # few shots
    generation_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'generation')

    task_type = f'baseline_{config["task_type"]}'
    prompt_builder = get_prompt_builder(task_type)(config)
    reward_function = RewardEvaluator(config)


    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=generation_few_shot_prompts,
    )

    save_dir = os.path.join(config['cache_dir'], 'baseline')
    os.makedirs(save_dir, exist_ok=True)
    run_dir = os.path.join(save_dir, config['run_name'])

    # Initialize model (M0)
    model = LLM(
        config['model_path'],
        download_dir=config['cache_dir'],
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config['gpu_memory_utilization'],
        enforce_eager=config['enforce_eager'],
        max_model_len=config['max_model_len'],
        seed=config['random_seed']
        # disable_log_stats=True,  # Disables logging statistics
        #disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_tokens=config['max_tokens'],
        n=1,
        seed=config['random_seed']
    )

    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col=f"initial_generation"
    )
    acc = KM(
        test_data, 
        target_col='initial_generation', 
        gt_col=config['gold_col'],
        evaluator=reward_function
        )
    logger.info(f"Initial Accuracy {acc}")

    test_data.save_to_disk(run_dir)
    logger.info("Baseline algorithm completed.")

if __name__ == "__main__":
    main()
