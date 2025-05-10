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
from generator_src.stasc_vllm_generation import collect_correction_stats

from utils.utils import construct_run_name

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
    parser = argparse.ArgumentParser(description="Run the Debate Algorithm")
    parser.add_argument("--model_config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--algo_config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # Load configuration
    algo_config = load_config(args.algo_config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)

    run_name = construct_run_name(
    args.model_config,
    args.data_config,
    args.algo_config,
    algo_config['run_name_specification']
    )
    algo_config['run_name'] = run_name

    config = {**algo_config, **model_config, **data_config}

    logger = setup_logger(config['run_name'], log_file=f"logs/general/{config['run_name']}.log")

    # Load dataset
    dataset = datasets.load_from_disk(str(config['data_path']))
    _, test_data = dataset["train"], dataset["test"]


    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], cache_dir=config['cache_dir'])

    # few shots
    initial_generation_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'debate_generation')
    correction_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'debate_correction')
    judge_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'debate_judge')

    task_type = f'debate_{config["task_type"]}'
    prompt_builder = get_prompt_builder(task_type)(config)
    reward_function = RewardEvaluator(config)


    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=initial_generation_few_shot_prompts,
    )
    correction_prompt_func = partial(
        prompt_builder.build_correction_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=correction_few_shot_prompts,
    )
    judge_prompt_func = partial(
        prompt_builder.build_judge_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=judge_few_shot_prompts,
    )

    save_dir = os.path.join(config['cache_dir'], 'debate')
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

    init_acc = KM(
        test_data, 
        target_col='initial_generation', 
        gt_col=config['gold_col'],
        evaluator=reward_function
    )
    logger.info(f"[INFO] Initial Accuracy {init_acc}")
    
    sampling_params.n = config['number_output_corrections']
    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=correction_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col=f"correction"
    )
    sampling_params.n = 1
    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=judge_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col=f"judgement"
    )


    revised_acc = KM(
        test_data, 
        target_col='judgement', 
        gt_col=config['gold_col'],
        evaluator=reward_function
    )
    logger.info(f"[INFO] Correction Judge Accuracy {revised_acc}")

    stats_test = collect_correction_stats(
        dataset=test_data,
        question_col=config['question_col'],
        reference_col=config['gold_col'],
        inital_answer_col='initial_generation',
        correction_col=f'judgement',
        reward_function=reward_function
    )
    logger.info(
        f"[INFO] Test Correction Statistics:\n"
        f"[INFO]       - Correct → Incorrect: {stats_test['correct_to_incorrect']:.2f}%\n"
        f"[INFO]       - Correct → Correct: {stats_test['correct_to_correct']:.2f}%\n"
        f"[INFO]       - Incorrect → Correct: {stats_test['incorrect_to_correct']:.2f}%"
    )

    test_data.save_to_disk(run_dir)
    logger.info("Debate algorithm completed.")

if __name__ == "__main__":
    main()
