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

def get_final_refined_answer(sample, initial_gen_col="self_refine_initial_generation",
                             feedback_prefix="self_refine_feedback_",
                             refinement_prefix="self_refine_refinement_",
                             trigger_phrase="!The Answer is correct!"):
    """
    Given a sample (a dictionary with keys for initial generation, feedback, and refinements),
    select the final answer according to the following rules:
      - Start with the initial generation.
      - For each iteration i:
          If the feedback message at iteration i contains the trigger_phrase,
          return the answer from the previous step (i-1 or the initial answer if i is 0).
          Else, if a refinement exists at iteration i, update the candidate answer and continue.
      - If no trigger phrase is found in any feedback, return the last available refinement (or the initial answer
        if no refinement exists).
    """
    # Extract initial answer (assumed stored as list with one element)
    try:
        candidate = sample.get(initial_gen_col, [""])[0]
    except Exception:
        candidate = sample.get(initial_gen_col, "")

    i = 0
    while True:
        fb_key = f"{feedback_prefix}{i}"
        ref_key = f"{refinement_prefix}{i}"

        # Retrieve feedback for iteration i
        feedback = sample.get(fb_key)
        if feedback is None:
            # No more feedback, so break the loop.
            break

        # If feedback is stored as a list, use its first element.
        if isinstance(feedback, list):
            feedback_text = feedback[0]
        else:
            feedback_text = feedback

        # Check for the trigger phrase in the feedback. If found,
        # return the candidate answer from the previous iteration.
        if trigger_phrase in feedback_text.lower():
            return candidate

        # Otherwise, check if a corresponding refinement exists.
        refinement = sample.get(ref_key)
        if refinement is None:
            # No refinement to update candidate, so stop.
            break

        # If refinement is a list, use its first element.
        if isinstance(refinement, list):
            candidate = refinement[0]
        else:
            candidate = refinement

        # Continue to next iteration.
        i += 1

    # After processing all iterations, return the candidate answer.
    return candidate


def KM_self_refine(ds, gt_col, evaluator,
                   initial_gen_col="self_refine_initial_generation",
                   feedback_prefix="self_refine_feedback_",
                   refinement_prefix="self_refine_refinement_",
                   trigger_phrase="is correct"):
    """
    Compute an average evaluation score over a dataset of samples, where the final answer is selected
    from a series of iterations stored in separate columns.

    For each sample:
      - The ground truth answer is at key gt_col.
      - The answer chain is stored as:
          initial answer at key initial_gen_col (stored as a list containing a single string)
          feedbacks at keys f"{feedback_prefix}{i}" (stored as list of one string) for each iteration i
          refinements at keys f"{refinement_prefix}{i}" (stored as list of one string) for each iteration i
      - The final answer is determined as follows:
          * If any feedback message contains the trigger_phrase, the candidate answer from the previous iteration
            is taken as final.
          * Otherwise, the final answer is the last refinement, or if no refinement exists, the initial answer.
    The evaluator is applied on the ground truth versus the final answer for each sample.
    """
    total_score = 0.0
    total_count = 0

    for sample in ds:
        ground_truth = sample[gt_col]
        # Select final answer by iterating through the stored feedback and refinements.
        final_answer = get_final_refined_answer(sample,
                                                initial_gen_col=initial_gen_col,
                                                feedback_prefix=feedback_prefix,
                                                refinement_prefix=refinement_prefix,
                                                trigger_phrase=trigger_phrase)
        total_score += evaluator(ground_truth=ground_truth, model_answer=final_answer)
        total_count += 1

    return total_score / total_count if total_count > 0 else 0.0


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
    parser = argparse.ArgumentParser(description="Run the Self-Refine Algorithm")
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
    train_data, test_data = dataset["train"], dataset["test"]


    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], cache_dir=config['cache_dir'])

    # few shots
    generation_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'self_refine_generation')
    feedback_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'self_refine_feedback')
    refine_few_shot_prompts = load_few_shot_prompts(config['few_shot_dir'], 'self_refine_refinement')

    task_type = f'self_refine_{config["task_type"]}'
    prompt_builder = get_prompt_builder(task_type)(config)
    reward_function = RewardEvaluator(config)

    # Prompt functions
    # conversation_gather_func = partial(
    #     gather_full_conversation, 
    #     question_col=config['question_col'],
    #     generation_col='self_refine_initial_generation',
    #     feedback_prefix='self_refine_feedback_',
    #     refinement_prefix='self_refine_refinement_')

    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=generation_few_shot_prompts,
    )

    feedback_prompt_func = partial(
        prompt_builder.build_feedback_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=feedback_few_shot_prompts,
    )
    refine_prompt_func = partial(
        prompt_builder.build_correction_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=refine_few_shot_prompts,
    )

    save_dir = os.path.join(config['cache_dir'], 'self_refine')
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

    train_data = perform_generation(
        data=train_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col=f"self_refine_initial_generation"
    )
    acc = KM(
        train_data, 
        target_col='self_refine_initial_generation', 
        gt_col=config['gold_col'],
        evaluator=reward_function
        )
    logger.info(f"Initial Accuracy {acc}")



    # Outer loop: for n in 1...N
    for iteration in range(config['num_refine_iterations']):
        logger.info(f"Starting feedback {iteration + 1}/{config['num_refine_iterations']}")

        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=feedback_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f"self_refine_feedback_{iteration}"  # store model's answer after rationale generation
        )

        logger.info(f"Starting refinement {iteration + 1}/{config['num_refine_iterations']}")


        train_data = perform_generation(
            data=train_data,
            model=model,
            prompt_func=refine_prompt_func,
            sampling_params=sampling_params,
            id_key=config['id_col'],
            output_col=f"self_refine_refinement_{iteration}"
        )

        acc = KM_self_refine(train_data,
        gt_col=config['gold_col'],
        evaluator=reward_function)
        logger.info(f"Refinement Accuracy {acc} at iteration {iteration + 1}/{config['num_refine_iterations']}")


    train_data.save_to_disk(run_dir)
    logger.info("Self-Refine algorithm completed.")

if __name__ == "__main__":
    main()
