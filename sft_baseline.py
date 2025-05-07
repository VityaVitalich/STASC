import argparse
from pathlib import Path
from utils.generation_utils import load_config
from utils.logger import setup_logger, run_subprocess_in_real_time
import os
import yaml
import subprocess
import logging
from functools import partial


system_prompt_init = (
    "You are a helpful reasoning assistant in general domain question answering. "
    "Please reason through the question step by step very shortly before giving a final answer."
)
initial_instructions = (
    "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
    "Step-by-step reasoning:\n"
    "Final Answer:\n"
)

def call_fine_tune(config_yaml_path: str, accelerate_config_path: str, logger: logging.Logger):
    """
    Calls fine_tune.py with the specified YAML config, logging output in real time.
    """
    cmd = [
        "accelerate", "launch",
        "--config_file", accelerate_config_path,
        "fine_tune.py",
        "--config_path", config_yaml_path
    ]
    returncode = run_subprocess_in_real_time(cmd, logger)

    if returncode != 0:
        raise RuntimeError(f"fine_tune.py failed with exit code {returncode}")

def call_vllm_generation(config_path, generation_model_path, ft_dataset_path, iteration, initial_generation, logger: logging.Logger):
    """
    Spawns a new Python process (stasc_vllm_generation.py) in real time.
    """
    cmd = [
        "python", "generator_src/stasc_vllm_generation.py",
        "--config_path", config_path,
        "--generation_model_path", generation_model_path,
        "--ft_dataset_path", ft_dataset_path,
        "--iteration", iteration,
    ]

    if initial_generation:
        cmd.append("--initial_generation")
    returncode = run_subprocess_in_real_time(cmd, logger)

    if returncode != 0:
        raise RuntimeError(f"stasc_vllm_generation.py failed with exit code {returncode}")

def ensure_directories(config):
    """Ensures all necessary directories exist."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/detailed", exist_ok=True)
    os.makedirs("logs/general", exist_ok=True)
    os.makedirs("configs/temp", exist_ok=True)
    stasc_dir = os.path.join(config['cache_dir'], 'sft_baseline')
    os.makedirs(stasc_dir, exist_ok=True)
    run_dir = os.path.join(stasc_dir, config['run_name'])
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_temporary_configs(config, ft_config, run_name):
    """Saves temporary config files."""
    temp_config_path = f"configs/temp/temp_config_{run_name}.yaml"
    temp_ft_config_path = f"configs/temp/temp_ft_config_{run_name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    with open(temp_ft_config_path, "w") as f:
        yaml.dump(ft_config, f, sort_keys=False)
    return temp_config_path, temp_ft_config_path

def generate_initial_answers(config, temp_config_path, generation_model_path, run_dir, logger):
    """Generates initial answers if needed."""
    if not config['initial_answer_with_new_model']:
        initial_ans_dataset_path = os.path.join(run_dir, "initial_data")

        call_vllm_generation(
            config_path=temp_config_path, 
            generation_model_path=generation_model_path,
            ft_dataset_path=initial_ans_dataset_path,
            iteration=str(0), 
            initial_generation=True,
            logger=logger
        )

        config['data_path'] = initial_ans_dataset_path
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f, sort_keys=False)

def add_sft_messages(sample, config):

    # Build user question
    question_text = sample[config['question_col']]
    user_question = (
        f"Question:\n{question_text}\n\n"
            "Strictly follow format Final Answer:"
    )

    gold = sample[config['gold_col']][0]
    answer = f"Final Answer: {gold}"

    # Compose messages
    messages = compose_chat_messages(
        system_prompt=system_prompt_init,
        instructions=initial_instructions,
        user_question=user_question,
    )

    messages = [
        {"role": "system", "content": system_prompt_init},
        {"role": "user", "content": initial_instructions + user_question},
        {"role": "assistant", "content": answer}
    ]
    sample['messages'] = messages
    return sample




def main():
    parser = argparse.ArgumentParser(description="Run the SFT Baseline")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--ft_config", type=str, required=True, help="Path to the Fine-Tuning YAML config file.")
    parser.add_argument("--accelerate_config_path", type=str, required=True, help="Path to the accelerate YAML config file.")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)
    ft_config = load_config(args.ft_config)

    run_dir = ensure_directories(config)
    temp_config_path, temp_ft_config_path = save_temporary_configs(config, ft_config, config['run_name'])   
    ft_dataset_path = os.path.join(run_dir, 'data')
    generation_model_path = os.path.join(run_dir, 'model')


    logger = setup_logger(
        config['run_name'], 
        log_all=f"logs/detailed/{config['run_name']}.log",
        log_info=f"logs/general/{config['run_name']}.log"
    )
    logger.info("\n" + yaml.dump(config, sort_keys=False, default_flow_style=False))
    logger.info("\n" + yaml.dump(ft_config, sort_keys=False, default_flow_style=False))

    dataset = datasets.load_from_disk(str(config['data_path']))
    train_data, test_data = dataset["train"], dataset["test"]

    train_data = train_data.map(partial(add_sft_messages, config=config))
    train_data.save_to_disk(ft_dataset_path)

    ft_config['model']['model_name_or_path'] = config['model_path']
    ft_config["data"]["dataset_name"] = ft_dataset_path
    ft_config["training"]["output_dir"] = generation_model_path
    ft_config['training']['run_name'] = f"{config['run_name']}"

    with open(temp_ft_config_path, "w") as f:
        yaml.dump(ft_config, f, sort_keys=False)

    # (7) Fine-tune on the combined correct solutions
    call_fine_tune(
        config_yaml_path=temp_ft_config_path,
        accelerate_config_path=args.accelerate_config_path,
        logger=logger
    )

    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], cache_dir=config['cache_dir'])

    # few shots

    prompt_builder = get_prompt_builder(config['task_type'])(config)
    reward_function = RewardEvaluator(config)

    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        context_col=config['context_col']
    )

    correction_prompt_func = partial(
        prompt_builder.build_correction_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        initial_answer_col='initial_generation',
    )

    # Initialize model (M0)
    model = LLM(
        generation_model_path,
        download_dir=config['cache_dir'],
        dtype=torch.bfloat16,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config['gpu_memory_utilization'],
        enforce_eager=config['enforce_eager'],
        max_model_len=config['max_model_len'],
        disable_log_stats=True,  # Disables logging statistics
        seed=config['random_seed']
        #disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=config['temperature'],
        top_p=config['top_p'],
        max_tokens=config['max_tokens'],
        n=1,
        seed=config['random_seed'],
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

    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=correction_prompt_func,
        sampling_params=sampling_params,
        id_key=config['id_col'],
        output_col=f"correction"
    )

    revised_acc = KM(
        test_data, 
        target_col='correction', 
        gt_col=config['gold_col'],
        evaluator=reward_function
    )
    logger.info(f"Correction Accuracy {revised_acc}")

    test_data.save_to_disk(run_dir)
    logger.info("SFT Baseline completed.")


if __name__ == "__main__":
    main()
