import argparse
from pathlib import Path
from utils.generation_utils import load_config
from utils.logger import setup_logger, run_subprocess_in_real_time
from utils.utils import construct_run_name
import os
import yaml
import subprocess
import logging


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
    stasc_dir = os.path.join(config['cache_dir'], 'STaSC')
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




def main():
    parser = argparse.ArgumentParser(description="Run the STaSC Algorithm")
    parser.add_argument("--model_config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--data_config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--algo_config", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument("--ft_config", type=str, required=True, help="Path to the Fine-Tuning YAML config file.")
    parser.add_argument("--accelerate_config_path", type=str, required=True, help="Path to the accelerate YAML config file.")
    args = parser.parse_args()

    # Load configuration
    algo_config = load_config(args.algo_config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    ft_config = load_config(args.ft_config)

    run_name = construct_run_name(
    args.model_config,
    args.data_config,
    args.algo_config,
    algo_config['run_name_specification']
    )
    algo_config['run_name'] = run_name

    config = {**algo_config, **model_config, **data_config}

    run_dir = ensure_directories(config)
    temp_config_path, temp_ft_config_path = save_temporary_configs(config, ft_config, config['run_name'])   

    logger = setup_logger(
        config['run_name'], 
        log_all=f"logs/detailed/{config['run_name']}.log",
        log_info=f"logs/general/{config['run_name']}.log"
    )

    logger.info("\n" + yaml.dump(config, sort_keys=False, default_flow_style=False))
    logger.info("\n" + yaml.dump(ft_config, sort_keys=False, default_flow_style=False))

    # Create iteration directories (iter_0, iter_1, ...)
    iteration_dirs = [os.path.join(run_dir, f"iter_{i}") for i in range(config['num_star_iterations'] + 1)]
    for iter_dir in iteration_dirs:
        os.makedirs(iter_dir, exist_ok=True)  # Ensure iteration directories exist

    # start from initial model
    generation_model_path = config['model_path']
    ft_config['model']['model_name_or_path'] = generation_model_path

    generate_initial_answers(config, temp_config_path, generation_model_path, run_dir, logger)


    # Outer loop: for n in 1...N
    for iteration in range(config['num_star_iterations']+1):
        logger.info(f"[INFO] Starting iteration {iteration}/{config['num_star_iterations']}")


        iter_dir = os.path.join(run_dir, f"iter_{iteration}")  # Get current iteration dir
        os.makedirs(iter_dir, exist_ok=True)
        ft_dataset_path = os.path.join(iter_dir, 'data')

        call_vllm_generation(
                    config_path=temp_config_path,
                    generation_model_path=generation_model_path,
                    ft_dataset_path=ft_dataset_path,
                    iteration=str(iteration),
                    initial_generation=False,
                    logger=logger
                )

        if iteration == config['num_star_iterations']:
            break


        # name for future model and modified configs
        generation_model_path = os.path.join(iter_dir, 'model')
        ft_config["data"]["dataset_name"] = ft_dataset_path
        ft_config["training"]["output_dir"] = generation_model_path
        ft_config['training']['run_name'] = f"{config['run_name']}_{iteration}"

        with open(temp_ft_config_path, "w") as f:
            yaml.dump(ft_config, f, sort_keys=False)

        # (7) Fine-tune on the combined correct solutions
        call_fine_tune(
            config_yaml_path=temp_ft_config_path,
            accelerate_config_path=args.accelerate_config_path,
            logger=logger
        )

        if not config['train_from_initial_model']:
            ft_config['model']['model_name_or_path'] = generation_model_path

        # End of iteration; M_{n} is now your updated model

    logger.info("[INFO] STaSC algorithm completed.")

if __name__ == "__main__":
    main()
