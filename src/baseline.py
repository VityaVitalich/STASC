import os
from functools import partial
from pathlib import Path
from typing import Any, Callable

import datasets
import hydra
import mlflow
import mlflow.data.pandas_dataset
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer  # pyright: ignore[reportPrivateImportUsage]
from vllm import LLM, SamplingParams

from configs.config import Config
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import RewardEvaluator
from utils.flatten import flatten_dict
from utils.generation_utils import generate_for_dataset, store_generation_results
from utils.logger import setup_logger
from utils.utils import KM

config_path = str((Path(__file__).parents[1] / "config").resolve())


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config):
    """Main function to run the baseline algorithm."""
    logger = setup_logger(
        cfg.run_name,
        log_all="logs/general/general.log",
        log_info=f"logs/general/{cfg.run_name}.log",
    )

    # Mlflow setup
    load_dotenv(".env")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri is None:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env file")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("STASC")
    mlflow.log_params(flatten_dict(cfg))

    # Load dataset
    test_data = datasets.load_dataset(cfg.dataset.data_path, split="test")
    mlflow.log_input(
        mlflow.data.pandas_dataset.from_pandas(test_data.to_pandas(), name=cfg.dataset.data_path),  # type: ignore[reportArgumentType]
        context="inference",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)

    # few shots
    generation_few_shot_prompts = load_few_shot_prompts(cfg.dataset.few_shot_dir, "generation")

    task_type = f"baseline_{cfg.dataset.task_type}"
    prompt_builder = get_prompt_builder(task_type)(cfg)  # type: ignore
    reward_function = RewardEvaluator(cfg)

    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=generation_few_shot_prompts,
    )

    # Initialize model (M0)
    model = LLM(
        cfg.model.model_path,
        # download_dir=cfg.model.cache_dir  ,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        enforce_eager=cfg.model.enforce_eager,
        max_model_len=cfg.model.max_model_len,
        seed=cfg.model.random_seed,
        # disable_log_stats=True,  # Disables logging statistics
        # disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        max_tokens=cfg.model.max_tokens,
        n=1,
        seed=cfg.model.random_seed,
    )

    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=cfg.dataset.id_col,
        output_col="initial_generation",
    )
    acc = KM(
        test_data,
        target_col="initial_generation",
        gt_col=cfg.dataset.gold_col,
        evaluator=reward_function,
    )
    mlflow.log_metric("ACC", acc)
    logger.info(f"[INFO] Initial Accuracy {acc}")

    test_data.save_to_disk("logs/test_data" + cfg.run_name)
    logger.info("Baseline algorithm completed.")


def perform_generation(
    data: Any,
    model: LLM,
    prompt_func: Callable[..., str],
    sampling_params: SamplingParams,
    id_key: str,
    output_col: str,
) -> datasets.Dataset:
    """Perform (rationale) generation or (rationalization) generation for the dataset.
    Store the generation results in the dataset under 'output_col'.
    """
    generation_results = generate_for_dataset(
        model=model,
        data=data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=id_key,
    )
    return store_generation_results(data, generation_results, result_col=output_col, id_col=id_key)


if __name__ == "__main__":
    main()
