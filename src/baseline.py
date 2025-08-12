import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable

import datasets
import hydra
import litellm._logging
import mlflow
import mlflow.data.pandas_dataset
import pandas as pd
from dotenv import load_dotenv
from encourage.llm import ResponseWrapper
from encourage.utils import FileManager
from transformers import AutoTokenizer  # pyright: ignore[reportPrivateImportUsage]
from vllm import SamplingParams

from configs.config import Config
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import RewardEvaluator
from utils.flatten import flatten_dict
from utils.generation_utils import generate_for_dataset
from utils.utils import KM

config_path = str((Path(__file__).parents[1] / "config").resolve())

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config):
    """Main function to run the baseline algorithm."""
    litellm._logging._disable_debugging()
    # mlflow_openai.autolog()
    logger.info(f"[INFO] Running baseline algorithm with config: {cfg}")

    # Mlflow setup
    load_dotenv(".env")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri is None:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env file")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("STASC")
    mlflow.log_params(flatten_dict(cfg))

    # Load dataset
    test_data = datasets.load_dataset(cfg.dataset.data_path, split="test[:100]")
    mlflow.log_input(
        mlflow.data.pandas_dataset.from_pandas(test_data.to_pandas(), name=cfg.dataset.data_path),  # type: ignore[reportArgumentType]
        context="inference",
    )

    # few shots
    generation_few_shot_prompts = load_few_shot_prompts(cfg.dataset.few_shot_dir, "generation")

    task_type = f"baseline_{cfg.dataset.task_type}"
    prompt_builder = get_prompt_builder(task_type)(cfg)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)

    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=generation_few_shot_prompts,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=cfg.model.temperature,
        top_p=cfg.model.top_p,
        max_tokens=cfg.model.max_tokens,
        n=1,
        seed=cfg.model.random_seed,
    )

    responses = perform_generation(
        data=test_data,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=cfg.dataset.id_col,
        output_col="initial_generation",
    )

    json_dump = [response.to_dict(truncated=False) for response in responses.response_data]
    FileManager(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir + "/inference_log.json"
    ).dump_json(json_dump)
    json_dump = [
        flatten_dict(response.to_dict(truncated=False)) for response in responses.response_data
    ]

    try:
        active_run = mlflow.active_run()
        mlflow.log_table(
            data=pd.DataFrame(json_dump), artifact_file=f"{active_run.info.run_name}.json"
        )
    except Exception as e:
        print(f"Failed to log table to MLflow: {e}")

    reward_function = RewardEvaluator(cfg)
    acc = KM(
        responses,
        evaluator=reward_function,
    )
    mlflow.log_metric("ACC", acc)
    logger.info(f"[INFO] Initial Accuracy {acc}")

    # test_data.save_to_disk("logs/test_data" + cfg.run_name)
    logger.info("Baseline algorithm completed.")


def perform_generation(
    data: Any,
    prompt_func: Callable[..., str],
    sampling_params: SamplingParams,
    id_key: str,
    output_col: str,
) -> ResponseWrapper:
    """Perform (rationale) generation or (rationalization) generation for the dataset.
    Store the generation results in the dataset under 'output_col'.
    """
    generation_results = generate_for_dataset(
        data=data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=id_key,
    )
    return generation_results
    return store_generation_results(data, generation_results, result_col=output_col, id_col=id_key)


if __name__ == "__main__":
    main()
