import logging
from typing import Any

import hydra.core.hydra_config
import mlflow
import pandas as pd
from datasets import Dataset
from encourage.utils import FileManager
from vllm import LLM, SamplingParams  # pyright: ignore[reportPrivateImportUsage]

from configs.config import Config
from evaluation.eval_utils import RewardEvaluator, evaluate_responses
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.flatten import flatten_dict
from utils.generation_utils import (
    generate_responses,
)

logger = logging.getLogger(__name__)


class BaseExecutor:
    def __init__(
        self, cfg: Config, test_data: Dataset, sampling_params: SamplingParams, model: LLM
    ):
        self.cfg = cfg
        self.test_data = test_data
        self.sampling_params = sampling_params
        self.model = model
        self.prompt_builder = get_prompt_builder(cfg.algo.name)(self.cfg)

    def execute_steps(self) -> None:
        """Executes the steps of the baseline algorithm."""
        with self.start_child_run("initial_generation"):
            mlflow.log_params(flatten_dict(self.cfg))
            responses, acc = self.step_1()
            self.track_responses(responses, step_number=1)
        mlflow.log_metric("ACC", acc)

    def step_1(self) -> Any:
        ## Prompt builder
        generation_few_shot_prompts = load_few_shot_prompts(
            self.cfg.dataset.few_shot_dir, "generation"
        )
        prompts = self.prompt_builder.build_initial_generation_prompts(
            dataset=self.test_data,
            id_col=self.cfg.dataset.id_col,
            reference_col=self.cfg.dataset.gold_col,
            few_shot_prompts=generation_few_shot_prompts,
        )
        responses = self.generate_responses(prompts)

        ## Calculate accuracy
        acc = self.evaluate_responses(responses)
        return responses, acc

    def generate_responses(self, prompts: Any) -> Any:
        """Generates responses for the dataset."""
        return generate_responses(
            cfg=self.cfg,
            prompt_collection=prompts,
            model=self.model,
            sampling_params=self.sampling_params,
        )

    def start_child_run(self, run_name: str) -> mlflow.ActiveRun:
        """Starts a child MLflow run."""
        return mlflow.start_run(run_name=run_name, nested=True)

    def track_responses(self, responses: Any, step_number: int) -> None:
        """Tracks the responses in MLflow."""
        json_dump = [response.to_dict(truncated=False) for response in responses.response_data]
        FileManager(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            + f"/inference_log_{step_number}.json"
        ).dump_json(json_dump)
        json_dump = [
            flatten_dict(response.to_dict(truncated=False)) for response in responses.response_data
        ]

        try:
            active_run = mlflow.active_run()
            run_name = active_run.info.run_name if active_run else "responses"
            mlflow.log_table(data=pd.DataFrame(json_dump), artifact_file=f"{run_name}.json")
        except Exception as e:
            print(f"Failed to log table to MLflow: {e}")

    def evaluate_responses(self, responses: Any) -> float:
        """Evaluates the responses and returns the accuracy."""
        reward_function = RewardEvaluator(self.cfg)
        acc = evaluate_responses(
            responses,
            reward_function,
        )
        mlflow.log_metric("ACC", acc)
        return acc
