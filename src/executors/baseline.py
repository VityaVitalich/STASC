import logging
from typing import Any

import hydra.core.hydra_config
import mlflow
import pandas as pd
from datasets import Dataset
from encourage.utils import FileManager
from vllm import LLM, SamplingParams  # pyright: ignore[reportPrivateImportUsage]

from config import Config
from evaluation.eval_utils import RewardEvaluator, evaluate_responses
from executors.factory import ExecutorRegistry
from helper.generation import generate_responses, init_model
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.flatten import flatten_dict

logger = logging.getLogger(__name__)


@ExecutorRegistry.register("baseline_cot")
@ExecutorRegistry.register("baseline_no_cot")
class BaseExecutor:
    def __init__(
        self,
        cfg: Config,
        test_data: Dataset,
        train_data: Any,
        sampling_params: SamplingParams,
    ):
        self.cfg = cfg
        self.test_data = test_data
        self.train_data = train_data  # It is not used in the Baseline!
        self.sampling_params = sampling_params
        self.prompt_builder = get_prompt_builder(cfg.algo.name)(self.cfg)
        self.root_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    def execute_steps(self) -> None:
        """Executes the steps of the baseline algorithm."""
        with self.start_child_run("initial_generation"):
            mlflow.log_params(flatten_dict(self.cfg))
            model = init_model(self.cfg)
            responses = self.step_1(self.test_data, model)
            self.track_responses(responses, step_number=1)

    def step_1(self, dataset: Dataset, model: LLM) -> Any:
        ## Prompt builder
        generation_few_shot_prompts = load_few_shot_prompts(
            self.cfg.dataset.few_shot_dir, "generation"
        )
        prompts = self.prompt_builder.build_initial_generation_prompts(
            dataset=dataset,
            id_col=self.cfg.dataset.id_col,
            reference_col=self.cfg.dataset.gold_col,
            few_shot_prompts=generation_few_shot_prompts,
        )
        responses = generate_responses(self.cfg, prompts, model, self.sampling_params)

        ## Calculate accuracy
        self.evaluate_responses(responses, "test")
        return responses

    def start_child_run(self, run_name: str) -> mlflow.ActiveRun:
        """Starts a child MLflow run."""
        return mlflow.start_run(run_name=run_name, nested=True)

    def track_responses(self, responses: Any, step_number: int | str) -> None:
        """Tracks the responses in MLflow."""
        json_dump = [response.to_dict(truncated=False) for response in responses.response_data]
        FileManager(self.root_folder + f"/inference_log_{step_number}.json").dump_json(json_dump)
        json_dump = [
            flatten_dict(response.to_dict(truncated=False)) for response in responses.response_data
        ]

        try:
            mlflow.log_table(
                data=pd.DataFrame(json_dump), artifact_file=f"inference_log_{step_number}.json"
            )
        except Exception as e:
            print(f"Failed to log table to MLflow: {e}")

    def evaluate_responses(self, responses: Any, split: str = "test") -> float:
        """Evaluates the responses and returns the accuracy."""
        reward_function = RewardEvaluator(self.cfg)
        acc = evaluate_responses(
            responses,
            reward_function,
        )
        mlflow.log_metric(f"{split}_ACC", acc)
        return acc
