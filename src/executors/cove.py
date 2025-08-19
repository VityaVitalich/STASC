import logging
from typing import Any, override

import mlflow
from datasets import Dataset
from encourage.llm import ResponseWrapper
from vllm import LLM, SamplingParams  # pyright: ignore[reportPrivateImportUsage]

from configs.config import Config
from evaluation.eval_utils import RewardEvaluator
from executors.baseline import BaseExecutor
from generator_src.stasc_vllm_generation import collect_correction_stats
from prompts.cove_builder import CoVePromptBuilder
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.flatten import flatten_dict

logger = logging.getLogger(__name__)


class CoveExecutor(BaseExecutor):
    def __init__(
        self, cfg: Config, test_data: Dataset, sampling_params: SamplingParams, model: LLM
    ) -> None:
        super().__init__(cfg, test_data, sampling_params, model)
        self.prompt_builder: CoVePromptBuilder = get_prompt_builder(cfg.algo.name)(self.cfg)

    def execute_steps(self) -> None:
        """Executes the steps of the baseline algorithm."""
        with self.start_child_run("initial_generation"):
            mlflow.log_params(flatten_dict(self.cfg))
            responses = self.step_1(self.test_data)
            self.track_responses(responses, 1)
        with self.start_child_run("verification_plan"):
            responses = self.step_2(responses)
            self.track_responses(responses, 2)
        with self.start_child_run("verification_execution"):
            responses = self.step_3(responses)
            self.track_responses(responses, 3)
        with self.start_child_run("revision"):
            responses, stats = self.step_4(responses)
            self.track_responses(responses, 4)
        mlflow.log_metric("CxI", stats["correct_to_incorrect"])
        mlflow.log_metric("CxC", stats["correct_to_correct"])
        mlflow.log_metric("IxC", stats["incorrect_to_correct"])

    @override
    def step_1(self, dataset: Dataset) -> Any:
        ## Prompt builder
        few_shot_prompts = load_few_shot_prompts(self.cfg.dataset.few_shot_dir, "generation")
        prompts = self.prompt_builder.build_initial_generation_prompts(
            dataset=dataset,
            id_col=self.cfg.dataset.id_col,
            reference_col=self.cfg.dataset.gold_col,
            few_shot_prompts=few_shot_prompts,
        )
        responses = self.generate_responses(prompts)

        ## Calculate accuracy
        self.evaluate_responses(responses, "test")
        return responses

    def step_2(self, responses: ResponseWrapper) -> Any:
        few_shot_prompts = load_few_shot_prompts(
            self.cfg.dataset.few_shot_dir, "cove_verification_plan"
        )
        self.test_data = self.test_data.add_column(
            "initial_generation",
            [response.response for response in responses],
        )
        prompts = self.prompt_builder.build_verification_plan_prompt(
            dataset=self.test_data,
            initial_answer_col="initial_generation",
            few_shot_prompts=few_shot_prompts,
        )
        responses = self.generate_responses(prompts)

        ## Calculate accuracy
        self.evaluate_responses(responses)
        return responses

    def step_3(self, responses: ResponseWrapper) -> Any:
        few_shot_prompts = load_few_shot_prompts(
            self.cfg.dataset.few_shot_dir, "cove_verification_execution"
        )
        self.test_data = self.test_data.add_column(
            "verification_plan",
            [response.response for response in responses],
        )
        prompts = self.prompt_builder.build_verification_execution_prompt(
            dataset=self.test_data,
            verification_plan_col="verification_plan",
            few_shot_prompts=few_shot_prompts,
        )
        responses = self.generate_responses(prompts)
        ## Calculate accuracy
        self.evaluate_responses(responses)
        return responses

    def step_4(self, responses: ResponseWrapper) -> Any:
        few_shot_prompts = load_few_shot_prompts(self.cfg.dataset.few_shot_dir, "cove_revision")
        self.test_data = self.test_data.add_column(
            "cove_revision",
            [response.response for response in responses],
        )
        prompts = self.prompt_builder.build_correction_prompt(
            dataset=self.test_data,
            initial_answer_col="initial_generation",
            verification_execution_col="verification_plan",
            few_shot_prompts=few_shot_prompts,
        )
        responses = self.generate_responses(prompts)
        ## Calculate accuracy
        self.evaluate_responses(responses)

        stats_test = collect_correction_stats(
            dataset=self.test_data,
            reward_function=RewardEvaluator(self.cfg),
            question_col=self.cfg.dataset.question_col,
            reference_col=self.cfg.dataset.gold_col,
            inital_answer_col="initial_generation",
            correction_col="cove_revision",
        )
        return responses, stats_test
