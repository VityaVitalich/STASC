import logging

import mlflow
from datasets import Dataset
from encourage.llm import ResponseWrapper
from vllm import LLM, SamplingParams  # pyright: ignore[reportPrivateImportUsage]

from configs.config import Config
from evaluation.eval_utils import RewardEvaluator, collect_correction_stats
from executors.baseline import BaseExecutor
from finetune.fine_tune import run_train
from generator_src.stasc_helper import collect_improving_corrections
from prompts.cove_builder import CoVePromptBuilder
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.flatten import flatten_dict
from utils.generation_utils import init_model, unload_model

logger = logging.getLogger(__name__)


class STASCExecutor(BaseExecutor):
    def __init__(
        self,
        cfg: Config,
        test_data: Dataset,
        train_data: Dataset,
        sampling_params: SamplingParams,
        model: LLM,
    ) -> None:
        super().__init__(cfg, test_data, sampling_params, model)
        self.train_data = train_data
        self.prompt_builder: CoVePromptBuilder = get_prompt_builder(cfg.algo.name)(self.cfg)
        self.train_dataset_path = cfg.dataset.data_path
        self.model_path = cfg.model.model_path

    def execute_steps(self) -> None:
        """Executes the steps of the baseline algorithm."""
        ## Init Generation
        with self.start_child_run("init_generation"):
            mlflow.log_params(flatten_dict(self.cfg))
            test_responses = self.step_1(self.test_data)
            train_responses = self.step_1(self.train_data)
            self.track_responses(test_responses, "test_init")
            self.track_responses(train_responses, "train_init")

            self.test_data = self.test_data.add_column(
                "initial_generation",
                [response.response for response in test_responses],
            )
            self.train_data = self.train_data.add_column(
                "initial_generation",
                [response.response for response in train_responses],
            )  # type: ignore
            self.evaluate_responses(test_responses, "test")
            self.evaluate_responses(train_responses, "train")

        for iteration in range(self.cfg.algo.num_star_iterations):
            with self.start_child_run(f"iteration_{iteration}"):
                mlflow.log_params(flatten_dict(self.cfg))
                test_responses = self.step_1(self.test_data)
                train_responses = self.step_1(self.train_data)
                self.track_responses(test_responses, f"test_{iteration}")
                self.track_responses(train_responses, f"train_{iteration}")

                self.test_data = self.test_data.add_column(
                    f"star_correction_{iteration}",
                    [response.response for response in test_responses],
                )
                self.train_data = self.train_data.add_column(
                    f"star_correction_{iteration}",
                    [response.response for response in train_responses],
                )  # type: ignore

                self.evaluate_responses(test_responses, "test")
                self.evaluate_responses(train_responses, "train")

                self.train_data = collect_improving_corrections(
                    dataset=self.train_data,
                    question_col=self.cfg.dataset.question_col,
                    reference_col=self.cfg.dataset.gold_col,
                    inital_answer_col="initial_generation",
                    correction_col=f"star_correction_{iteration}",
                    id_col=self.cfg.dataset.id_col,
                    strict_improvement=self.cfg.algo.only_better_correction,
                    reward_function=RewardEvaluator(self.cfg),
                    prompt_builder=self.prompt_builder,
                )
                self.train_data.save_to_disk(self.root_folder + f"/iteration_{iteration}")
                self.train_dataset_path = self.root_folder + f"/iteration_{iteration}"

                stats_test = collect_correction_stats(
                    dataset=self.test_data,
                    reward_function=RewardEvaluator(self.cfg),
                    question_col=self.cfg.dataset.question_col,
                    reference_col=self.cfg.dataset.gold_col,
                    inital_answer_col="initial_generation",
                    correction_col=f"star_correction_{iteration}",
                )
                mlflow.log_metric("CxI", stats_test["correct_to_incorrect"])
                mlflow.log_metric("CxC", stats_test["correct_to_correct"])
                mlflow.log_metric("IxC", stats_test["incorrect_to_correct"])

            with self.start_child_run(f"fine-tune_{iteration}"):
                self.fine_tune(iteration)
                logger.info(
                    f"Changing Model path to:model/{self.cfg.model.model_name_short}_{iteration}"
                )
                self.model_path = f"model/{self.cfg.model.model_name_short}_{iteration}"

        logger.info("STASC algorithm completed.")

    def step_1(self, dataset: Dataset) -> ResponseWrapper:
        few_shot_prompts = load_few_shot_prompts(self.cfg.dataset.few_shot_dir, "generation")
        """Initial generation step."""
        # Prompt builder
        prompts = self.prompt_builder.build_initial_generation_prompts(
            dataset=dataset,
            id_col=self.cfg.dataset.id_col,
            reference_col=self.cfg.dataset.gold_col,
            few_shot_prompts=few_shot_prompts,
        )
        return self.generate_responses(prompts)

    def fine_tune(self, iteration: int) -> None:
        unload_model(self.model)
        self.cfg.dataset.data_path = self.train_dataset_path
        self.cfg.model.model_path = self.model_path
        run_train(self.cfg, iteration)
        self.model = init_model(self.cfg)
