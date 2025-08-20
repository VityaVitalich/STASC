import copy
import logging

import mlflow
from datasets import Dataset, concatenate_datasets
from encourage.llm import ResponseWrapper
from vllm import LLM, SamplingParams

from config import Config
from evaluation.eval_utils import RewardEvaluator, collect_correction_stats
from executors.baseline import BaseExecutor
from executors.factory import ExecutorRegistry
from finetune.fine_tune import run_train
from helper.generation import generate_responses, init_model, unload_model
from helper.stasc import filter_corrections
from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from prompts.stasc_builder import StascQABuilder
from utils.flatten import flatten_dict

logger = logging.getLogger(__name__)


@ExecutorRegistry.register("stasc")
class STASCExecutor(BaseExecutor):
    def __init__(
        self,
        cfg: Config,
        test_data: Dataset,
        train_data: Dataset,
        sampling_params: SamplingParams,
    ) -> None:
        super().__init__(cfg, test_data, train_data, sampling_params)
        self.root_train_data = train_data
        self.prompt_builder: StascQABuilder = get_prompt_builder(cfg.algo.name)(self.cfg)
        self.train_dataset_path = cfg.dataset.data_path
        self.root_model_path_m0 = cfg.model.model_path
        self.model_path_m1 = cfg.model.model_path

    def execute_steps(self) -> None:
        """Executes the steps of the baseline algorithm."""
        ## Init Generation
        with self.start_child_run("init_generation"):
            mlflow.log_params(flatten_dict(self.cfg))

            root_model = init_model(self.cfg, self.root_model_path_m0)
            test_responses = self.step_1(self.test_data, root_model)
            train_responses = self.step_1(self.root_train_data, root_model)
            unload_model(root_model)
            self.evaluate_responses(test_responses, "test")
            self.evaluate_responses(train_responses, "train")

            self.track_responses(test_responses, "test_init")
            self.track_responses(train_responses, "train_init")

            self.root_train_data = self.root_train_data.add_column(
                "initial_generation",
                [response.response for response in train_responses],
            )  # type: ignore

        for iteration in range(self.cfg.algo.num_iterations):
            with self.start_child_run(f"iteration_{iteration}"):
                mlflow.log_params(flatten_dict(self.cfg))

                ## Step 1: Sample Initial Answers
                if not self.cfg.algo.fixed_initialization:
                    model = init_model(self.cfg, self.model_path_m1)
                    train_responses = self.step_1(self.root_train_data, model)
                    unload_model(model)
                    self.evaluate_responses(test_responses, "test")

                    self.root_train_data = self.root_train_data.remove_columns("initial_generation")
                    self.root_train_data = self.root_train_data.add_column(
                        "initial_generation",
                        [response.response for response in train_responses],
                    )  # type: ignore
                    logger.info("Override initial_generation for train data")

                ## Step 2: Sample Corrections
                self.step_2(iteration)

                ## Step 3: Filter Corrections
                ## Deciding how corrections are filtered
                self.step_3(iteration)

            # Step 4: Fine-Tune Model
            with self.start_child_run(f"fine-tune_{iteration}"):
                self.step_4(iteration)

                ## Step 5: Track Performance on Test Set
                model = init_model(self.cfg, self.model_path_m1)
                test_responses = self.step_1(self.test_data, model)
                self.track_responses(test_responses, f"test_{iteration}")
                unload_model(model)
                self.evaluate_responses(test_responses, "test")

                ## Update Generation Model Path for Step 1 to get Evolving Initialization
                if not self.cfg.algo.fixed_initialization:
                    self.generation_model_path = (
                        f"model/{self.cfg.model.model_name_short}_{iteration}"
                    )
                mlflow.log_param("generation_model_path", self.generation_model_path)

        logger.info("STASC algorithm completed.")

    def step_1(self, dataset: Dataset, model: LLM) -> ResponseWrapper:
        few_shot_prompts = load_few_shot_prompts(self.cfg.dataset.few_shot_dir, "generation")
        """Initial generation step."""
        # Prompt builder
        prompts = self.prompt_builder.build_initial_generation_prompts(
            dataset=dataset,
            id_col=self.cfg.dataset.id_col,
            reference_col=self.cfg.dataset.gold_col,
            few_shot_prompts=few_shot_prompts,
        )
        return generate_responses(self.cfg, prompts, model, self.sampling_params)

    def step_2(self, iteration: int) -> None:
        """Step 2: Sample Corrections."""
        few_shot_prompts = load_few_shot_prompts(self.cfg.dataset.few_shot_dir, "correction")
        # Prompt builder
        prompts = self.prompt_builder.build_correction_prompts(
            dataset=self.root_train_data,
            initial_answer_col="initial_generation",
            few_shot_prompts=few_shot_prompts,
        )
        sampling_params = copy.deepcopy(self.sampling_params)
        sampling_params.n = self.cfg.algo.number_corrections
        sampling_params.seed = None
        model = init_model(self.cfg, self.model_path_m1)
        correction_responses = generate_responses(self.cfg, prompts, model, sampling_params)
        unload_model(model)
        self.evaluate_responses(correction_responses, "train")
        self.track_responses(correction_responses, f"correction_{iteration}")

        temp_datasets = []
        for i in range(self.cfg.algo.number_corrections):
            ds = self.root_train_data.add_column(
                f"star_correction_{iteration}",
                [response.response[i] for response in correction_responses],
            )  # type: ignore
            temp_datasets.append(ds)
        self.train_data = concatenate_datasets(temp_datasets)

    def step_3(self, iteration: int) -> None:
        """Step 3: Filter Corrections."""
        ## Deciding how corrections are filtered
        mode = "improving" if self.cfg.algo.improving_filter else "non_decreasing"

        self.train_data = filter_corrections(
            dataset=self.train_data,
            reward_function=RewardEvaluator(self.cfg),
            prompt_builder=self.prompt_builder,
            question_col=self.cfg.dataset.question_col,
            reference_col=self.cfg.dataset.gold_col,
            init_answer_col="initial_generation",
            corr_answer_col=f"star_correction_{iteration}",
            id_col=self.cfg.dataset.id_col,
            mode=mode,
            threshold=0,
        )
        self.train_data.save_to_disk(self.root_folder + f"/iteration_{iteration}")
        self.train_dataset_path = self.root_folder + f"/iteration_{iteration}"
        # Log the train_data (HF dataset) to MLflow as an artifact
        mlflow.log_table(self.train_data.to_pandas(), artifact_file=f"train_data_{iteration}.json")  # type: ignore
        logger.info(f"Finished logging train_data for iteration {iteration} to MLflow")

        stats_test = collect_correction_stats(
            dataset=self.train_data,  # type: ignore
            reward_function=RewardEvaluator(self.cfg),
            reference_col=self.cfg.dataset.gold_col,
            inital_answer_col="initial_generation",
            correction_col=f"star_correction_{iteration}",
        )
        mlflow.log_metric("CxI", stats_test["correct_to_incorrect"])
        mlflow.log_metric("CxC", stats_test["correct_to_correct"])
        mlflow.log_metric("IxC", stats_test["incorrect_to_correct"])

    def step_4(self, iteration: int) -> None:
        """Step 4: Fine-Tuning."""
        self.cfg.dataset.data_path = self.train_dataset_path
        ## Decide whether to change between fixed and evolving fine-tuning
        if self.cfg.algo.fixed_fine_tuning:
            self.cfg.model.model_path = self.root_model_path_m0
        else:
            self.cfg.model.model_path = self.model_path_m1
        logger.info(f"Model Path changed to: {self.cfg.model.model_path}")

        run_train(self.cfg, iteration)
        mlflow.log_param("model_path_m1", self.model_path_m1)

        ## Override the new model name
        self.model_path_m1 = f"model/{self.cfg.model.model_name_short}_{iteration}"
        logger.info(f"M1 Path is now: {self.model_path_m1}")
