import logging
import os
import random
from pathlib import Path

import datasets
import hydra
import mlflow
import mlflow.data.pandas_dataset
from datasets import Dataset
from dotenv import load_dotenv
from vllm import SamplingParams

import executors  # noqa: F401, E402
from config import Config
from executors.factory import ExecutorRegistry
from utils.flatten import flatten_dict

config_path = str((Path(__file__).parents[1] / "config").resolve())

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=config_path, config_name="defaults")
def main(cfg: Config):
    """Main function to run the baseline algorithm."""
    logger.info(f"[INFO] Running baseline algorithm with config: {cfg}")

    # Mlflow setup
    load_dotenv(".env")
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_tracking_uri is None:
        raise ValueError("MLFLOW_TRACKING_URI not set in .env file")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("STASC")

    with mlflow.start_run(run_name=cfg.algo.name + "_" + str(random.choice(range(1000)))):
        logger.info(f"[INFO] Starting MLflow run with name: {cfg.run_name}")
        mlflow.log_params(flatten_dict(cfg))
        # Load dataset
        test_data: Dataset = datasets.load_dataset(cfg.dataset.data_path, split=cfg.dataset.split)  # type: ignore
        train_data: Dataset = datasets.load_dataset(cfg.dataset.data_path, split="train")  # type: ignore
        mlflow.log_input(
            mlflow.data.pandas_dataset.from_pandas(
                test_data.to_pandas(),  # type: ignore[reportArgumentType]
                name=cfg.dataset.data_path,
            ),
            context="inference",
        )

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=cfg.model.temperature,
            top_p=cfg.model.top_p,
            max_tokens=cfg.model.max_tokens,
            n=1,
            seed=cfg.model.random_seed,
        )

        ## Handle all variants of STASC
        name = cfg.algo.name
        if "stasc" in cfg.algo.name:
            name = "stasc"
        if "star" in cfg.algo.name:
            name = "stasc"

        ExecutorRegistry.create(
            name,
            cfg,
            test_data,
            train_data,
            sampling_params,
        ).execute_steps()


if __name__ == "__main__":
    main()
