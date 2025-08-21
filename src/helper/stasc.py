import logging

import mlflow
import numpy as np
import pandas as pd
from datasets import Dataset

from evaluation.eval_utils import RewardEvaluator
from prompts.stasc_builder import StascQABuilder

logger = logging.getLogger(__name__)


def extract_single_reference(ref_array, row_idx):
    if isinstance(ref_array, np.ndarray) and ref_array.shape[0] == 1:
        return ref_array[0]
    elif isinstance(ref_array, np.ndarray):
        logger.warning(
            f"Row {row_idx}: reference_col has {ref_array.shape[0]} elements, expected 1."
        )
        return ref_array[0] if ref_array.shape[0] > 0 else ""
    else:
        logger.warning(f"Row {row_idx}: reference_col is not an array.")
        return ref_array


def dataset_to_df(
    dataset: Dataset,
    reward_function: RewardEvaluator,
    reference_col: str,
    init_answer_col: str,
    corr_answer_col: str,
) -> pd.DataFrame:
    # Convert dataset -> DataFrame
    df: pd.DataFrame = dataset.to_pandas()  # type: ignore
    # Ensure reference_col is a single string, warn if not

    df[reference_col] = df[reference_col].apply(
        lambda x: list(x) if isinstance(x, np.ndarray) else x
    )

    # Compute rewards
    df["init_reward"] = df.apply(
        lambda r: reward_function(
            ground_truth=r[reference_col],
            model_answer=r[init_answer_col],
        ),
        axis=1,
    )
    df["corr_reward"] = df.apply(
        lambda r: reward_function(
            ground_truth=r[reference_col],
            model_answer=r[corr_answer_col],
        ),
        axis=1,
    )
    df["init_reward"] = df["init_reward"].astype(bool)
    df["corr_reward"] = df["corr_reward"].astype(bool)
    return df


def filter_corrections(
    dataset: Dataset,
    reward_function: RewardEvaluator,
    prompt_builder: StascQABuilder,
    question_col="question",
    reference_col="reference",
    context_col="",
    init_answer_col="initial_answer",
    corr_answer_col="correction_answer",
    id_col="id",
    mode="improving",
):
    """Filter corrections based on reward function, using DataFrame ops."""
    df = dataset_to_df(
        dataset,
        reward_function,
        reference_col,
        init_answer_col,
        corr_answer_col,
    )

    ## Collect stats about the new dataset
    collect_correction_stats(df)

    # Apply filtering logic
    if mode == "non_decreasing":
        # both correct
        df["use_sample"] = df["corr_reward"] & df["init_reward"]
    elif mode == "improving":
        # correction correct but init not
        df["use_sample"] = df["corr_reward"] & (~df["init_reward"])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Filter rows
    filtered_df = df[df["use_sample"]].copy()

    # Build messages for training
    filtered_df["messages"] = filtered_df.apply(
        lambda r: prompt_builder.build_correction_messages_with_final_answer(
            r[question_col],
            r[init_answer_col],
            r[corr_answer_col],
            r.get(context_col, [""]),
        ),
        axis=1,
    )

    # Append "_gen" to ids
    filtered_df[id_col] = filtered_df[id_col].astype(str) + "_gen"

    print(f"[INFO] Filtered {len(filtered_df)} corrections in mode={mode}")
    mlflow.log_param("filtered_corrections", len(filtered_df))

    # Convert back to HF dataset
    return Dataset.from_pandas(
        filtered_df[
            [
                id_col,
                question_col,
                reference_col,
                init_answer_col,
                corr_answer_col,
                "messages",
                "init_reward",
                "corr_reward",
            ]
        ]
    )


def collect_correction_stats(df: pd.DataFrame) -> None:
    # Transition categories
    conditions = {
        "correct_to_incorrect": (df["init_reward"] & ~df["corr_reward"]),
        "correct_to_correct": (df["init_reward"] & df["corr_reward"]),
        "incorrect_to_correct": (~df["init_reward"] & df["corr_reward"]),
        "incorrect_to_incorrect": (~df["init_reward"] & ~df["corr_reward"]),
    }

    total = len(df)
    if total != 0:
        stats = {k: (v.sum() / total) * 100 for k, v in conditions.items()}
        mlflow.log_metric("CxI", stats["correct_to_incorrect"])
        mlflow.log_metric("CxC", stats["correct_to_correct"])
        mlflow.log_metric("IxC", stats["incorrect_to_correct"])
        mlflow.log_metric("IxI", stats["incorrect_to_incorrect"])
