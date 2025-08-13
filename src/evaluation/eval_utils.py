import warnings
from typing import Any, Callable, Dict, Iterable, List, Union

from encourage.llm import ResponseWrapper

from configs.config import Config
from evaluation.math_grader import grade_answer
from evaluation.qa_grader import EM_compute, F1_compute, has_answer
from evaluation.qwen_math_parser import extract_answer
from utils.flatten import flatten_predictions


class RewardEvaluator:
    """Evaluates model answers against ground truth using a chosen reward function."""

    REWARD_FUNCTIONS: dict[str, Callable] = {
        "in_acc": has_answer,
        "f1": F1_compute,
        "em": EM_compute,
        "math_acc": grade_answer,
    }

    def __init__(self, config: Config) -> None:
        """Args:
        config (Config): Configuration object with dataset settings.

        """
        self.config = config
        self.mode: str = config.dataset.evaluator_mode
        self.reward_function: Callable = self._get_reward_function(
            config.dataset.evaluator_function
        )
        self.extractor: Callable = self._select_extractor(config.dataset.task_type)

        if config.dataset.evaluator_function == "math_acc" and self.mode != "final":
            warnings.warn(
                "Reward Function is `Math Acc` but Evaluator is not `Final`. Setting to `Final`.",
                stacklevel=2,
            )
            self.mode = "final"

    def _get_reward_function(self, name: str) -> Callable:
        """Returns the reward function for the given name."""
        if name not in self.REWARD_FUNCTIONS:
            raise ValueError(f"Unknown reward function: {name}")
        return self.REWARD_FUNCTIONS[name]

    def _select_extractor(self, task_type: str) -> Callable:
        """Selects the appropriate answer extractor based on task type."""
        return self.extract_final_answer if task_type == "qa" else extract_answer

    @staticmethod
    def extract_final_answer(generated_text: str, answer_marker: str = "Final Answer:") -> str:
        """Extracts final answer after the answer marker (case-insensitive)."""
        clean_text = generated_text.lower().replace("\r", "")
        idx = clean_text.find(answer_marker.lower())
        return generated_text[idx + len(answer_marker) :].strip() if idx != -1 else ""

    def __call__(self, ground_truth: List, model_answer: str):
        """Evaluates model answer according to the selected mode."""
        if self.mode == "default":
            return self.reward_function(ground_truth, model_answer)
        elif self.mode == "final":
            final_ans = self.extractor(
                generated_text=model_answer,
                answer_marker=self.config.dataset.evaluator_answer_marker,
            )
            return self.reward_function(ground_truth, final_ans)
        raise ValueError(f"Unknown mode {self.mode}")


def evaluate_responses(responses: ResponseWrapper, evaluator: RewardEvaluator) -> float:
    """Calculate the accuracy of the model's responses against the ground truth."""
    total_score = 0
    total_count = 0

    for response in responses.response_data:
        corr_ans: str = response.meta_data.tags.get("reference_answer", "")
        model_ans: str = response.response

        if not isinstance(model_ans, list):
            scores = [evaluator(ground_truth=corr_ans, model_answer=model_ans)]
        elif isinstance(model_ans[0], list):
            scores = [
                evaluator(ground_truth=corr_ans, model_answer=sub_ans)
                for sub_list in model_ans
                for sub_ans in sub_list
            ]
        else:
            scores = [evaluator(ground_truth=corr_ans, model_answer=ans) for ans in model_ans]

        total_score += sum(scores)
        total_count += len(scores)

    return total_score / total_count if total_count > 0 else 0.0


def collect_correction_stats(
    dataset: Iterable[Dict[str, Union[str, List[Any], Any]]],
    reward_function: Callable[..., bool],
    question_col: str = "question",
    reference_col: str = "reference",
    inital_answer_col: str = "star_correction_initial_generation",
    correction_col: str = "star_correction",
) -> Dict[str, float]:
    """Computes statistics on answer corrections as percentages:.

    - correct_to_incorrect: Correct → Incorrect
    - correct_to_correct:   Correct → Correct
    - incorrect_to_correct: Incorrect → Correct

    Args:
        dataset: Iterable of rows containing answers.
        reward_function: Function to check correctness (returns bool).
        question_col: Column name for the question (unused here but kept for consistency).
        reference_col: Column name for ground truth answers.
        inital_answer_col: Column name for initial model answers.
        correction_col: Column name for corrected model answers.

    Returns:
        A dictionary with percentages for each category.

    """
    correct_to_incorrect = correct_to_correct = incorrect_to_correct = total_corrections = 0

    for row in dataset:
        reference = row[reference_col]
        initial_answers = row[inital_answer_col]  # type: ignore
        corrected_answers = row[correction_col]  # type: ignore

        for init_answer in initial_answers:  # type: ignore
            init_correct = reward_function(ground_truth=reference, model_answer=init_answer)

            for correction in flatten_predictions(corrected_answers):
                correction_correct = reward_function(
                    ground_truth=reference, model_answer=correction
                )
                total_corrections += 1

                if init_correct and not correction_correct:
                    correct_to_incorrect += 1
                elif init_correct and correction_correct:
                    correct_to_correct += 1
                elif not init_correct and correction_correct:
                    incorrect_to_correct += 1

    if total_corrections == 0:
        return dict.fromkeys(
            ["correct_to_incorrect", "correct_to_correct", "incorrect_to_correct"], 0.0
        )  # ty: ignore

    return {
        "correct_to_incorrect": (correct_to_incorrect / total_corrections) * 100,
        "correct_to_correct": (correct_to_correct / total_corrections) * 100,
        "incorrect_to_correct": (incorrect_to_correct / total_corrections) * 100,
    }
