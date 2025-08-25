import warnings
from typing import Callable, List

from encourage.llm import ResponseWrapper

from config import Config
from evaluation.math_grader import grade_answer
from evaluation.qa_grader import EM_compute, F1_compute, has_answer
from evaluation.qwen_math_parser import extract_answer


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
        if type(generated_text) is not str:
            print(f"Generated text is not a string: {generated_text}")
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
