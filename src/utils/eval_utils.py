import warnings
from typing import List

from configs.config import Config
from utils.math_grader import grade_answer
from utils.qa_grader import EM_compute, F1_compute, has_answer
from utils.qwen_math_parser import extract_answer

reward_functions = {
    "in_acc": has_answer,
    "f1": F1_compute,
    "em": EM_compute,
    "math_acc": grade_answer,
}


def split_rationale_and_final_answer(generated_text: str, answer_marker: str = "Final Answer:"):
    """Splits a STaR-style generation into two parts:
      1) The rationale (everything after 'Step-by-step reasoning:' until the answer marker)
      2) The final answer (everything after the answer marker).

    The search for the answer marker (and the rationale marker) is performed in a case-insensitive
    manner.
    """
    rationale_marker = "Step-by-step reasoning:"
    text = generated_text.replace("\r", "")

    # Convert to lower-case for case-insensitive search.
    lower_text = text.lower()
    rationale_marker_lower = rationale_marker.lower()
    answer_marker_lower = answer_marker.lower()

    rationale = ""
    final_ans = ""

    # Find indices using the lower-case text.
    rationale_start_idx = lower_text.find(rationale_marker_lower)
    answer_start_idx = lower_text.find(answer_marker_lower)

    if rationale_start_idx != -1:
        rationale_start = rationale_start_idx + len(rationale_marker)
        if answer_start_idx != -1 and answer_start_idx > rationale_start:
            rationale = text[rationale_start:answer_start_idx].strip()
        else:
            rationale = text[rationale_start:].strip()
    if answer_start_idx != -1:
        answer_start = answer_start_idx + len(answer_marker)
        final_ans = text[answer_start:].strip()

    # return rationale, final_ans
    return final_ans


class RewardEvaluator:
    def __init__(self, config: Config):
        """mode: A string specifying the evaluation mode.
              'default' uses has_answer as-is.
              'final' extracts the final answer from the generated text.
              Other modes can be added.
        config: Additional configuration parameters.
        """
        self.config = config
        self.mode = config.dataset.evaluator_mode
        self.reward_function = reward_functions[config.dataset.evaluator_function]
        self.extractor = (
            split_rationale_and_final_answer
            if self.config.dataset.task_type == "qa"
            else extract_answer
        )

        if (self.config.dataset.evaluator_function == "math_acc") and (
            self.config.dataset.evaluator_mode != "final"
        ):
            warnings.warn(
                "Reward Function is `Math Acc` but Evaluator is not `Final`. Setting to `Final`",
                stacklevel=2,
            )
            self.mode = "final"

    def __call__(self, ground_truth: List, model_answer: str):
        if self.mode == "default":
            return self.reward_function(ground_truth, model_answer)
        elif self.mode == "final":
            final_ans = self.extractor(
                generated_text=model_answer,
                answer_marker=self.config.dataset.evaluator_answer_marker,
            )
            return self.reward_function(ground_truth, final_ans)
        else:
            raise ValueError(f"Unknown mode {self.mode}")
