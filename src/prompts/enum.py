from enum import Enum
from typing import Type

from prompts.base import BasePromptBuilder
from prompts.baseline_builder import BaselineMathPromptBuilder, BaselineQAPromptBuilder
from prompts.cove_builder import CoVeMathPromptBuilder, CoVeQAPromptBuilder
from prompts.debate_builder import DebateQAPromptBuilder
from prompts.math_builder import ScoreMathPromptBuilder
from prompts.qa_builder import QAPromptBuilder
from prompts.self_refine_builder import SelfRefineMathPromptBuilder, SelfRefineQAPromptBuilder


class PromptBuilderType(Enum):
    QA = "qa"
    MATH = "math"
    SELF_REFINE_QA = "self_refine_qa"
    SELF_REFINE_MATH = "self_refine_math"
    BASELINE_QA = "baseline_qa"
    BASELINE_MATH = "baseline_math"
    COVE_QA = "cove_qa"
    COVE_MATH = "cove_math"
    DEBATE_QA = "debate_qa"


PROMPT_BUILDERS = {
    PromptBuilderType.QA: QAPromptBuilder,
    PromptBuilderType.MATH: ScoreMathPromptBuilder,
    PromptBuilderType.SELF_REFINE_QA: SelfRefineQAPromptBuilder,
    PromptBuilderType.SELF_REFINE_MATH: SelfRefineMathPromptBuilder,
    PromptBuilderType.BASELINE_QA: BaselineQAPromptBuilder,
    PromptBuilderType.BASELINE_MATH: BaselineMathPromptBuilder,
    PromptBuilderType.COVE_QA: CoVeQAPromptBuilder,
    PromptBuilderType.COVE_MATH: CoVeMathPromptBuilder,
    PromptBuilderType.DEBATE_QA: DebateQAPromptBuilder,
}


def get_prompt_builder(task_type: str) -> Type[BasePromptBuilder]:
    try:
        enum_type = PromptBuilderType(task_type.lower())
        return PROMPT_BUILDERS[enum_type]
    except ValueError as err:
        raise ValueError(
            f"Unknown task_type={task_type}. Available types: ["
            f"{[e.value for e in PromptBuilderType]}"
            f"]"
        ) from err
