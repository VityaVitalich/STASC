from enum import Enum
from typing import Any, Type

from prompts.baseline_builder import (
    BaselineQAPromptBuilder,
)
from prompts.cove_builder import CoVeQAPromptBuilder
from prompts.stasc_builder import StascQABuilder


class PromptBuilderType(Enum):
    BASELINE_COT = "baseline_cot"
    BASELINE_NO_COT = "baseline_no_cot"
    COVE = "cove"
    STASC = "stasc"


PROMPT_BUILDERS = {
    PromptBuilderType.BASELINE_COT: BaselineQAPromptBuilder,
    PromptBuilderType.BASELINE_NO_COT: BaselineQAPromptBuilder,
    PromptBuilderType.COVE: CoVeQAPromptBuilder,
    PromptBuilderType.STASC: StascQABuilder,
}


def get_prompt_builder(task_type: str) -> Type[Any]:
    try:
        enum_type = PromptBuilderType(task_type.lower())
        return PROMPT_BUILDERS[enum_type]
    except ValueError as err:
        raise ValueError(
            f"Unknown task_type={task_type}. Available types: ["
            f"{[e.value for e in PromptBuilderType]}"
            f"]"
        ) from err
