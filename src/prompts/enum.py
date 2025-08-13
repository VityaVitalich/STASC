from enum import Enum
from typing import Any, Type

from prompts.baseline_builder import (
    BaselineQAPromptBuilder,
)
from prompts.cove_builder import CoVeQAPromptBuilder


class PromptBuilderType(Enum):
    BASE = "baseline_cot"
    COVE = "cove"


PROMPT_BUILDERS = {
    PromptBuilderType.BASE: BaselineQAPromptBuilder,
    PromptBuilderType.COVE: CoVeQAPromptBuilder,
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
