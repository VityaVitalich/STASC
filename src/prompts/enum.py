from enum import Enum
from typing import Any, Type

from prompts.baseline_builder import (
    BaselineCOTPromptBuilder,
    BaselineNoCOTPromptBuilder,
)
from prompts.cove_builder import CoVeQAPromptBuilder
from prompts.stasc_builder import StascQABuilder, StascRAGBuilder


class PromptBuilderType(Enum):
    BASELINE_COT = "baseline_cot"
    BASELINE_NO_COT = "baseline_no_cot"
    COVE = "cove"
    STASC = "stasc"
    STASC_EIE = "stasc_eie"
    STASC_EIF = "stasc_eif"
    STASC_ENE = "stasc_ene"
    STASC_ENF = "stasc_enf"
    STASC_FIE = "stasc_fie"
    STASC_FIF = "stasc_fif"
    STASC_FNE = "stasc_fne"
    STASC_FNF = "stasc_fnf"
    STASC_RAG = "stasc_rag"


PROMPT_BUILDERS = {
    PromptBuilderType.BASELINE_COT: BaselineCOTPromptBuilder,
    PromptBuilderType.BASELINE_NO_COT: BaselineNoCOTPromptBuilder,
    PromptBuilderType.COVE: CoVeQAPromptBuilder,
    PromptBuilderType.STASC: StascQABuilder,
    PromptBuilderType.STASC_EIE: StascQABuilder,
    PromptBuilderType.STASC_EIF: StascQABuilder,
    PromptBuilderType.STASC_ENE: StascQABuilder,
    PromptBuilderType.STASC_ENF: StascQABuilder,
    PromptBuilderType.STASC_FIE: StascQABuilder,
    PromptBuilderType.STASC_FIF: StascQABuilder,
    PromptBuilderType.STASC_FNE: StascQABuilder,
    PromptBuilderType.STASC_FNF: StascQABuilder,
    PromptBuilderType.STASC_RAG: StascRAGBuilder,
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
