from .qa_builder import QAPromptBuilder
from .math_builder import ScoreMathPromptBuilder

PROMPT_BUILDERS = {
    "qa": QAPromptBuilder,
    "math": ScoreMathPromptBuilder,
}

def get_prompt_builder(task_type: str):
    builder_cls = PROMPT_BUILDERS.get(task_type.lower())
    if not builder_cls:
        raise ValueError(f"Unknown task_type={task_type}. "
                         f"Available types: {list(PROMPT_BUILDERS.keys())}")
    return builder_cls