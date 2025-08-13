# prompt_builders/base.py

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from datasets import Dataset


class BasePromptBuilder(ABC):
    """Abstract interface for building prompts/messages for a particular domain or task."""

    @abstractmethod
    def _create_user_question(self, question_text: str) -> Any:
        """Creates a prompt from just the question or problem text."""
        pass

    @abstractmethod
    def build_initial_generation_prompts(
        self,
        dataset: Dataset,
        id_col: Optional[str],
        reference_col: Optional[str],
        few_shot_prompts: Optional[List[dict]],
        *args,
        **kwargs,
    ) -> Any:
        """Build and return a final text prompt for the *initial generation* step.
        (Typically a single string.).
        """
        pass

    @abstractmethod
    def build_verification_plan_prompt(
        self,
        dataset: Dataset,
        initial_answer_col: str = "initial_generation",
        few_shot_prompts: List[dict] = [],
        *args,
        **kwargs,
    ) -> Any:
        """Build and return one or more text prompts for the *verification plan* step.
        (Often a list of strings, one per sample.).
        """
        pass

    @abstractmethod
    def build_correction_prompt(self, sample, tokenizer, *args, **kwargs) -> Any:
        """Build and return one or more final text prompts for the *correction* step.
        (Often a list of strings, one per initial answer.).
        """
        pass

    @abstractmethod
    def build_correction_messages_with_final_answer(
        self, question, init_answer, correction, *args, **kwargs
    ) -> Any:
        """Build and return a short conversation (as a list of message dicts)
        that ends with the final correction as the assistant's last message.
        (Used in collect_corrections-type scenarios.).
        """
        pass
