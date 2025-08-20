from typing import Any, List

from datasets import Dataset
from encourage.prompts import Conversation, MetaData, Prompt, PromptCollection

from config import Config


class BaselinePromptBuilder:
    """A prompt builder for baseline tasks.
    This builder supports three stages:
      1) Initial generation of the answer.
    """

    system_prompt: str = ""
    initial_instructions: str = ""

    def __init__(self, config: Config) -> None:
        """:param config: A dictionary that can include:
        - question_col: The key for the user's question.
        """
        self.config = config
        self.question_col = config.dataset.question_col
        self.context_col = config.dataset.context_col
        self.use_cot = config.algo.use_cot
        self.use_init_context = config.algo.use_init_context
        self.num_documents = config.algo.num_documents

        if self.use_init_context:
            self.initial_instructions = (
                "Consider the Documents provided below\n" + self.initial_instructions
            )

    def _create_user_question(self, question_text: str) -> str:
        if self.use_cot:
            return (
                f"Question:\n{question_text}\n\n"
                "Reason step by step very shortly, then conclude with the answer. "
                "Strictly follow format Step-by-step reasoning: and Final Answer:"
            )
        else:
            return (
                f"Question:\n{question_text}\n\n"
                "Output a short final answer without any other reasoning. "
                "Strictly follow format Final Answer:"
            )

    def _create_context(self, contexts: List[str], num_documents: int) -> str:
        limit = min(num_documents, len(contexts))
        resulting_string = ""
        for i, document in enumerate(contexts[:limit]):
            resulting_string += f"Document {i}\n\n"
            resulting_string += f"{document}\n\n"

        return resulting_string

    def _prepend_context(self, prompt, sample, context_col) -> str:
        all_context = sample.get(context_col, [""])
        context_prompt = self._create_context(all_context, num_documents=self.num_documents)
        return context_prompt + prompt

    def build_initial_generation_prompts(
        self,
        dataset: Dataset,
        id_col: str = "",
        reference_col: str = "",
        few_shot_prompts: List[dict] = [],
    ) -> Any:
        """Builds the prompt for the initial answer generation."""
        prompts = []
        for sample in dataset:
            sample = sample if isinstance(sample, dict) else sample.to_dict()  # type: ignore
            question_text = sample.get(self.question_col, "")
            user_question = self._create_user_question(question_text)
            user_question = (
                self._prepend_context(user_question, sample, self.context_col)
                if self.use_init_context
                else user_question
            )

            conversation = Conversation(user_prompt=self.system_prompt)
            conversation.add_message("user", self.initial_instructions)
            if few_shot_prompts:
                for prompt in few_shot_prompts:
                    conversation.add_message("user", prompt["prompts"][0])
            conversation.add_message("user", user_question)
            prompts.append(
                Prompt(
                    id=sample.get(id_col, ""),
                    conversation=conversation,
                    meta_data=MetaData({"reference_answer": sample.get(reference_col, "")}),
                )
            )
        return PromptCollection.from_prompts(prompts)


class BaselineCOTPromptBuilder(BaselinePromptBuilder):
    # System prompts and instructions for each step
    system_prompt = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly,"
        "and then provide the final answer."
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )


class BaselineNoCOTPromptBuilder(BaselinePromptBuilder):
    system_prompt = (
        "You are a helpful assistant in general domain question answering. "
        "Please output only the final answer without any other information."
    )

    initial_instructions = (
        "Generate only a final answer without any additional information.\nFinal Answer:\n"
    )


class BaselineMathPromptBuilder(BaselinePromptBuilder):
    # System prompts and instructions for each step
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"

    initial_instructions = (
        "Please output the final answer immediately, do not include any "
        "other information, and put your final answer within \\boxed{{}}"
    )
