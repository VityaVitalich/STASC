from datasets import Dataset
from encourage.prompts import Conversation, MetaData, Prompt, PromptCollection

from config import Config
from prompts.baseline_builder import BaselineCOTPromptBuilder


class StascPromptBuilder(BaselineCOTPromptBuilder):
    system_prompt: str = ""
    initial_instructions: str = ""
    correction_instructions: str = ""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.use_corr_context = config.algo.use_corr_context

    def build_correction_messages_with_final_answer(
        self,
        question: str,
        init_answer: str,
        correction: str,
        all_context: list[str],
    ) -> list[dict[str, str]]:
        """New helper method for a single conversation that ends
        with the final correction as the assistant's last message.

        The 'question' is optional but let's incorporate it so we have context.
        The idea is to mirror the logic of 'build_correction_prompt'
        but actually include the final corrected answer as the assistant response.
        """
        user_question = self._create_user_question(question)
        user_question = (
            (self._create_context(all_context, self.num_documents) + user_question)
            if self.use_init_context
            else user_question
        )

        correction_prompt = (
            (self._create_context(all_context, self.num_documents) + user_question)
            if self.use_corr_context
            else self.correction_instructions
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.initial_instructions},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": correction_prompt},
            {"role": "assistant", "content": correction},
        ]

        return messages

    def build_correction_prompts(
        self,
        dataset: Dataset,
        initial_answer_col: str = "initial_generation",
        few_shot_prompts: list[dict] = [],
    ) -> PromptCollection:
        """Builds prompts for revising the response based on executed verification."""
        prompts = []

        for sample in dataset:
            sample = sample if isinstance(sample, dict) else sample.to_dict()  # type: ignore

            question_text = sample.get(self.question_col, "")
            user_question = self._create_user_question(question_text)
            if self.use_init_context:
                user_question = self._prepend_context(user_question, sample, self.context_col)

            conversation = Conversation(user_prompt=self.system_prompt)
            conversation.add_message("user", self.initial_instructions)
            conversation.add_message("user", user_question)
            conversation.add_message("assistant", sample.get(initial_answer_col, ""))
            conversation.add_message("user", self.correction_instructions)

            # Add few-shot examples
            if few_shot_prompts:
                for prompt in few_shot_prompts:
                    conversation.add_message("user", prompt["prompts"][0])

            prompts.append(
                Prompt(
                    id=sample.get("id", ""),
                    conversation=conversation,
                    meta_data=MetaData(
                        {
                            "reference_answer": sample.get(self.config.dataset.gold_col, ""),
                            "initial_answer": sample.get(initial_answer_col, ""),
                            "correction_instructions": sample.get(self.correction_instructions, ""),
                        }
                    ),
                )
            )

        return PromptCollection.from_prompts(prompts)


class StascQABuilder(StascPromptBuilder):
    system_prompt: str = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly,"
        "and then provide the final answer."
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    correction_instructions = (
        "Consider the question and the initial answer. "
        "Generate a correction to the initial answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria."
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )


class StascRAGBuilder(StascQABuilder):
    # For the *initial generation* step
    system_prompt: str = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please use the provided documents and reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions: str = (
        "Consider the Documents provided below\n"
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    correction_instructions = (
        "Consider the question, the initial answer and provided documents. "
        "Generate a correction to the initial answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria."
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )
