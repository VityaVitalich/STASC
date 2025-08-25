# prompt_builders/star_qa_builder.py


from typing import List

from datasets import Dataset
from encourage.prompts import Conversation, MetaData, Prompt, PromptCollection

from config import Config
from prompts.baseline_builder import BaselineCOTPromptBuilder


class DebatePromptBuilder(BaselineCOTPromptBuilder):
    initial_system_prompt: str = ""
    initial_instructions: str = ""
    verification_plan_prompt: str = ""
    verification_execution_instructions: str = ""
    system_prompt_init: str = ""
    correction_instructions: str = ""
    finalize_instructions: str = ""
    most_common_instructions: str = ""

    def __init__(self, cfg: Config) -> None:
        self.judge_prompt = (
            self.finalize_instructions
            if cfg.algo.finalize_judgement
            else self.most_common_instructions
        )

    def build_correction_prompts(
        self,
        dataset: Dataset,
        initial_answer_col: str = "initial_generation",
        few_shot_prompts: List[dict] = [],
    ) -> PromptCollection:
        """Builds prompts for planning verification of initial responses."""
        prompts = []

        for sample in dataset:
            sample = sample if isinstance(sample, dict) else sample.to_dict()  # type: ignore
            question_text = sample.get(self.question_col, "")
            user_question = self._create_user_question(question_text)

            # Prepend context if needed
            if self.use_init_context:
                user_question = self._prepend_context(user_question, sample, self.context_col)

            conversation = Conversation(user_prompt=self.system_prompt_init)
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
                    id=sample.get("id", ""),  # or pass id_col if you have a column
                    conversation=conversation,
                    meta_data=MetaData(
                        {
                            "reference_answer": sample.get(self.config.dataset.gold_col, ""),
                            "initial_answer": sample.get(initial_answer_col, ""),
                        }
                    ),
                )
            )

        return PromptCollection.from_prompts(prompts)

    def build_judge_prompts(
        self,
        dataset: Dataset,
        initial_answer_col: str = "initial_generation",
        correction_col: str = "correction",
        few_shot_prompts: List[dict] = [],
    ) -> PromptCollection:
        """Builds prompts for judging corrections of initial responses."""
        prompts = []

        for sample in dataset:
            sample = sample if isinstance(sample, dict) else sample.to_dict()  # type: ignore
            question_text = sample.get(self.question_col, "")
            user_question = self._create_user_question(question_text)

            # Prepend context if needed
            if self.use_init_context:
                user_question = self._prepend_context(user_question, sample, self.context_col)

            initial_answer = sample.get(initial_answer_col, "")
            all_corrections = self._concat_corrections(sample.get(correction_col, []))

            conversation = Conversation(user_prompt=self.system_prompt)
            conversation.add_message("user", user_question)
            conversation.add_message("assistant", initial_answer)
            conversation.add_message("assistant", all_corrections)
            conversation.add_message("user", self.judge_prompt)

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
                            "initial_answer": initial_answer,
                            "corrections": all_corrections,
                        }
                    ),
                )
            )

        return PromptCollection.from_prompts(prompts)


class DebateQAPromptBuilder(DebatePromptBuilder):
    # For the *initial generation* step
    system_prompt_init = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # For the *correction* step
    system_prompt_corr = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Your task is to correct the initial response if it is incorrect."
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

    finalize_instructions = (
        "You are a fair and trustworthy judge, your task is to read several responses and conclude with the final answer. "
        "You are provided the question, initial answer and several corrections above. "
        "Read them, think about them and output the final answer, based on those responses.\n"
        "Think about them step-by-step and output the answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    most_common_instructions = (
        "Your task is to read several responses and output the most common final answer. "
        "You are provided the question, initial answer and several corrections above. "
        "Read them and output the most common final answer of those responses.\n"
        "Do not add any extra information or think about question, just extract the most common final answer. \n"
        "Think step-by-step and output the answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )
