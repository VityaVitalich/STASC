from typing import List

from datasets import Dataset
from encourage.prompts import Conversation, MetaData, Prompt, PromptCollection

from config import Config
from prompts.baseline_builder import BaselineCOTPromptBuilder


class CoVePromptBuilder(BaselineCOTPromptBuilder):
    """A prompt builder for baseline tasks.
    This builder supports three stages:
      1) Initial generation of the answer.
    """

    system_prompt: str = ""
    initial_instructions: str = ""
    verification_plan_prompt: str = ""
    verification_execution_instructions: str = ""
    revision_prompt: str = ""

    def __init__(self, config: Config) -> None:
        """:param config: A dictionary that can include:
        - question_col: The key for the user's question.
        """
        self.config = config
        self.question_col = config.dataset.question_col
        self.system_prompt = self.system_prompt
        self.instructions = self.initial_instructions
        self.use_init_context = config.algo.use_init_context

    def _create_user_question(self, question_text: str):
        return (
            f"Question:\n{question_text}\n\n"
            "Reason step by step very shortly, then conclude with the answer. "
            "Strictly follow format Step-by-step reasoning: and Final Answer:"
        )

    def build_verification_plan_prompt(
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

            conversation = Conversation(user_prompt=self.system_prompt)
            conversation.add_message("user", self.initial_instructions)
            conversation.add_message("user", user_question)
            conversation.add_message("assistant", sample.get(initial_answer_col, ""))
            conversation.add_message("user", self.verification_plan_prompt)

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

    def build_verification_execution_prompt(
        self,
        dataset: Dataset,
        verification_plan_col: str = "verification_plan",
        few_shot_prompts: List[dict] = [],
    ) -> PromptCollection:
        """Builds prompts for executing verification based on the verification plan."""
        prompts = []

        for sample in dataset:
            sample = sample if isinstance(sample, dict) else sample.to_dict()  # type: ignore

            conversation = Conversation(user_prompt=self.system_prompt)
            conversation.add_message("user", self.verification_execution_instructions)
            conversation.add_message("user", sample.get(verification_plan_col, []))

            # Add few-shot examples
            if few_shot_prompts:
                for prompt in few_shot_prompts:
                    conversation.add_message("user", prompt["prompts"][0])

            prompts.append(
                Prompt(
                    id=sample.get("id", ""),  # or use id_col if defined
                    conversation=conversation,
                    meta_data=MetaData(
                        {
                            "reference_answer": sample.get(self.config.dataset.gold_col, ""),
                            "verification_plan": sample.get(verification_plan_col, []),
                        }
                    ),
                )
            )

        return PromptCollection.from_prompts(prompts)

    def build_correction_prompts(
        self,
        dataset: Dataset,
        initial_answer_col: str = "initial_generation",
        verification_execution_col: str = "verification_execution",
        few_shot_prompts: List[dict] = [],
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
            conversation.add_message("assistant", sample.get(verification_execution_col, ""))
            conversation.add_message("user", self.revision_prompt)

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
                            "verification_execution": sample.get(verification_execution_col, ""),
                        }
                    ),
                )
            )

        return PromptCollection.from_prompts(prompts)


class CoVeQAPromptBuilder(CoVePromptBuilder):
    # System prompts and instructions for each step
    system_prompt = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )

    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    verification_plan_prompt = (
        "Now your task is to plan a verification for provided answer.\n"
        "Divide the provided answer by the atomic facts or reasoning traces.\n"
        "For each atomic fact and reasoning trace generate a verififcation question.\n"
        "Strictly output in the following format:\n"
        "<fact in passage>, <verification question>\n"
        "<fact in passage>, <verification question>"
    )

    verification_execution_instructions = (
        "Now your task is to execute a verification for every question in provided verification plan.\n"
        "You must answer at every question in the verification plan\n"
        "Strictly output in the following format:\n"
        "<fact in passage>, <verification question>, <your answer>\n"
    )

    revision_prompt = (
        "Now your task is to revise the initial answer according to the verifications.\n"
        "Building on the verifications, prioritize information in verififcation answers and perform"
        "the reasoning once again and output the final answer\n"
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )


class CoVeMathPromptBuilder(CoVeQAPromptBuilder):
    # System prompts and instructions for each step
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"
