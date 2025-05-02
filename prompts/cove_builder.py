import json
from typing import List, Any
from prompts.prompt_schemas import compose_chat_messages
from .base import BasePromptBuilder

# generate initial response
# generate a verification plan. fact - verification
# execute verifications
# generate a revision
class CoVePromptBuilder(BasePromptBuilder):
    """
    A prompt builder for baseline tasks.
    This builder supports three stages:
      1) Initial generation of the answer.
    """

    def __init__(self, config: dict):
        """
        :param config: A dictionary that can include:
            - question_col: The key for the user's question.
        """
        self.config = config
        self.question_col = config.get("question_col", "question")


    def _create_user_question(self, question_text: str):
        
        return (
                f"Question:\n{question_text}\n\n"
                "Reason step by step very shortly, then conclude with the answer. "
                "Strictly follow format Step-by-step reasoning: and Final Answer:"
            )


    def build_initial_generation_prompt(self, sample: dict, tokenizer: Any, few_shot_prompts: List[dict] = None, tokenize: bool = False, *args, **kwargs) -> Any:
        """
        Builds the prompt for the initial answer generation.
        """
        question_text = sample.get(self.question_col, "")
        user_question = self._create_user_question(question_text)

        messages = compose_chat_messages(
            system_prompt=self.initial_system_prompt,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_verification_plan_prompt(self, sample: dict, tokenizer: Any, initial_answer_col="initial_generation",
        few_shot_prompts: List[dict] = None, tokenize: bool = False, *args, **kwargs) -> Any:
        """
        Builds a prompt for planning a verification of initial response
        """

        question_text = sample.get(self.question_col, "")
        user_question = self._create_user_question(question_text)
        initial_answers = sample.get(initial_answer_col, [])
        init_ans = initial_answers[0]
        
        messages = [
                {"role": "user", "content": self.initial_instructions},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_ans},
                {"role": "user", "content": self.verification_plan_prompt},
        ]

        if few_shot_prompts:
            messages.extend(few_shot_prompts)

            # Convert messages to final text
        final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
        )
        return final_prompt


    def build_verification_execution_prompt(self, sample: dict, tokenizer: Any, verification_plan_col="verification_plan",
        few_shot_prompts: List[dict] = None, tokenize: bool = False, *args, **kwargs) -> Any:
        """
        Builds a prompt for executing a verification
        """

        verification_plan = sample.get(verification_plan_col, [])[0]

        messages = [
                {"role": "user", "content": self.verification_execution_instructions},
                {"role": "user", "content": verification_plan},
        ]

        if few_shot_prompts:
            messages.extend(few_shot_prompts)

            # Convert messages to final text
        final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
        )
        return final_prompt

    def build_correction_prompt(self, sample: dict, tokenizer: Any, initial_answer_col='initial_generation', verification_execution_col="verification_execution",
        few_shot_prompts: List[dict] = None, tokenize: bool = False, *args, **kwargs) -> Any:
        """
        Builds a prompt for revising the response based on the verification executed
        """
        question_text = sample.get(self.question_col, "")
        user_question = self._create_user_question(question_text)
        initial_answer = sample.get(initial_answer_col, [])[0]
        verification = sample.get(verification_execution_col, [])[0]

        messages = [
                {"role": "user", "content": self.initial_instructions},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": initial_answer},
                {"role": "assistant", "content": verification},
                {"role": "user", "content": self.revision_prompt},
        ]

        if few_shot_prompts:
            messages.extend(few_shot_prompts)

            # Convert messages to final text
        final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
        )
        return final_prompt

    def build_correction_messages_with_final_answer(self, *args, **kwargs):
        raise NotImplementedError

class CoVeQAPromptBuilder(CoVePromptBuilder):

    # System prompts and instructions for each step
    initial_system_prompt = (
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
        "Building on the verifications, prioritize information in verififcation answers and perform the reasoning once again and output the final answer\n"
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )


class CoVeMathPromptBuilder(CoVePromptBuilder):

    # System prompts and instructions for each step
    initial_cot_system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"

    initial_no_cot_system_prompt = "Please output the final answer immediately, do not include any other information, and put your final answer within \\boxed{{}}"

    initial_cot_instructions = None

    initial_no_cot_instructions = None
