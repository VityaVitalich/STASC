# prompt_builders/score_math_builder.py

import sys
sys.path.append('../')
sys.path.append('./')

import os
import json
from functools import partial
from collections.abc import Iterable

from .base import BasePromptBuilder
from prompts.prompt_schemas import compose_chat_messages


class ScoreMathPromptBuilder(BasePromptBuilder):
    """
    An OOP-style prompt builder for a math/score domain (like your original code).
    It inherits from BasePromptBuilder, using more robust method names.
    """

    ## SCORE Prompts

    # initial_instructions = (
    #     "You are a math expert. When you respond, respond only with the Solution of the final Problem, thinking step by step."
    #     " At the end of the Solution, when you give your final answer, write it in the form"
    #     ' \"Final Answer: The final answer is $answer$\"'
    # )

    # # The instructions for correction
    # correction_instructions = (
    #     "There might be an error in the solution above because of lack of understanding of the question."
    #     " Please correct the error, if any, and rewrite the solution. Only output the final solution!"
    #     ' At the end of the Solution, when you give your final answer, write it in the form'
    #     ' \"Final Answer: The final answer is $answer$.\"'
    # )

    # init_system_prompt = None

    ## Qwen2.5-Math-CoT Prompts

    initial_instructions = None

    # # The instructions for correction
    correction_instructions = (
        "There might be an error in the solution above because of lack of understanding of the question."
        " Please correct the error, if any, and rewrite the solution. Put your final answer within \\boxed{{}}"
    )

    init_system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"

    def _create_user_question(self, question_text: str):
        # SCORE type
       # return f"Problem:\n{question_text}\n\nSolution:"

       # QWEN2.5-math type
        return f"{question_text}\n"

    def build_initial_generation_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):
        """
        Returns a single prompt string for initial generation.
        """

        # The user question
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        # Build the message sequence
        messages = compose_chat_messages(
            system_prompt=self.init_system_prompt,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        # Return the final merged prompt
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_correction_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        initial_answer_col="inital_answer",
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):
        """
        Returns a list of final prompt strings (one per initial answer).
        """

        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)
        initial_answers = sample.get(initial_answer_col, [])

        all_correction_prompts = []
        for init_ans in initial_answers:
            messages = [
                # score type
             #   {"role": "user", "content": self.initial_instructions},
            # qwen2.5 type
                {"role": "system", "content": self.init_system_prompt},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_ans},
                {"role": "user", "content": self.correction_instructions},
            ]


            # Convert messages to final text
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
            )
            all_correction_prompts.append(final_prompt)

        return all_correction_prompts

    def build_correction_messages_with_final_answer(
        self,
        question,
        init_answer,
        correction,
        *args,
        **kwargs
    ):
        """
        Creates a short conversation with the final corrected answer 
        as the assistant's last response.
        Returns a list of message dicts.
        """
        user_question = self._create_user_question(question)

        messages = [
            # score type
            # {"role": "user", "content": self.initial_instructions},
            # qwen type
            {"role": "system", "content": self.init_system_prompt},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": self.correction_instructions},
            {"role": "assistant", "content": correction},
        ]

        return messages
