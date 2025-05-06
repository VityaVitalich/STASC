# prompt_builders/star_qa_builder.py

import sys
sys.path.append('../')
sys.path.append('./')

import os
import json
from functools import partial
from collections.abc import Iterable
from typing import List

from .base import BasePromptBuilder
from prompts.prompt_schemas import compose_chat_messages


class DebatePromptBuilder(BasePromptBuilder):



    def __init__(self, config):
        
        self.config = config
        self.judge_prompt = self.finalize_instructions if config['finalize_judge'] else self.most_common_instructions 

    def _create_user_question(self, question_text: str):

        return (
            f"Question:\n{question_text}\n\n"
            "Reason step by step very shortly, then conclude with the answer. "
            "Strictly follow format Step-by-step reasoning: and Final Answer:"
        )

    def _concat_corrections(self, corrections: List):
        result = ""
        for i, correction in enumerate(corrections):
            result += "Correction {i}: {correction}\n\n"
        return result

    def build_initial_generation_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        few_shot_prompts=None,
        tokenize=False,
        context_col='context',
        *args,
        **kwargs
    ):

        # Build user question
        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        # Compose messages
        messages = compose_chat_messages(
            system_prompt=self.system_prompt_init,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )

        # Return merged prompt
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

        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        initial_answers = sample.get(initial_answer_col, [])


        all_correction_prompts = []
        for init_ans in initial_answers:
            messages = [
                {"role": "system", "content": self.system_prompt_init},
                {"role": "user", "content": self.initial_instructions},
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": init_ans},
                {"role": "user", "content": self.correction_instructions},
            ]

            if few_shot_prompts:
                messages.extend(few_shot_prompts)

            # Convert messages to final text
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=True
            )
            all_correction_prompts.append(final_prompt)

        return all_correction_prompts

    def build_judge_prompt(
        self,
        sample,
        tokenizer,
        question_col="question",
        initial_answer_col="inital_answer",
        correction_col='correction'
        few_shot_prompts=None,
        tokenize=False,
        *args,
        **kwargs
    ):

        question_text = sample.get(question_col, "")
        user_question = self._create_user_question(question_text)

        initial_answers = sample.get(initial_answer_col, [])
        all_corrections = self._concat_corrections(sample[correction_col])

        messages = [
                {"role": "user", "content": (question_text + 
                    initial_answers +
                    all_corrections +
                    self.judge_prompt)},
        ]


    def build_correction_messages_with_final_answer(
        self,
        question,
        init_answer,
        correction,
        all_context,
        few_shot_prompts=None,
        *args,
        **kwargs
    ):
        pass


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

