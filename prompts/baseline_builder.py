import json
from typing import List, Any
from prompts.prompt_schemas import compose_chat_messages
from .base import BasePromptBuilder


class BaselinePromptBuilder(BasePromptBuilder):
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
        self.context_col = config.get('context_col', 'context')
        self.use_cot = config['use_cot']
        self.use_init_context = config['use_init_context']
        self.num_documents = config['num_documents']

        self.system_prompt = self.initial_cot_system_prompt if self.use_cot else self.initial_no_cot_system_prompt
        self.instructions = self.initial_cot_instructions if self.use_cot else self.initial_no_cot_instructions

        if self.use_init_context:
            self.instructions = "Consider the Documents provided below\n" + self.instructions


    def _create_user_question(self, question_text: str):
        
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

    
    def _create_context(self, contexts: List[str], num_documents: int):
        
        limit = min(num_documents, len(contexts))
        resulting_string = ""
        for i, document in enumerate(contexts[:limit]):
            resulting_string += f'Document {i}\n\n'
            resulting_string += f'{document}\n\n'

        return resulting_string

    def _prepend_context(self, prompt, sample, context_col):

        all_context = sample.get(context_col, [""])
        context_prompt = self._create_context(all_context, num_documents=self.num_documents)
        return context_prompt + prompt

    def build_initial_generation_prompt(self, sample: dict, tokenizer: Any, few_shot_prompts: List[dict] = None, tokenize: bool = False, *args, **kwargs) -> Any:
        """
        Builds the prompt for the initial answer generation.
        """
        question_text = sample.get(self.question_col, "")
        user_question = self._create_user_question(question_text)
        user_question = self._prepend_context(user_question, sample, self.context_col) if self.use_init_context else user_question

        messages = compose_chat_messages(
            system_prompt=self.system_prompt,
            instructions=self.instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=True
        )

    def build_correction_prompt(self, *args, **kwargs) -> Any:
        raise NotImplementedError
        
    def build_correction_messages_with_final_answer(self, *args, **kwargs):
        raise NotImplementedError

class BaselineQAPromptBuilder(BaselinePromptBuilder):

    # System prompts and instructions for each step
    initial_cot_system_prompt = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_no_cot_system_prompt = (
        "You are a helpful assistant in general domain question answering. "
        "Please output only the final answer without any other information."
    )

    initial_cot_instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    initial_no_cot_instructions = (
        "Generate only a final answer without any additional information.\n"
        "Final Answer:\n"
    )

class BaselineMathPromptBuilder(BaselinePromptBuilder):

    # System prompts and instructions for each step
    initial_cot_system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}"

    initial_no_cot_system_prompt = "Please output the final answer immediately, do not include any other information, and put your final answer within \\boxed{{}}"

    initial_cot_instructions = None

    initial_no_cot_instructions = None
