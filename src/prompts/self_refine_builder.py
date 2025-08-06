from typing import Any, List

from prompts.base import BasePromptBuilder
from prompts.prompt_schemas import compose_chat_messages


class SelfRefinePromptBuilder(BasePromptBuilder):
    """A prompt builder for self-refinement tasks.
    This builder supports three stages:
      1) Initial generation of the answer.
      2) Feedback on the generated answer.
      3) Refinement of the answer based on the latest feedback.
    """

    initial_system_prompt: str = ""
    initial_instructions: str = ""
    feedback_instructions: str = ""
    refinement_instructions: str = ""

    def __init__(self, config: dict):
        """:param config: A dictionary that can include:
        - question_col: The key for the user's question.
        - initial_gen_col: The key for the initial generated answer.
        - feedback_prefix: Prefix used for feedback rounds (e.g., "self_refine_feedback_").
        - refinement_prefix: Prefix used for refinement rounds (e.g., "self_refine_refinement_").
        """
        self.config = config
        self.question_col = config.get("question_col", "question")
        self.initial_gen_col = config.get("initial_gen_col", "self_refine_initial_generation")
        self.feedback_prefix = config.get("feedback_prefix", "self_refine_feedback_")
        self.refinement_prefix = config.get("refinement_prefix", "self_refine_refinement_")

    def _create_user_question(self, question_text: str):
        return (
            f"Question:\n{question_text}\n\n"
            "Reason step by step very shortly, then conclude with the answer. "
            "Strictly follow format Step-by-step reasoning: and Final Answer:"
        )

    def _gather_conversation_history(self, sample: dict) -> List[dict]:
        """Gathers the entire conversation from the sample.
        The conversation includes:
          - The user's initial question.
          - The initial assistant answer.
          - Each feedback and corresponding refinement round.
        """
        messages = []

        # Add the user's question.
        question_text = sample.get(self.question_col, "")
        user_question = self._create_user_question(question_text)

        messages.append({"role": "user", "content": question_text})

        # Add the initial generation (assumed to be a list with one element).
        initial_answers = sample.get(self.initial_gen_col, [])
        if initial_answers:
            initial_answer = initial_answers[0]
            messages.append({"role": "assistant", "content": f"Initial Answer: {initial_answer}"})

        # Iterate over subsequent feedback/refinement rounds.
        i = 0
        while True:
            fb_key = f"{self.feedback_prefix}{i}"
            ref_key = f"{self.refinement_prefix}{i}"

            feedback = sample.get(fb_key)
            refinement = sample.get(ref_key)

            # Stop iterating if no further feedback is found.
            if feedback is None:
                break

            # Append feedback message.
            feedback_text = feedback[0] if isinstance(feedback, list) else feedback
            messages.append({"role": "assistant", "content": f"Feedback {i}: {feedback_text}"})

            # Append corresponding refinement message if present.
            if refinement is None:
                break
            refinement_text = refinement[0] if isinstance(refinement, list) else refinement
            messages.append({"role": "assistant", "content": f"Refinement {i}: {refinement_text}"})
            i += 1

        return messages

    def build_initial_generation_prompt(
        self,
        sample: dict,
        tokenizer: Any,
        few_shot_prompts: List[dict] = [],
        tokenize: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Builds the prompt for the initial answer generation."""
        question_text = sample.get(self.question_col, "")
        user_question = self._create_user_question(question_text)
        messages = compose_chat_messages(
            system_prompt=self.initial_system_prompt,
            instructions=self.initial_instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts,
        )
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=False
        )

    def build_feedback_prompt(
        self,
        sample: dict,
        tokenizer: Any,
        few_shot_prompts: List[dict] = [],
        tokenize: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Builds the prompt for generating feedback on the previous assistant answer."""
        conversation = self._gather_conversation_history(sample)
        messages = [
            {"role": "user", "content": self.feedback_instructions},
        ]
        messages.extend(conversation)
        if few_shot_prompts:
            messages.extend(few_shot_prompts)
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=False
        )

    def build_correction_prompt(
        self,
        sample: dict,
        tokenizer: Any,
        few_shot_prompts: List[dict] = [],
        tokenize: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Builds the prompt for refining the assistant's answer using the most recent feedback."""
        conversation = self._gather_conversation_history(sample)
        messages = [
            {"role": "user", "content": self.refinement_instructions},
        ]
        messages.extend(conversation)
        if few_shot_prompts:
            messages.extend(few_shot_prompts)
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=True, enable_thinking=False
        )

    def build_correction_messages_with_final_answer(self, *args, **kwargs):
        raise NotImplementedError


class SelfRefineQAPromptBuilder(SelfRefinePromptBuilder):
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
    feedback_instructions = (
        "Below is the conversation so far. Please carefully review the assistant's latest answer, "
        "identify any issues consequently if there are any and conclude with overall Total Score of response."
        "If no issues are identified and the answer is correct output: !The Answer is correct!"
    )

    refinement_instructions = (
        "Consider the question, previous answers and feedback. "
        "Generate a refinement to the previous answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria. "
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )


class SelfRefineMathPromptBuilder(SelfRefinePromptBuilder):
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
    feedback_instructions = (
        "Below is the conversation so far. Please carefully review the assistant's latest answer, "
        "identify any issues consequently if there are any and conclude with overall Total Score of response."
        "If no issues are identified and the answer is correct output: !The Answer is correct!"
    )

    refinement_instructions = (
        "Consider the question, previous answers and feedback. "
        "Generate a refinement to the previous answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria. "
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )
