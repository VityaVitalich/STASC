from configs.config import Config
from prompts.baseline_builder import BaselineQAPromptBuilder


class StascQABuilder(BaselineQAPromptBuilder):
    system_prompt_init = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step very shortly before giving a final answer."
    )
    initial_instructions = (
        "Generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
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

    # For the *initial generation* step
    rag_system_prompt_init = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please use the provided documents and reason through the question step by step very shortly before giving a final answer."
    )
    rag_initial_instructions = (
        "Consider the Documents provided below, then "
        "generate a short chain-of-thought rationale very shortly, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # For the *correction* step
    rag_system_prompt_corr = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Your task is to correct the initial response if it is incorrect with the use of provided documents."
    )
    rag_correction_instructions = (
        "Consider the question, the initial answer and provided documents. "
        "Generate a correction to the initial answer if it is incorrect. "
        "Disregard the information you already have, look for other options. "
        "Do not use the information that does not match your criteria."
        "Think about your correction step-by-step and output answer in following format:\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.use_corr_context = config.algo.use_corr_context
        self.system_prompt = (
            self.rag_system_prompt_init if self.use_init_context else self.system_prompt_init
        )
        self.init_instructions = (
            self.rag_initial_instructions if self.use_init_context else self.initial_instructions
        )
        self.corr_instructions = (
            self.rag_correction_instructions
            if self.use_corr_context
            else self.correction_instructions
        )

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
            else self.corr_instructions
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.init_instructions},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": init_answer},
            {"role": "user", "content": correction_prompt},
            {"role": "assistant", "content": correction},
        ]

        return messages
