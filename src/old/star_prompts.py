from prompts.prompt_schemas import compose_chat_messages

# ^ referencing your helper that builds a system-prompt + instructions + few-shot + user block
# And `tokenizer.apply_chat_template(...)` that finalizes the prompt string.


def star_rationale_generation_prompt(
    sample, tokenizer, question_col="question", few_shot_prompts=None
):
    """A STaR-style rationale-generation prompt:
    - Instructs the model to generate step-by-step reasoning (chain-of-thought)
    - Followed by the final answer
    - DOES NOT reveal the correct (ground-truth) answer as a 'hint'.
    """
    # 1) System role: instructions about generating a chain-of-thought
    system_prompt = (
        "You are a helpful reasoning assistant in general domain question answering. "
        "Please reason through the question step by step before giving a final answer."
    )

    # 2) High-level instructions for the user block
    #    Feel free to rephrase for your domain or tasks
    instructions = (
        "Generate a short chain-of-thought rationale, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # 3) The user question itself
    question_text = sample.get(question_col, "")
    user_question = (
        f"Question:\n{question_text}\n\nReason step by step, then conclude with the answer."
    )

    # 4) Build the message sequence
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,  # Optionally include a few chain-of-thought exemplars
    )

    # 5) Return the final merged prompt
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def star_rationalization_prompt(
    sample,
    tokenizer,
    question_col="question",
    gold_col="reference",
    few_shot_prompts=None,
    *args,
    **kwargs,
):
    """A STaR-style rationalization prompt:
    - Instructs the model to generate a chain-of-thought rationale
    - Includes the correct final answer as a "hint" (the 'add_hint' approach from STaR)
    - Then asks the model to produce that rationale + final answer (matching the correct one).
    - This is used for rationalizing a question AFTER we already know the correct answer.
    """
    # 1) System role: instructions
    system_prompt = (
        "You are a helpful reasoning assistant. "
        "You already know the correct final answer. Provide a concise rationale explaining "
        "why that answer is correct, step by step, then restate the correct answer."
    )

    # 2) High-level instructions
    instructions = (
        "Below is the question and the correct final answer. "
        "Generate a chain-of-thought that logically leads to that final answer. "
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # 3) The user block includes both the question and the correct answer (as the 'hint')
    question_text = sample.get(question_col, "")
    gold_answer = ",".join(sample.get(gold_col, ""))
    user_question = (
        f"Question:\n{question_text}\n\n"
        f"Correct (Gold) Answers: {gold_answer}\n\n"
        "Write a step-by-step explanation that arrives at this answer."
    )

    # 4) Build the message sequence
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,  # Could also omit or add exemplars
    )

    # 5) Return the final merged prompt
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
