import json
import os
from collections.abc import Iterable


def load_few_shot_prompts(few_shot_dir, prompt_type) -> list:
    """Loads a single few-shot file from `few_shot_dir/` based on the prompt_type.
    For example, if prompt_type="critic", we look for "critic.json" in that dir.

    Returns a list of role/content dicts, or None if no file is found.
    """
    file_name = f"{prompt_type}.json"  # e.g. "critic.json"
    file_path = os.path.join(few_shot_dir, file_name)
    if not os.path.exists(file_path):
        print(
            f"[INFO] No few-shot file found for prompt_type='{prompt_type}' at '{file_path}'. Using none."
        )
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Few-shot file must contain a JSON list of role/content dicts. Found: {file_path}"
        )

    print(f"[INFO] Loaded {len(data)} few-shot messages from {file_path}")
    return data


def flatten_list(nested_list):
    """Recursively flatten a nested list (ignores strings as iterables)."""
    flat_list = []
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, str):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def compose_chat_messages(
    system_prompt: str, instructions: str, user_question: str, few_shot_prompts=None
):
    """Constructs the chat messages in the following order:
    1) system_prompt  (role: system)
    2) instructions   (role: user)
    3) few_shot_prompts (list of roles: user/assistant)
    4) user_question  (role: user)
    """
    messages = []

    # 1) System prompt
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})

    # 2) Instructions
    if instructions:
        messages.append({"role": "user", "content": instructions})

    # 3) Few-shot prompts
    if few_shot_prompts and isinstance(few_shot_prompts, list):
        messages.extend(few_shot_prompts)

    # 4) Actual user question
    messages.append({"role": "user", "content": user_question})

    return messages


def generate_prompt(
    sample,
    tokenizer,
    use_context_col="none",
    question_col="question",
    few_shot_prompts=None,
    *args,
    **kwargs,
):
    """Generates a Q&A prompt, optionally including context. Uses `question_col` for question text."""
    system_prompt = "You are a helpful assistant tasked with answering questions."

    instructions = "Answer the following question concisely.\n"
    if use_context_col != "none":
        instructions = "Use the following documents if they are relevant:\n\n" + instructions

    # Potential context
    context_str = ""
    if use_context_col != "none":
        all_context = sample.get(use_context_col, "")
        if isinstance(all_context, list):
            all_context = [f"Passage {i}: {passage}" for i, passage in enumerate(all_context)]
            all_context = "\n".join(all_context)
        context_str = f"{all_context}\n\n"

    question_text = sample.get(question_col, "No question provided.")
    user_question = f"{context_str}Question: {question_text}\n\nAnswer briefly:"

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,
    )

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_when_to_retrieve_prompt(
    sample,
    tokenizer,
    use_context_col="none",
    question_col="question_text",
    few_shot_prompts=None,
    *args,
    **kwargs,
):
    """If context is provided, include it. Otherwise just rely on internal knowledge.
    By default, we assume the question is in `question_text` (but you can override).
    """
    system_prompt = "You are a helpful assistant."

    instructions = ""
    context_str = ""
    if use_context_col != "none":
        all_context = sample.get(use_context_col, "")
        if isinstance(all_context, list):
            all_context = [f"Passage {i}: {passage}" for i, passage in enumerate(all_context)]
            all_context = "\n".join(all_context)
        instructions = "Use the provided information or your internal knowledge to answer:\n"
        context_str = f"{all_context}\n\n"
    else:
        instructions = "Answer based on your internal knowledge:\n"

    question_text = sample.get(question_col, "No question provided.")
    user_question = f"{context_str}Question: {question_text}\nAnswer in one or few words:"

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,
    )

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def critic_prompt(
    sample, tokenizer, answer_col, question_col="question", few_shot_prompts=None, *args, **kwargs
):
    """Critiques an existing answer, focusing on factual correctness.
    `question_col` determines which field has the question text.
    """
    system_prompt = (
        "You are a helpful critic assistant. Identify factual errors in the given answer, "
        "or confirm correctness if there are no errors."
    )
    instructions = (
        "Explain any factual inaccuracies succinctly (in bullet points). If correct, say so briefly. "
        "Do not provide a revised answer yourself."
    )
    question_text = sample.get(question_col, "No question provided.")
    user_question = (
        f"QUESTION: {question_text}\nANSWER: {sample.get(answer_col, 'No answer provided.')}"
    )

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,
    )

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def revision_prompt(
    sample,
    tokenizer,
    critic_col,
    answer_col,
    question_col="question",
    few_shot_prompts=None,
    *args,
    **kwargs,
):
    """Generates revision prompts based on criticisms found in `critic_col`.
    Each criticism yields a sub-prompt.
    Uses `question_col` to fetch the question text.
    """
    system_prompt = (
        "You are a helpful assistant tasked with revising answers based on provided criticisms."
    )
    instructions = "Revise the INITIAL ANSWER in one sentence based on the given CRITICISM."

    criticisms = sample.get(critic_col, [])
    if not isinstance(criticisms, list):
        criticisms = [criticisms]  # In case it's a single string

    result_prompts = []
    question_text = sample.get(question_col, "No question provided.")
    initial_answer = sample.get(answer_col, "No initial answer provided.")

    for criticism in criticisms:
        user_question = (
            f"QUESTION: {question_text}\n\n"
            f"INITIAL ANSWER: {initial_answer}\n\n"
            f"CRITICISM: {criticism}\n\n"
            "REVISED ANSWER:"
        )

        messages = compose_chat_messages(
            system_prompt=system_prompt,
            instructions=instructions,
            user_question=user_question,
            few_shot_prompts=few_shot_prompts,
        )

        result_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        result_prompts.append(result_prompt)

    return result_prompts


def verbalized_confidence_1s(
    sample, tokenizer, question_col="question", few_shot_prompts=None, *args, **kwargs
):
    """Prompts the model to provide a step-by-step solution and probability (0.0 to 1.0).
    Uses `question_col` to extract the question from `sample`.
    """
    system_prompt = (
        "You are a helpful assistant tasked with answering questions and providing "
        "a confidence level for your answers."
    )

    instructions = (
        "Provide your best guess and the probability that it is correct (0.0 to 1.0) for the following question."
        "Give ONLY the guess and probability, no other words orexplanation."
        "For example:\n\nGuess: <most likely guess, as short as possible; not a complete sentence, just the guess!>"
        "Probability: <the probability between 0.0 and 1.0 that your guess is correct, without any extra commentary whatsoever; just the probability!>\n\n"
    )

    question_text = sample.get(question_col, "No question provided.")
    user_question = f"Question: {question_text}"

    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,
    )

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def verbalized_confidence_1s_topk(
    sample, tokenizer, k, question_col="question", few_shot_prompts=None, *args, **kwargs
):
    """Prompts the model to provide the top k guesses with probabilities (0.0 to 1.0) for the given question.
    Uses `question_col` to extract the question from `sample`.
    """
    # System prompt: basic introduction about the assistant
    system_prompt = (
        "You are a helpful assistant tasked with answering questions and providing "
        "a confidence level for each of your answers."
    )

    # Instructions for the model
    instructions = (
        f"Provide your {k} best guesses and the probability that each is correct (0.0 to 1.0) for the following question. "
        "Give ONLY the guesses and probabilities, no other words or explanation. For example:\n\n"
    )

    # Dynamically build the instruction part for the top k guesses
    top_k_instructions = ""
    for i in range(1, k + 1):
        top_k_instructions += f"G{i}: <the {i}-th most likely guess, as short as possible; not a complete sentence, just the guess!>\n"
        top_k_instructions += f"P{i}: <the probability between 0.0 and 1.0 that G{i} is correct, without any extra commentary whatsoever; just the probability!>\n"

    # Add the question text as the final prompt
    question_text = sample.get(question_col, "No question provided.")
    user_question = f"The question is: {question_text}"

    # Construct messages for the tokenizer
    messages = compose_chat_messages(
        system_prompt=system_prompt,
        instructions=instructions,
        user_question=user_question,
        few_shot_prompts=few_shot_prompts,
    )

    # Return the formatted prompt using the tokenizer
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# A dictionary to easily retrieve prompt functions by name
prompt_mapper = {
    "generate": generate_prompt,
    "critic": critic_prompt,
    "revise": revision_prompt,
    "when_to_retrieve": generate_when_to_retrieve_prompt,
    "verbalized_1s": verbalized_confidence_1s,
    "verbalized_1s_topk": verbalized_confidence_1s_topk,
}
