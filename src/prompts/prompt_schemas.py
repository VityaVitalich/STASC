import json
import os


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
