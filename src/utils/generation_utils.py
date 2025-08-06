from pathlib import Path

import yaml

from configs.config import Config


def load_config(config_path: Path) -> dict:
    """Loads a YAML configuration file.

    Args:
        config_path (Path): Path to the YAML config file.

    Returns:
        dict: Configuration parameters.

    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def extract_parameters(config_dict: dict) -> Config:
    """Extracts and initializes the Config data class from the configuration dictionary.

    Args:
        config_dict (dict): Configuration parameters loaded from YAML.

    Returns:
        Config: An instance of the Config data class with all parameters.

    """
    return Config(
        model_path=Path(config_dict["model_path"]),
        cache_dir=config_dict.get("cache_dir", "/home/cache/"),
        gpu_memory_utilization=config_dict.get("gpu_memory_utilization", 0.8),
        enforce_eager=config_dict.get("enforce_eager", True),
        max_model_len=config_dict.get("max_model_len", 12288),
        data_path=Path(config_dict["data_path"]),
        id_col=config_dict.get("id_col", "id"),
        question_col=config_dict.get("question_col", "question"),
        use_context_col=config_dict.get("use_context_col", "none"),
        answer_col=config_dict.get("answer_col", "no_context_response"),
        critic_col=config_dict.get("critic_col", "no_context_response_critic_2shot"),
        prompt_type=config_dict.get("prompt_type", "critic"),
        few_shot_dir=config_dict.get("few_shot_dir", "few_shots"),
        result_col=config_dict.get("result_col", "model_outputs"),
        number_output_seq=config_dict.get("number_output_seq", 1),
        temperature=config_dict.get("temperature", 0.6),
        top_p=config_dict.get("top_p", 0.9),
        max_tokens=config_dict.get("max_tokens", 256),
        verbalized_top_k=config_dict.get("verbalized_top_k", 1),
    )


def clean_completion_text(text: str, marker: str = "assistant:") -> str:
    r"""Keeps only the content after the first occurrence of `marker` (case-insensitive).
    If the marker is not found, returns the original text stripped.

    Example:
      Input: "System: Some instructions\nAssistant: Here is the final answer\n"
      Output: "Here is the final answer"

    """
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    marker_lower = marker.lower()

    idx = text_lower.find(marker_lower)
    if idx == -1:
        # Marker not found => return everything
        return text_stripped

    # If found, skip everything before + the marker itself
    idx_end = idx + len(marker_lower)

    # Return what’s after "assistant:", stripping leading/trailing whitespace
    new_text = text_stripped[idx_end:]
    return new_text.strip()


def gather_item_prompts(data, prompt_function, id_key="question_idx"):
    """For each sample in `data`, calls `prompt_function(sample)` to get
    one or more prompt strings. Stores them in a structure along with the sample ID.

    Returns:
      A list of dicts, each dict has:
        {
          "id": <sample ID>,
          "prompts": [list_of_prompt_strings]
        }

    """
    grouped_data = []
    for idx, sample in enumerate(data):
        # Get the item ID (or fallback to loop index if none)
        group_id = sample.get(id_key, idx)

        # The prompt function can return a single string or multiple strings
        prompts = prompt_function(sample)
        if isinstance(prompts, str):
            prompts = [prompts]

        grouped_data.append({"id": group_id, "prompts": prompts})
    return grouped_data


def flatten_prompts(grouped_data):
    """Takes a list of items in the form:
      [
        {"id": ..., "prompts": [p1, p2, ...]},
        {"id": ..., "prompts": [p3, ...]},
        ...
      ]
    and flattens them into a single list of prompt strings.

    Returns:
      A list of all prompt strings in order.

    """
    all_prompts = []
    for item in grouped_data:
        all_prompts.extend(item["prompts"])
    return all_prompts


def unflatten_results(grouped_data, generation_results):
    current_index = 0
    for item in grouped_data:
        num_prompts = len(item["prompts"])
        item_results = []

        local_gen = generation_results[current_index : current_index + num_prompts]
        for i, gen_result in enumerate(local_gen):
            # Just keep the text after "Assistant:"
            completions = [clean_completion_text(out.text) for out in gen_result.outputs]

            item_results.append(completions)

        item["completions"] = item_results
        current_index += num_prompts

    return grouped_data


def generate_for_dataset(model, data, prompt_function, sampling_params, id_key="id"):
    """High-level function that:
      1) Gathers prompts from each item in `data`.
      2) Flattens all prompts into a single list for batched generation.
      3) Calls model.generate(...) once on that flattened list.
      4) Unflattens the results back into each item's dictionary.
      5) Returns the final list of items, each with
         "id" and "prompts_and_completions".

    Structure of returned value:
      [
        {
          "id": <sample_id>,
          "prompts_and_completions": [
            {
              "prompt": <string?>,
              "completions": [list_of_generation_outputs]
            },
            ...
          ]
        },
        ...
      ]
    """
    # 1) Gather prompts from each item
    grouped_data = gather_item_prompts(data, prompt_function, id_key=id_key)

    # 2) Flatten prompts
    all_prompts = flatten_prompts(grouped_data)

    # 3) Generate in one batch call
    generation_results = model.generate(all_prompts, sampling_params, use_tqdm=True)

    # 4) Unflatten the generation results back
    final_data = unflatten_results(grouped_data, generation_results)

    return final_data


def store_generation_results(dataset_split, results, result_col="model_outputs", id_col="id"):
    """Merges the generation results back into the dataset split, storing them in a new column.

    Args:
      dataset_split (Dataset): A Hugging Face dataset split (e.g., data["train"]).
      results (list): A list of dicts, each with:
          {
            "id": <some_id>,
            "prompts_and_completions": [...]
          }
      result_col (str): The name of the new column to create in the dataset.
      id_col (str): The name of the column used to match rows to results["id"].

    Returns:
      The updated dataset_split (in memory).
      NOTE: If you want to save it to disk, call dataset_split.save_to_disk(<some_new_path>).

    """
    # Build a map from item_id -> prompts_and_completions
    # so we can quickly retrieve results for each row by ID.
    id2completions = {item["id"]: item["completions"] for item in results}

    def map_function(example):
        """For each row in the dataset, store the matching
        prompts_and_completions in the new column.
        If there's no match, store None.
        """
        example_id = example[id_col]
        example[result_col] = id2completions.get(example_id)

        # unflatten list
        if len(example[result_col]) == 1:
            example[result_col] = example[result_col][0]
        return example

    # We apply map row by row (batched=False) for clarity
    updated_split = dataset_split.map(map_function, batched=False)
    return updated_split
