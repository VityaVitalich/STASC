import argparse
import shutil
from functools import partial
from pathlib import Path

import datasets
import torch
from datasets import Dataset
from vllm import LLM, SamplingParams

# Local imports
from prompts.prompt_schemas import load_few_shot_prompts, prompt_mapper
from utils.generation_utils import (
    extract_parameters,
    generate_for_dataset,
    load_config,
    store_generation_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate with optional few-shot examples using a config file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    # 1) Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config_dict = load_config(config_path)

    # 2) Extract and validate parameters using the Config data class
    config = extract_parameters(config_dict)

    if config.prompt_type not in prompt_mapper:
        valid_types = list(prompt_mapper.keys())
        raise ValueError(
            f"Unknown prompt_type='{config.prompt_type}'. Must be one of: {valid_types}"
        )

    # 3) Load dataset
    data = datasets.load_from_disk(str(config.data_path))
    train_data: Dataset = data["train"]  # type: ignore
    test_data: Dataset = data["test"]  # type: ignore

    # Basic column existence checks for question_col, id_col, etc.
    for col in [config.question_col, config.id_col]:
        if col not in train_data.column_names:
            raise ValueError(f"Column '{col}' not found in train split.")
        if col not in test_data.column_names:
            raise ValueError(f"Column '{col}' not found in test split.")

    # Check context column if it's not "none"
    if config.use_context_col != "none":
        for split, split_name in [(train_data, "train"), (test_data, "test")]:
            if config.use_context_col not in split.column_names:
                raise ValueError(
                    f"Context column '{config.use_context_col}' not found in {split_name} split."
                )

    # 5) Load few-shot prompts from few_shot_dir and prompt_type
    few_shot_prompts = load_few_shot_prompts(config.few_shot_dir, config.prompt_type)

    # 6) Load model
    model = LLM(
        str(config.model_path),
        download_dir=config.cache_dir,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config.gpu_memory_utilization,
        enforce_eager=config.enforce_eager,
        max_model_len=config.max_model_len,
    )
    tokenizer = model.get_tokenizer()

    # 7) Create sampling params
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
        skip_special_tokens=False,
        n=config.number_output_seq,
    )

    # 8) Create prompt function with partial binding
    prompt_func = partial(
        prompt_mapper[config.prompt_type],
        tokenizer=tokenizer,
        use_context_col=config.use_context_col,
        question_col=config.question_col,
        answer_col=config.answer_col,
        critic_col=config.critic_col,
        few_shot_prompts=few_shot_prompts,  # loaded from file or None
        k=config.verbalized_top_k,
    )

    # 9) Generate for train split
    print("[INFO] Generating for train split")
    train_results = generate_for_dataset(
        model=model,
        data=train_data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=config.id_col,  # Or whichever key holds your unique IDs
    )

    # Insert results into a new column
    train_data = store_generation_results(
        dataset_split=train_data,
        results=train_results,
        result_col=config.result_col,
        id_col=config.id_col,
    )

    # 10) Generate for test split
    print("[INFO] Generating for test split")
    test_results = generate_for_dataset(
        model=model,
        data=test_data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=config.id_col,
    )

    # Insert results into a new column
    test_data = store_generation_results(
        dataset_split=test_data,
        results=test_results,
        result_col=config.result_col,
        id_col=config.id_col,
    )

    # Overwrite in place
    new_ds = datasets.DatasetDict({"train": train_data, "test": test_data})
    output_path = str(config.data_path)
    temp_path = f"{output_path}_temp"
    # Save the modified dataset
    new_ds.save_to_disk(temp_path)
    shutil.rmtree(output_path)
    shutil.move(temp_path, output_path)

    # # 1) Remove the old folder
    # shutil.rmtree(output_path, ignore_errors=True)
    # # 2) Build a new DatasetDict with updated train/test
    # new_ds = datasets.DatasetDict({"train": train_data, "test": test_data})
    # # 3) Save back to the same folder
    # new_ds.save_to_disk(str(output_path))
    print(f"[INFO] Overwrote dataset at {output_path}")


if __name__ == "__main__":
    main()
