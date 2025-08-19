import torch
from datasets import DatasetDict, load_from_disk
from transformers import TrainingArguments

from configs.config import DataTrainingArguments


def load_hf_datasets(data_args: DataTrainingArguments, dataset_path: str) -> DatasetDict:
    """Loads or creates a dataset from either disk or Hugging Face Hub.
    Optionally creates a validation split if `validation_split_percentage` > 0
    and a validation set does not already exist.
    Also subsamples the dataset to `dataset_percentage` if < 100.
    """
    dataset = load_from_disk(dataset_path)
    raw_datasets = {"train": dataset}

    # 2. Optionally create a validation split if percentage > 0 and doesn't exist
    if data_args.validation_split_percentage > 0 and "validation" not in raw_datasets:
        train_test = dataset.train_test_split(
            test_size=data_args.validation_split_percentage, seed=data_args.seed
        )
        raw_datasets["train"] = train_test["train"]
        raw_datasets["validation"] = train_test["test"]

    # 3. Optionally downsample the dataset to `data_args.dataset_percentage`
    # if data_args.dataset_percentage < 100:
    #     dataset_frac = data_args.dataset_percentage / 100.0
    #     # Subsample train
    #     dataset_parts = raw_datasets["train"].train_test_split(
    #         train_size=dataset_frac, seed=data_args.seed
    #     )
    #     raw_datasets["train"] = dataset_parts["train"]

    #     # Subsample validation only if it exists
    #     if "validation" in raw_datasets:
    #         dataset_parts = raw_datasets["validation"].train_test_split(
    #             test_size=dataset_frac, seed=data_args.seed
    #         )
    #         raw_datasets["validation"] = dataset_parts["test"]

    return DatasetDict(raw_datasets)


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith(
        (" ", "\n", "\t")
    ):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format_chat_template(example, tokenizer, architecture):
    """Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncate=True,
        return_tensors="pt",
        enable_thinking=False,
    ).squeeze(0)

    special_token = 0
    model_name = "unknown"
    if "qwen" in architecture.lower():
        model_name = "qwen"
        special_token = 151644
        # qwen special token is <im_start>, however it also appends assistant and \n
    if "phi" in architecture.lower():
        model_name = "phi"
        # phi special token is <assistant>
        special_token = 32001
    if "llama" in architecture.lower():
        model_name = "llama"
        special_token = 128006
        # llama special start token is <|start_header_id|>, however it also appends assistant and end_header_id and \n\n
        # FIX FOR 3.1. Do not add generation prompt at the end
        input_ids = input_ids[:-4]

    # find the special tokens for starting assistance generation
    special_assistant_start_indices = (input_ids == special_token).nonzero(as_tuple=True)[0]
    # take last one and + 1 for indexing in python
    last_assistant = special_assistant_start_indices[-1] + 1

    if model_name == "qwen":
        # so we would not count 2 additional tokens (assistant, \n, as they are appended at the beggining and is part of template)
        last_assistant += 2
    if model_name == "llama":
        last_assistant += 3

    # Generate labels: -100 for tokens before pre-last EOS, rest as input_ids
    labels = torch.full_like(input_ids, fill_value=-100)
    labels[last_assistant:] = input_ids[last_assistant:]

    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_training_args(training_cfg: DataTrainingArguments) -> TrainingArguments:
    """Build TrainingArguments for HF Trainer, including FSDP config if present."""
    return TrainingArguments(
        overwrite_output_dir=True,
        learning_rate=training_cfg.learning_rate,
        num_train_epochs=training_cfg.num_train_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        max_steps=training_cfg.max_steps,
        save_strategy=training_cfg.save_strategy,
        save_steps=training_cfg.save_steps,
        eval_strategy=training_cfg.evaluation_strategy,
        eval_steps=training_cfg.eval_steps,
        weight_decay=training_cfg.weight_decay,
        warmup_ratio=training_cfg.warmup_ratio,
        warmup_steps=training_cfg.warmup_steps,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        logging_steps=training_cfg.logging_steps,
        do_train=training_cfg.do_train,
        do_eval=training_cfg.do_eval,
        remove_unused_columns=True,
        bf16=True,
        report_to=["mlflow"],
    )
