from itertools import chain

import torch
from datasets import load_dataset, load_from_disk


def load_hf_datasets(data_args):
    """Loads or creates a dataset from either disk or Hugging Face Hub.
    Optionally creates a validation split if `validation_split_percentage` > 0
    and a validation set does not already exist.
    Also subsamples the dataset to `dataset_percentage` if < 100.
    """
    raw_datasets = {}
    # 1. Load the dataset (either from disk or from HF Hub)
    if data_args.dataset_name is not None:
        if data_args.load_from_disk:
            raw_datasets = load_from_disk(data_args.dataset_name)
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                streaming=data_args.streaming,
            )
    else:
        raise ValueError("`data_args.dataset_name` must be provided.")

    # 2. Optionally create a validation split if percentage > 0 and doesn't exist
    if data_args.validation_split_percentage > 0 and "validation" not in raw_datasets:
        split_pct = data_args.validation_split_percentage
        raw_datasets["validation"] = load_dataset(  # type: ignore
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{split_pct}%]",
            streaming=data_args.streaming,
        )
        raw_datasets["train"] = load_dataset(  # type: ignore
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{split_pct}%:]",
            streaming=data_args.streaming,
        )

    # 3. Optionally downsample the dataset to `data_args.dataset_percentage`
    if data_args.dataset_percentage < 100:
        dataset_frac = data_args.dataset_percentage / 100.0
        # Subsample train
        dataset_parts = raw_datasets["train"].train_test_split(  # pyright: ignore[reportAttributeAccessIssue]
            train_size=dataset_frac, seed=data_args.seed
        )
        raw_datasets["train"] = dataset_parts["train"]  # type: ignore

        # Subsample validation only if it exists
        if "validation" in raw_datasets:
            dataset_parts = raw_datasets["validation"].train_test_split(  # pyright: ignore[reportAttributeAccessIssue]
                test_size=dataset_frac, seed=data_args.seed
            )
            raw_datasets["validation"] = dataset_parts["test"]  # type: ignore

    return raw_datasets


def tokenize_datasets(data_args, raw_datasets, tokenizer):
    dataset_type = list(raw_datasets.keys())[0]
    column_names = list(raw_datasets[dataset_type].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output

    if not data_args.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    return tokenized_datasets


def format_datasets(data_args, tokenized_datasets, tokenizer):
    block_size = min(data_args.block_size, tokenizer.model_max_length)
    print(block_size)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    if not data_args.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )

    return lm_datasets


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


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += (
                    "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                )
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

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
