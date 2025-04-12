import sys

sys.path.append('../')
sys.path.append('./')

import argparse
import yaml
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, List

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from peft import get_peft_model, TaskType, LoraConfig
from utils.generation_utils import load_config
from utils.ft_utils import (
    load_hf_datasets,
    encode_with_messages_format,
    encode_with_prompt_completion_format,
    encode_with_messages_format_chat_template
)
import os

IGNORE_INDEX = -100


# ---------------------------
# 1) Define Data Classes
# ---------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = False
    torch_dtype: Optional[str] = None
    model_revision: str = "main"
    token: Optional[str] = None


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    block_size: Optional[int] = 1024
    validation_split_percentage: Optional[int] = 5
    dataset_percentage: Optional[int] = 100
    seed: int = 42
    streaming: bool = False
    overwrite_cache: bool = False
    preprocessing_num_workers: Optional[int] = None
    load_from_disk: bool = False


@dataclass
class LoRAArguments:
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=list)
    dora: bool = False


# ---------------------------
# 2) Helper Functions
# ---------------------------

def build_model_args(model_cfg: dict) -> ModelArguments:
    return ModelArguments(
        model_name_or_path=model_cfg["model_name_or_path"],
        tokenizer_name=model_cfg.get("tokenizer_name"),
        cache_dir=model_cfg.get("cache_dir"),
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast_tokenizer=model_cfg.get("use_fast_tokenizer", False),
        torch_dtype=model_cfg.get("torch_dtype"),
    )


def build_data_args(data_cfg: dict) -> DataTrainingArguments:
    return DataTrainingArguments(
        dataset_name=data_cfg.get("dataset_name"),
        dataset_config_name=data_cfg.get("dataset_config_name"),
        block_size=data_cfg.get("block_size", 1024),
        validation_split_percentage=data_cfg.get("validation_split_percentage", 5),
        dataset_percentage=data_cfg.get("dataset_percentage", 100),
        seed=data_cfg.get("seed", 42),
        streaming=data_cfg.get("streaming", False),
        overwrite_cache=data_cfg.get("overwrite_cache", False),
        preprocessing_num_workers=data_cfg.get("preprocessing_num_workers"),
        load_from_disk=data_cfg.get("load_from_disk", False),
    )


def build_training_args(training_cfg: dict) -> TrainingArguments:
    """
    Build TrainingArguments for HF Trainer, including FSDP config if present.
    """
    return TrainingArguments(
        output_dir=training_cfg["output_dir"],
        overwrite_output_dir=True,
        learning_rate=training_cfg["learning_rate"],
        num_train_epochs=training_cfg["num_train_epochs"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
        max_steps=training_cfg.get("max_steps", -1),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 500),
        evaluation_strategy=training_cfg.get("evaluation_strategy", "steps"),
        eval_steps=training_cfg.get("eval_steps", 500),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
        warmup_steps=training_cfg.get("warmup_steps", 0),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        logging_steps=training_cfg.get("logging_steps", 10),
        do_train=training_cfg.get("do_train", True),
        do_eval=training_cfg.get("do_eval", True),
        report_to=training_cfg.get("report_to", ["none"]),
        run_name=training_cfg.get("run_name", "experiment"),
        remove_unused_columns=True,
    )


def build_lora_args(peft_cfg: dict) -> LoRAArguments:
    return LoRAArguments(
        use_lora=peft_cfg.get("use_lora", False),
        lora_rank=peft_cfg.get("lora_rank", 8),
        lora_alpha=peft_cfg.get("lora_alpha", 16),
        lora_dropout=peft_cfg.get("lora_dropout", 0.1),
        lora_target_modules=peft_cfg.get("lora_target_modules", []),
        dora=peft_cfg.get("dora", False),
    )


# ---------------------------
# 3) The Main Training Logic
# ---------------------------
def run_train(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    lora_args: LoRAArguments,
):
    """
    The main fine-tuning routine using HF Trainer + FSDP via accelerate.
    """
    # 1) Convert torch_dtype string
    if model_args.torch_dtype == "auto":
        dtype = "auto"
    elif model_args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16

    # 2) Load model
    print(f"[INFO] Loading Model at {model_args.model_name_or_path}")

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype,
    )

    # 3) Potentially wrap model in LoRA
    if lora_args.use_lora:
        print(f"[INFO] Adding LoRA with R {lora_args.lora_rank}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_args.lora_rank,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            init_lora_weights=True,
            #use_dora=lora_args.dora
        )
        model = get_peft_model(model, lora_config)

    # 4) Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )

    # Ensure PAD token exists
    if tokenizer.pad_token is None:
        # for llama model starting from 3 version
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer.pad_token_id = 128004
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    # 5) Load dataset
    print(f"[INFO] Loading Dataset from at {data_args.dataset_name}")
    raw_datasets = load_hf_datasets(data_args)

    # 6) Tokenize dataset
    train_cols = raw_datasets["train"].column_names
    if "prompt" in train_cols and "completion" in train_cols:
        encode_function = lambda ex: encode_with_prompt_completion_format(
            ex, tokenizer, max_seq_length=data_args.block_size
        )
    elif "messages" in train_cols:
        encode_function = lambda ex: encode_with_messages_format_chat_template(
            ex, tokenizer, config.architectures[0]
        )
    else:
        raise ValueError(
            "No matching columns found. Please have either 'prompt'/'completion' or 'messages' in your dataset."
        )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[
            col for col in train_cols
            if col not in ["input_ids", "labels", "attention_mask", "position_ids"]
        ],
        desc="Tokenizing and reformatting instruction data",
        load_from_cache_file=False
    )

    lm_datasets.set_format(type="pt")
    # Filter out any examples that are all -100
    lm_datasets = lm_datasets.filter(lambda x: (x["labels"] != -100).any(), load_from_cache_file=False)

    # 7) Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest"
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation", None)

    # 8) Create Trainer
    #  HF Trainer automatically integrates with Accelerate under the hood
    #  when you run "accelerate launch" on this script.

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9) Train
    trainer.train()
    trainer.save_model(training_args.output_dir)



# ---------------------------
# 4) The main() function
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Path to the YAML config file", required=True)
    args = parser.parse_args()

    # 1) Load YAML config
    config_dict = load_config(args.config_path)

    model_cfg = config_dict.get("model", {})
    data_cfg = config_dict.get("data", {})
    training_cfg = config_dict.get("training", {})
    peft_cfg = config_dict.get("peft", {})

    # 2) Build argument dataclasses
    model_args = build_model_args(model_cfg)
    data_args = build_data_args(data_cfg)
    training_args = build_training_args(training_cfg)
    lora_args = build_lora_args(peft_cfg)

    os.environ["WANDB_PROJECT"] = config_dict['training']['project_name']
    os.environ["WANDB_DIR"] = config_dict['model']['cache_dir']
    os.environ["WANDB_CACHE_DIR"] = config_dict['model']['cache_dir']



    # 3) Run training
    run_train(model_args, data_args, training_args, lora_args)


if __name__ == "__main__":
    main()
