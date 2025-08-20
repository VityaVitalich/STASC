import logging
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model  # pyright: ignore[reportPrivateImportUsage]
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
)

from config import Config
from finetune.utils import (
    build_training_args,
    encode_with_messages_format_chat_template,
    encode_with_prompt_completion_format,
    load_hf_datasets,
)

config_path = str((Path(__file__).parents[2] / "config").resolve())

logger = logging.getLogger(__name__)


# ---------------------------
# 3) The Main Training Logic
# ---------------------------
def run_train(cfg: Config, iteration: int) -> None:
    """The main fine-tuning routine using HF Trainer + FSDP via accelerate."""
    # 1) Convert torch_dtype string
    dtype = None
    if cfg.model.torch_dtype == "auto":
        dtype = "auto"
    elif cfg.model.torch_dtype == "bfloat16":
        dtype = torch.bfloat16

    # 2) Load model
    print(f"[INFO] Loading Model at {cfg.model.model_path}")

    config = AutoConfig.from_pretrained(cfg.model.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_path,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=dtype,
    )

    # 3) Potentially wrap model in LoRA
    if cfg.lora.use_lora:
        print(f"[INFO] Adding LoRA with R {cfg.lora.lora_rank}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.lora.lora_rank,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.lora_target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)

    # 4) Load tokenizer
    tokenizer_kwargs = {
        "use_fast": cfg.model.use_fast_tokenizer,
        "revision": cfg.model.model_revision,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path, **tokenizer_kwargs)

    # Ensure PAD token exists
    if tokenizer.pad_token is None:
        # for llama model starting from 3 version
        if "llama" in cfg.model.model_path.lower():
            tokenizer.pad_token_id = 128004
            tokenizer.pad_token = "<|finetune_right_pad_id|>"
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            model.resize_token_embeddings(len(tokenizer))

    # 5) Load dataset
    print(f"[INFO] Loading Dataset from at {cfg.dataset.data_path}")
    raw_datasets: DatasetDict = load_hf_datasets(
        data_args=cfg.training, dataset_path=cfg.dataset.data_path
    )

    # 6) Tokenize dataset
    train_cols = raw_datasets["train"].column_names
    if "prompt" in train_cols and "completion" in train_cols:

        def encode_function(ex):
            return encode_with_prompt_completion_format(
                ex, tokenizer, max_seq_length=cfg.training.block_size
            )
    elif "messages" in train_cols:

        def encode_function(ex):
            architecture = (
                config.architectures[0]
                if config.architectures and len(config.architectures) > 0
                else None
            )
            return encode_with_messages_format_chat_template(ex, tokenizer, architecture)
    else:
        raise ValueError(
            "No matching columns found. Please have either 'prompt'/'completion' or 'messages' in your dataset."
        )

    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=cfg.training.preprocessing_num_workers,
        remove_columns=[
            col
            for col in train_cols
            if col not in ["input_ids", "labels", "attention_mask", "position_ids"]
        ],
        desc="Tokenizing and reformatting instruction data",
        load_from_cache_file=False,
    )

    lm_datasets.set_format(type="pt")
    # Filter out any examples that are all -100
    lm_datasets = lm_datasets.filter(
        lambda x: (x["labels"] != -100).any(), load_from_cache_file=False
    )
    logger.info(f"Filtered dataset size: {lm_datasets.num_rows}")

    # 7) Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation", None)

    # 8) Create Trainer
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg.training),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # 9) Train
    trainer.train()
    trainer.save_model(f"model/{cfg.model.model_name_short}_{iteration}")
