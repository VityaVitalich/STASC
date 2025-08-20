from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Algo:
    name: str = ""

    use_cot: bool = False
    use_init_context: bool = False
    num_documents: int = 0
    use_corr_context: bool = False

    num_iterations: int = 0

    fixed_initialization: bool = True
    improving_filter: bool = False
    fixed_fine_tuning: bool = True
    number_corrections: int = 3

    finalize_judgement: bool = False
    num_refine_iterations: int = 5


@dataclass
class Dataset:
    # Required fields
    task_type: str = ""
    data_path: str = ""
    split: str = ""
    id_col: str = ""
    question_col: str = ""
    gold_col: str = ""
    context_col: str = ""
    evaluator_mode: str = ""
    evaluator_function: str = ""
    evaluator_answer_marker: str = ""

    # Optional fields and defaults
    use_context_col: Optional[str] = None
    answer_col: str = "no_context_response"  # Column containing existing answers
    critic_col: str = "no_context_response_critic_2shot"  # Column with criticisms
    verbalized_top_k: int = 1  # top-k for verbalized UE 1S
    prompt_type: str = "critic"  # Options: generate, critic, revise, etc.
    few_shot_dir: str = "few_shots"  # Directory containing few-shot JSON files
    result_col: str = "model_outputs"  # New column name to store generation results
    number_output_seq: int = 1  # Number of sequences to generate per prompt


@dataclass
class Model:
    model_path: str
    model_name_short: str
    cache_dir: str
    gpu_memory_utilization: float
    enforce_eager: bool
    max_model_len: int
    random_seed: int
    temperature: float
    top_p: float
    max_tokens: int
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = False
    torch_dtype: Optional[str] = None
    model_revision: str = "main"


@dataclass
class DataTrainingArguments:
    block_size: Optional[int] = 1024
    validation_split_percentage: Optional[int] = 5
    dataset_percentage: Optional[int] = 100
    seed: int = 42
    streaming: bool = False
    preprocessing_num_workers: Optional[int] = 4
    load_from_disk: bool = False
    learning_rate: float = 1.0e-6
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    max_steps: int = -1
    save_strategy: str = "no"
    save_steps: int = 1
    evaluation_strategy: str = "no"
    eval_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    warmup_steps: int = 0
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    do_train: bool = True
    do_eval: bool = False


@dataclass
class LoRAArguments:
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=list)
    dora: bool = False


@dataclass
class Config:
    """Configuration class for all runs."""

    algo: Algo
    dataset: Dataset
    model: Model
    training: DataTrainingArguments
    lora: LoRAArguments

    run_name: str = ""
    accelerate_config_path: str = "config/accelerate.yaml"
