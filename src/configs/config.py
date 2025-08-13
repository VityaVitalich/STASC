from dataclasses import dataclass
from typing import Optional


@dataclass
class Algo:
    name: str = ""
    use_cot: bool = False
    run_name_specification: str = ""
    use_init_context: bool = False
    num_documents: int = 0
    use_corr_context: bool = False
    num_star_iterations: int = 0

    initial_answer_with_new_model: bool = True
    only_better_correction: bool = True
    train_from_initial_model: bool = True

    number_output_initial_generations: int = 1
    number_output_corrections: int = 1

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


@dataclass
class Config:
    """Configuration class for all runs."""

    model: Model
    dataset: Dataset
    algo: Algo

    base_url: str = ""
    vllm_port: int = 18123
    run_name: str = ""
