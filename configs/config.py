from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Model Configuration
    model_path: Path

    # Dataset Configuration
    data_path: Path
    id_col: str = "id"  # Unique identifier column in the dataset
    question_col: str = "question"  # Column containing the question text
    use_context_col: str = "none"  # Column name for context or "none" if not used
    answer_col: str = "no_context_response"  # Column containing existing answers
    critic_col: str = "no_context_response_critic_2shot"  # Column with criticisms
    verbalized_top_k: int = 1  # top-k for verbalized UE 1S

    # Generation Configuration
    prompt_type: str = "critic"  # Options: generate, critic, revise, etc.
    few_shot_dir: str = "few_shots"  # Directory containing few-shot JSON files
    result_col: str = "model_outputs"  # New column name to store generation results
    number_output_seq: int = 1  # Number of sequences to generate per prompt

    # Sampling Parameters
    temperature: float = 0.6  # Sampling temperature
    top_p: float = 0.9  # Top-p (nucleus) sampling
    max_tokens: int = 256  # Maximum number of tokens to generate

    cache_dir: str = "/home/cache/"
    gpu_memory_utilization: float = 0.8  # GPU memory utilization (0.0 to 1.0)
    enforce_eager: bool = True  # Whether to enforce eager execution
    max_model_len: int = 12288  # Maximum model length
