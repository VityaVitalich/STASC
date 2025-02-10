# Paper Title

This repository contains code for paper SELF-TAUGHT SELF-CORRECTION FOR SMALL LANGUAGE MODELS

## 🚀 Running STaSC

To reproduce the experiments and run different versions of the **Generalized Self-Taught Self-Correction (G-STaSC) algorithm** as described in the paper, use the following command with the specified configuration files:

```bash
CUDA_VISIBLE_DEVICES=0,1 python star_correction.py \
  --config configs/self_correction_star_config.yaml \
  --ft_config configs/fine_tune.yaml \
  --accelerate_config_path configs/accelerate_config.yaml
```

The `self_correction_star_config.yaml` file, which defines the key parameters for self-correction, is detailed in [Section: Self-Correction Configuration](#self-correction-configuration).

The `fine_tune.yaml` file, which defines the key parameters for fine-tuning, is detailed in [Section: Fine-Tuning Configuration](#fine-tuning-configuration).

The `accelerate_config.yaml` file defines the parameters for distributed training. Please see [Accelerate Documentation](https://huggingface.co/docs/accelerate/index)



## Self Correction Configuration

### 📌 Model Configuration
| Parameter                | Value                                      | Description |
|--------------------------|--------------------------------------------|-------------|
| `model_path`             | `Qwen/Qwen2.5-1.5B-Instruct`           | Pretrained model checkpoint path. |
| `cache_dir`             | `/home/cache/`       | Path to the cache directory for storing model weights. |
| `gpu_memory_utilization` | `0.9`                                    | GPU memory allocation fraction (0.0 - 1.0). |
| `enforce_eager`         | `True`                                    | Whether to enforce eager execution for debugging. |
| `max_model_len`         | `8192`                                   | Maximum sequence length for the model. |

### 📊 Dataset Configuration
| Parameter         | Value                       | Description |
|------------------|---------------------------|-------------|
| `data_path`      | `data/datasets/s_nq`       | Path to the dataset used for QA evaluation. |
| `id_col`         | `question_id`              | Unique identifier column in the dataset. |
| `question_col`   | `question_text`            | Column containing the input question text. |
| `gold_col`       | `reference`                | Column containing the reference (gold) answer. |

### 🔁 Self-Correction Algorithm Parameters
| Parameter                        | Value   | Description |
|----------------------------------|---------|-------------|
| `num_star_iterations`            | `10`     | Number of self-correction iterations. |
| `run_name`                       | `test`  | Name for the experiment run. |
| `initial_answer_with_new_model`  | `True`  | Whether to generate initial answers with \(M_0\) (True) or \(M_{n-1}\) (False). |
| `only_better_correction`         | `True`  | Whether to use strictly improving corrections (True) or allow neutral ones (False). |
| `train_from_initial_model`       | `True`  | Whether to fine-tune from \(M_0\) (True) or from \(M_{n-1}\) (False). |

### 📝 Generation Configuration
| Parameter                      | Value         | Description |
|--------------------------------|-------------|-------------|
| `few_shot_dir`                 | `few_shots` | Directory containing few-shot examples for prompting. |
| `number_output_initial_generations` | `5` | Number of initial generations per question. |
| `number_output_corrections`    | `5` | Number of correction generations per question. |

### 🎲 Sampling Parameters
| Parameter       | Value  | Description |
|---------------|--------|-------------|
| `temperature` | `0.6`  | Sampling temperature (higher = more randomness). |
| `top_p`       | `0.9`  | Nucleus (Top-p) sampling threshold. |
| `max_tokens`  | `256`  | Maximum number of tokens to generate. |


## 🔧 Fine-Tuning Configuration

This section describes the fine-tuning configuration. The configuration is divided into **model setup, dataset handling, training parameters, and parameter-efficient fine-tuning (PEFT) options**.

### 📌 Model Configuration
| Parameter            | Value                                      | Description |
|----------------------|------------------------------------------|-------------|
| `model_name_or_path` | `Qwen/Qwen2.5-1.5B-Instruct`     | Pretrained model checkpoint path. |
| `tokenizer_name`     | `null`                                   | Uses the default tokenizer of the model. |
| `cache_dir`         | `/home/cache/`       | Path to store cached model weights. |
| `trust_remote_code`  | `true`                                   | Allows execution of remote model code. |
| `use_fast_tokenizer` | `true`                                   | Uses a fast tokenizer for efficiency. |
| `torch_dtype`        | `"bfloat16"`                             | Data type for model execution. |

### 📊 Dataset Configuration
| Parameter                     | Value                  | Description |
|--------------------------------|------------------------|-------------|
| `dataset_name`                 | `data/datasets/s_nq`  | Path to the dataset. |
| `block_size`                   | `1024`                | NOT USED. |
| `validation_split_percentage`   | `0`                   | Percentage of dataset used for validation. |
| `dataset_percentage`            | `100`                 | Percentage of the dataset used for training. |
| `seed`                          | `42`                  | Random seed for reproducibility. |
| `streaming`                     | `false`               | Whether to load data in streaming mode. |
| `overwrite_cache`               | `false`               | Whether to overwrite dataset cache. |
| `preprocessing_num_workers`      | `4`                   | Number of workers for dataset preprocessing. |
| `load_from_disk`                | `true`                | Loads dataset from disk instead of re-downloading. |

### 🚀 Training Configuration
| Parameter                         | Value                      | Description |
|------------------------------------|----------------------------|-------------|
| `output_dir`                      | `./my-finetuned-llama-fsdp` | Directory to save the fine-tuned model. |
| `learning_rate`                   | `1.0e-5`                    | Learning rate for training. |
| `num_train_epochs`                | `1`                         | Number of training epochs. |
| `per_device_train_batch_size`      | `2`                         | Batch size per GPU for training. |
| `per_device_eval_batch_size`       | `2`                         | Batch size per GPU for evaluation. |
| `gradient_accumulation_steps`      | `1`                         | Steps to accumulate gradients before updating. |
| `gradient_checkpointing`           | `false`                     | Whether to enable gradient checkpointing. |
| `max_steps`                        | `-1`                        | Maximum training steps (-1 means max_steps is disabled). |
| `save_strategy`                    | `"no"`                      | Whether to save checkpoints. |
| `save_steps`                        | `1`                         | Step interval for saving checkpoints. |
| `evaluation_strategy`              | `"no"`                      | Whether to run evaluation during training. |
| `eval_steps`                        | `1`                         | Step interval for evaluation. |
| `weight_decay`                     | `0.1`                        | L2 weight regularization. |
| `warmup_ratio`                     | `0.03`                       | Ratio of warmup steps. |
| `lr_scheduler_type`                | `"cosine"`                   | Learning rate scheduling strategy. |
| `logging_steps`                    | `10`                         | Step interval for logging metrics. |
| `do_train`                         | `true`                       | Whether to perform training. |
| `do_eval`                          | `false`                      | Whether to perform evaluation. |
| `report_to`                        | `["wandb"]`                  | Logging destination (e.g., Weights & Biases). |
| `run_name`                         | `"test_STaR"`                | Name of the training run. |
| `project_name`                     | `"STaR"`                     | Project name for tracking. |

### 🏗️ Parameter-Efficient Fine-Tuning (PEFT) Configuration
| Parameter           | Value   | Description |
|---------------------|---------|-------------|
| `use_lora`         | `false` | Whether to enable LoRA for fine-tuning. |
| `lora_rank`        | `8`     | LoRA rank for low-rank adaptation. |
| `lora_alpha`       | `16`    | LoRA scaling factor. |
| `lora_dropout`     | `0.1`   | Dropout rate for LoRA layers. |
| `lora_target_modules` | `["query_key_value"]` | Target layers for LoRA adaptation. |
| `dora`             | `false` | Whether to enable DoRA (Decoupled LoRA). |

---
