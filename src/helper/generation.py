import contextlib
import gc
from typing import Optional

import ray
import torch
from encourage.llm import ResponseWrapper
from encourage.prompts import PromptCollection
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)

from config import Config


def generate_responses(
    cfg: Config,
    prompt_collection: PromptCollection,
    model: LLM,
    sampling_params: SamplingParams,
) -> ResponseWrapper:
    """Generates responses for a dataset using the prompt builder and sampling parameters."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)

    reformatted_prompts = []
    for prompt in prompt_collection:
        reformatted_prompt = tokenizer.apply_chat_template(
            prompt.conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt.meta_data["reformatted_prompt"] = reformatted_prompt
        reformatted_prompts.append(reformatted_prompt)

    request_outputs = model.generate(reformatted_prompts, sampling_params=sampling_params)

    responses = ResponseWrapper.from_request_output(request_outputs, prompt_collection)  # type: ignore

    for response in responses.response_data:
        if isinstance(response.response, list):
            response.response = [
                r.strip().lower() if isinstance(r, str) else r for r in response.response
            ]
        elif isinstance(response.response, str):
            response.response = response.response.strip().lower()

    return responses


def init_model(cfg: Config, model_path: Optional[str] = None) -> LLM:
    """Initializes the model with the given configuration."""
    final_model_path = model_path if model_path else cfg.model.model_path

    print("Initializing model from Path:", final_model_path)
    return LLM(
        model=final_model_path,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        enforce_eager=cfg.model.enforce_eager,
        max_model_len=cfg.model.max_model_len,
        seed=cfg.model.random_seed,
        dtype=cfg.model.torch_dtype,
        tensor_parallel_size=torch.cuda.device_count(),
    )


def unload_model(model: LLM) -> None:
    # Delete the llm object and free the memory
    destroy_model_parallel()
    destroy_distributed_environment()
    del model.llm_engine.model_executor
    del model
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory.")
