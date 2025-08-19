import contextlib
import gc

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

from configs.config import Config


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
        response.response = response.response.strip().lower()

    return responses


def init_model(cfg: Config) -> LLM:
    """Initializes the model with the given configuration."""
    print("Initializing model from Path:", cfg.model.model_path)
    return LLM(
        model=cfg.model.model_path,
        # model="models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
        # download_dir="/ltstorage/home/strich/STASC/model/",
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        enforce_eager=cfg.model.enforce_eager,
        max_model_len=cfg.model.max_model_len,
        dtype="bfloat16",
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
