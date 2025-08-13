import torch
from encourage.llm import ResponseWrapper
from encourage.prompts import PromptCollection
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from configs.config import Config


def generate_responses(
    cfg: Config,
    prompt_collection: PromptCollection,
    model: LLM,
    sampling_params: SamplingParams,
) -> ResponseWrapper:
    """Generates responses for a dataset using the prompt builder and sampling parameters."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)

    reformatted_prompts = [
        tokenizer.apply_chat_template(
            prompt.conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        for prompt in prompt_collection
    ]
    request_outputs = model.generate(reformatted_prompts, sampling_params=sampling_params)

    responses = ResponseWrapper.from_request_output(request_outputs, prompt_collection)  # type: ignore

    for response in responses.response_data:
        response.response = response.response.strip().lower()

    return responses


def init_model(cfg: Config) -> LLM:
    """Initializes the model with the given configuration."""
    return LLM(
        model=cfg.model.model_path,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        enforce_eager=cfg.model.enforce_eager,
        max_model_len=cfg.model.max_model_len,
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
    )
