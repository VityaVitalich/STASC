import json
import tempfile
from multiprocessing import Queue
from pathlib import Path
from typing import Any, Optional

import torch
from encourage.llm import Response, ResponseWrapper
from encourage.prompts import PromptCollection
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import Config


def execute_llm_call(
    cfg: Config,
    model_path: str,
    prompts: Any,
    sampling_params: SamplingParams,
    queue: Queue,
) -> None:
    """Executes an LLM call and returns the responses."""
    model = init_model(cfg, model_path=model_path)
    responses = generate_responses(cfg, prompts, model, sampling_params)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_out:
        out_path = Path(tmp_out.name)
    with open(out_path, "w") as f:
        responses = [response.to_dict() for response in responses.response_data]
        json.dump(responses, f)
    queue.put(out_path)


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


def transform_json_to_responses(responses_path: str) -> ResponseWrapper:
    with open(responses_path, "r") as f:
        json_file = json.load(f)
    responses = []
    for i in range(0, len(json_file)):
        response = {key: value for key, value in json_file[i].items() if key != "processing_time"}
        response = Response.from_dict(response)
        responses.append(response)
    responses = ResponseWrapper(responses=responses)
    return responses
