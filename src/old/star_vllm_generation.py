import argparse
from functools import partial

import datasets
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluation.eval_utils import has_answer
from helper.generation import (
    generate_for_dataset,
    load_config,
    store_generation_results,
)
from old.star_prompts import star_rationale_generation_prompt, star_rationalization_prompt
from prompts.prompt_schemas import load_few_shot_prompts


def split_rationale_and_final_answer(generated_text: str):
    """Splits a STaR-style generation into two parts:
      1) The rationale (everything after 'Step-by-step reasoning:' until 'Final Answer:')
      2) The final answer (everything after 'Final Answer:')
    If either marker is missing, we do a best-effort parse.
    """
    rationale_marker = "Step-by-step reasoning:"
    answer_marker = "Final Answer:"
    rationale = ""
    final_ans = ""

    text = generated_text.replace("\r", "")  # normalize newlines if desired

    # Locate the markers
    rationale_start_idx = text.find(rationale_marker)
    answer_start_idx = text.find(answer_marker)

    if rationale_start_idx != -1:
        rationale_start = rationale_start_idx + len(rationale_marker)
        if answer_start_idx != -1 and answer_start_idx > rationale_start:
            rationale = text[rationale_start:answer_start_idx].strip()
        else:
            # If there's no final_answer marker or it's out of order
            rationale = text[rationale_start:].strip()

    if answer_start_idx != -1:
        answer_start = answer_start_idx + len(answer_marker)
        final_ans = text[answer_start:].strip()

    return rationale, final_ans


def add_rationale_split(dataset, answers_col, rationale_col_name, final_ans_col_name):
    def split_rationales(example):
        answers = example[answers_col]

        rationales = []
        final_ans_ls = []
        for ans in answers:
            rationale, final_ans = split_rationale_and_final_answer(ans)
            rationales.append(rationale)
            final_ans_ls.append(final_ans)

        example[rationale_col_name] = rationales
        example[final_ans_col_name] = final_ans_ls

        return example

    return dataset.map(split_rationales)


def flatten_correct_rationales(
    dataset,
    question_col="question",
    reference_col="reference",
    # Generation columns
    gen_answer_col="star_generation_answer",
    gen_rationale_col="star_generation_rationale",
    # Rationalization columns
    rat_answer_col="star_rationalization_answer",
    rat_rationale_col="star_rationalization_rationale",
    id_col="id",
    output_path="temp_dataset",
):
    """For each row in dataset, we do:

    1) Check if there's at least one correct plain generation answer in gen_answer_col.
       - If yes, collect all (answer, rationale) pairs from generation that are correct.
         (We ignore rationalization.)
    2) Otherwise, collect all (answer, rationale) pairs from rationalization that are correct.

    We create a new row for EACH correct pair:
      {
        id, question, reference, rationale, answer, messages
      }

    Returns a brand-new HF Dataset with these flattened rows.
    """
    system_prompt = (
        "You are a helpful reasoning assistant. "
        "Please reason through the question step by step before giving a final answer."
    )

    instructions = (
        "Generate a short chain-of-thought rationale, and then provide the final answer.\n"
        "Step-by-step reasoning:\n"
        "Final Answer:\n"
    )

    # Prepare lists for final flattened data
    new_ids = []
    new_questions = []
    new_refs = []
    new_rationales = []
    new_messages = []

    # Iterate over each row
    for row in dataset:
        row_id = row[id_col]
        question = row[question_col]
        reference = row[reference_col]

        # 1) Retrieve generation answers/rationales
        gen_answers = row[gen_answer_col]
        gen_rationales = row[gen_rationale_col]

        # 2) Retrieve rationalization answers/rationales
        rat_answers = row[rat_answer_col]
        rat_rationales = row[rat_rationale_col]

        # 3) Check if generation has at least one correct
        gen_correct_indices = [i for i, ans in enumerate(gen_answers) if has_answer(reference, ans)]
        use_generation = len(gen_correct_indices) > 0

        if use_generation:
            # We only take the correct generation pairs
            for i in gen_correct_indices:
                new_ids.append(f"{row_id}_gen_{i}")
                new_questions.append(question)
                new_refs.append(reference)
                new_rationales.append(gen_rationales[i])

                user_question = (
                    f"Question:\n{question}\n\nReason step by step, then conclude with the answer."
                )
                rationale = gen_rationales[i]
                assistant_content = f"{rationale}\nFinal Answer: {reference[0]}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instructions},
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_content},
                ]
                new_messages.append(messages)
        else:
            # 4) If none correct in generation, we look at rationalization
            rat_correct_indices = [
                i for i, ans in enumerate(rat_answers) if has_answer(reference, ans)
            ]
            for i in rat_correct_indices:
                new_ids.append(f"{row_id}_rat_{i}")
                new_questions.append(question)
                new_refs.append(reference)
                new_rationales.append(rat_rationales[i])

                user_question = (
                    f"Question:\n{question}\n\nReason step by step, then conclude with the answer."
                )
                rationale = rat_rationales[i]
                assistant_content = f"{rationale}\nFinal Answer: {reference[0]}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instructions},
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_content},
                ]
                new_messages.append(messages)

    # Build the new dictionary
    flattened_data = {
        id_col: new_ids,
        question_col: new_questions,
        reference_col: new_refs,
        "rationale": new_rationales,
        "messages": new_messages,
    }

    # Convert to a new HF Dataset
    flattened_dataset = DatasetDict({"train": Dataset.from_dict(flattened_data)})
    print(flattened_dataset)

    flattened_dataset.save_to_disk(output_path)
    return flattened_dataset


def perform_generation(data, model, prompt_func, sampling_params, id_key, output_col):
    """Perform (rationale) generation or (rationalization) generation for the dataset.
    Store the generation results in the dataset under 'output_col'.
    """
    generation_results = generate_for_dataset(
        model=model,
        data=data,
        prompt_function=prompt_func,
        sampling_params=sampling_params,
        id_key=id_key,
    )
    return store_generation_results(data, generation_results, result_col=output_col, id_col=id_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--generation_model_path", type=str, required=True)
    parser.add_argument("--ft_dataset_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    iteration = args.iteration

    config = load_config(args.config_path)

    # Load dataset
    dataset = datasets.load_from_disk(str(config["data_path"]))
    train_data, test_data = dataset["train"], dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], cache_dir=config["cache_dir"])

    # few shots

    generation_few_shot_prompts = load_few_shot_prompts(config["few_shot_dir"], "star_generation")
    rationalization_few_shot_prompts = load_few_shot_prompts(
        config["few_shot_dir"], "star_rationalization"
    )

    # Prompt functions
    # 1) Rationale generation prompt (Step 3)
    rationale_prompt_func = partial(
        star_rationale_generation_prompt,
        tokenizer=tokenizer,
        question_col=config["question_col"],
        few_shot_prompts=generation_few_shot_prompts,
    )
    # 2) Rationalization prompt that includes the gold answer as a hint (Step 4)
    rationalization_prompt_func = partial(
        star_rationalization_prompt,
        tokenizer=tokenizer,
        question_col=config["question_col"],
        gold_col=config["gold_col"],
        few_shot_prompts=rationalization_few_shot_prompts,
    )

    # Initialize model (M0)
    model = LLM(
        args.generation_model_path,
        download_dir=config["cache_dir"],
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        max_model_len=config["max_model_len"],
        # disable_log_stats=True,  # Disables logging statistics
        # disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        n=config["number_output_seq"],
    )

    # (3) Generate rationales for all samples: (rhat_i, yhat_i) = M_{n-1}(x_i)
    # We'll store the predicted final answer in 'result_col' or a new column like "pred_answer"
    print(f"[INFO] Starting generation at {iteration + 1}/{config['num_star_iterations']}")

    train_data = perform_generation(
        data=train_data,
        model=model,
        prompt_func=rationale_prompt_func,
        sampling_params=sampling_params,
        id_key=config["id_col"],
        output_col=f"star_generation_{iteration}",  # store model's answer after rationale generation
    )
    train_data = add_rationale_split(
        train_data,
        answers_col=f"star_generation_{iteration}",
        rationale_col_name=f"star_generation_rationale_{iteration}",
        final_ans_col_name=f"star_generation_answer_{iteration}",
    )

    # (4) Rationalization for all samples, but the prompt includes the gold answer
    # We'll store this rationalized final answer in a new column, e.g., "rationalized_answer"
    print(f"[INFO] Starting rationalization at {iteration + 1}/{config['num_star_iterations']}")

    train_data = perform_generation(
        data=train_data,
        model=model,
        prompt_func=rationalization_prompt_func,
        sampling_params=sampling_params,
        id_key=config["id_col"],
        output_col=f"star_rationalization_{iteration}",
    )
    train_data = add_rationale_split(
        train_data,
        answers_col=f"star_rationalization_{iteration}",
        rationale_col_name=f"star_rationalization_rationale_{iteration}",
        final_ans_col_name=f"star_rationalization_answer_{iteration}",
    )

    flatten_correct_rationales(
        dataset=train_data,
        question_col=config["question_col"],
        reference_col=config["gold_col"],
        gen_answer_col=f"star_generation_answer_{iteration}",
        gen_rationale_col=f"star_generation_rationale_{iteration}",
        rat_answer_col=f"star_rationalization_answer_{iteration}",
        rat_rationale_col=f"star_rationalization_rationale_{iteration}",
        id_col=config["id_col"],
        output_path=args.ft_dataset_path,
    )


if __name__ == "__main__":
    main()
