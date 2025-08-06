import os

os.environ["VLLM_LOGGING_LEVEL"] = "FATAL"

import argparse
from functools import partial

import datasets
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompts.enum import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import RewardEvaluator
from utils.generation_utils import generate_for_dataset, load_config, store_generation_results
from utils.utils import KM, flatten_predictions


def collect_correction_stats(
    dataset,
    reward_function,
    question_col="question",
    reference_col="reference",
    inital_answer_col="star_correction_initial_generation",
    correction_col="star_correction",
):
    """Computes statistics on corrections as percentages:
    - Percentage of samples that went from Correct to Incorrect
    - Percentage of samples that remained Correct
    - Percentage of samples that improved from Incorrect to Correct.

    Returns:
        A dictionary with percentages for each category.

    """
    correct_to_incorrect = 0
    correct_to_correct = 0
    incorrect_to_correct = 0
    total_corrections = 0

    # Iterate over each row in the dataset
    for row in dataset:
        reference = row[reference_col]

        # Check initial correctness
        initial_answers = row[inital_answer_col]
        corrected_answers = row[correction_col]

        for init_answer in initial_answers:
            init_is_correct = reward_function(ground_truth=reference, model_answer=init_answer)

            for correction in flatten_predictions(corrected_answers):
                correction_is_correct = reward_function(
                    ground_truth=reference, model_answer=correction
                )

                total_corrections += 1

                if init_is_correct and not correction_is_correct:
                    correct_to_incorrect += 1
                elif init_is_correct and correction_is_correct:
                    correct_to_correct += 1
                elif not init_is_correct and correction_is_correct:
                    incorrect_to_correct += 1

    # Avoid division by zero
    if total_corrections == 0:
        stats = {
            "correct_to_incorrect": 0.0,
            "correct_to_correct": 0.0,
            "incorrect_to_correct": 0.0,
        }
    else:
        stats = {
            "correct_to_incorrect": (correct_to_incorrect / total_corrections) * 100,
            "correct_to_correct": (correct_to_correct / total_corrections) * 100,
            "incorrect_to_correct": (incorrect_to_correct / total_corrections) * 100,
        }

    return stats


def collect_improving_corrections(
    dataset,
    reward_function,
    prompt_builder,
    question_col="question",
    reference_col="reference",
    context_col="context",
    # Initial Answer column
    inital_answer_col="star_generation_answer",
    # Correction column
    correction_col="star_rationalization_answer",
    id_col="id",
    output_path="temp_dataset",
    strict_improvement=True,
    *kwargs,
):
    # Prepare lists for final flattened data
    new_ids = []
    new_questions = []
    new_refs = []
    new_corrections = []
    new_answers = []
    new_messages = []

    # Iterate over each row
    for row in dataset:
        row_id = row[id_col]
        question = row[question_col]
        reference = row[reference_col]
        all_context = row.get(context_col, [""])

        # 1) Retrieve generation answers/rationales
        for init_answer in row[inital_answer_col]:
            for correction in flatten_predictions(row[correction_col]):
                # 3) Check if there is an improvement
                init_is_correct = reward_function(ground_truth=reference, model_answer=init_answer)
                correction_is_correct = reward_function(
                    ground_truth=reference, model_answer=correction
                )

                if strict_improvement:
                    use_sample = correction_is_correct > init_is_correct
                else:
                    use_sample = init_is_correct and correction_is_correct

                if use_sample:
                    new_ids.append(f"{row_id}_gen")
                    new_questions.append(question)
                    new_refs.append(reference)
                    new_answers.append([init_answer])
                    new_corrections.append([correction])

                    messages = prompt_builder.build_correction_messages_with_final_answer(
                        question, init_answer, correction, all_context
                    )
                    new_messages.append(messages)

    # Build the new dictionary
    flattened_data = {
        id_col: new_ids,
        question_col: new_questions,
        reference_col: new_refs,
        inital_answer_col: new_answers,
        correction_col: new_corrections,
        "messages": new_messages,
    }

    print(f"[INFO] Filtered {len(new_ids)} Corrections")

    # Convert to a new HF Dataset
    flattened_dataset = DatasetDict({"train": Dataset.from_dict(flattened_data)})
    flattened_dataset.save_to_disk(output_path)

    print(f"[INFO] Saved filtered corrections to {output_path}")

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


# --- Generation Helper ---


def run_generation_phase(
    data_train,
    data_test,
    model,
    prompt_func,
    output_col,
    num_outputs,
    id_key,
    config,
    phase_label,
    sampling_params,
    **kwargs,
):
    """Sets the sampling parameter, performs generation on train and test splits,
    computes accuracy via KM, and prints results.
    Returns updated train and test datasets.
    """
    print(f"[INFO] Starting {phase_label} generation...")
    sampling_params.n = num_outputs
    new_train = perform_generation(
        data_train, model, prompt_func, sampling_params, id_key, output_col, **kwargs
    )
    sampling_params.n = 1
    new_test = perform_generation(
        data_test, model, prompt_func, sampling_params, id_key, output_col, **kwargs
    )

    return new_train, new_test


def branch_initial_generation(
    config,
    model,
    train_data,
    test_data,
    init_prompt_func,
    sampling_params,
    iteration,
    ft_dataset_path,
    reward_function,
):
    new_train, new_test = run_generation_phase(
        train_data,
        test_data,
        model,
        init_prompt_func,
        output_col="star_correction_initial_generation",
        num_outputs=config["number_output_initial_generations"],
        id_key=config["id_col"],
        config=config,
        phase_label="Initial Answer",
        sampling_params=sampling_params,
    )
    print(
        f"[INFO] Initial Train Accuracy at step {iteration}: {KM(new_train, target_col='star_correction_initial_generation', gt_col=config['gold_col'], evaluator=reward_function)}"
    )
    print(
        f"[INFO] Initial Test Accuracy at step {iteration}: {KM(new_test, target_col='star_correction_initial_generation', gt_col=config['gold_col'], evaluator=reward_function)}"
    )
    return new_train, new_test


def branch_correction_generation(
    config,
    model,
    train_data,
    test_data,
    corr_prompt_func,
    sampling_params,
    iteration,
    ft_dataset_path,
    reward_function,
    prompt_builder,
):
    new_train, new_test = run_generation_phase(
        train_data,
        test_data,
        model,
        corr_prompt_func,
        output_col=f"star_correction_{iteration}",
        num_outputs=config["number_output_corrections"],
        id_key=config["id_col"],
        config=config,
        phase_label="Correction",
        sampling_params=sampling_params,
    )
    print(
        f"[INFO] Correction Train Accuracy at step {iteration}: {KM(new_train, target_col=f'star_correction_{iteration}', gt_col=config['gold_col'], evaluator=reward_function)}"
    )
    print(
        f"[INFO] Correction Test Accuracy at step {iteration}: {KM(new_test, target_col=f'star_correction_{iteration}', gt_col=config['gold_col'], evaluator=reward_function)}"
    )
    collect_improving_corrections(
        dataset=new_train,
        question_col=config["question_col"],
        reference_col=config["gold_col"],
        inital_answer_col="star_correction_initial_generation",
        correction_col=f"star_correction_{iteration}",
        id_col=config["id_col"],
        output_path=ft_dataset_path,
        strict_improvement=config["only_better_correction"],
        reward_function=reward_function,
        prompt_builder=prompt_builder,
    )
    stats_train = collect_correction_stats(
        dataset=new_train,
        question_col=config["question_col"],
        reference_col=config["gold_col"],
        inital_answer_col="star_correction_initial_generation",
        correction_col=f"star_correction_{iteration}",
        reward_function=reward_function,
    )
    stats_test = collect_correction_stats(
        dataset=new_test,
        question_col=config["question_col"],
        reference_col=config["gold_col"],
        inital_answer_col="star_correction_initial_generation",
        correction_col=f"star_correction_{iteration}",
        reward_function=reward_function,
    )
    print(
        f"[INFO] Train Correction Statistics at step {iteration}:\n"
        f"[INFO]       - Correct → Incorrect: {stats_train['correct_to_incorrect']:.2f}%\n"
        f"[INFO]       - Correct → Correct: {stats_train['correct_to_correct']:.2f}%\n"
        f"[INFO]       - Incorrect → Correct: {stats_train['incorrect_to_correct']:.2f}%"
    )
    print(
        f"[INFO] Test Correction Statistics at step {iteration}:\n"
        f"[INFO]       - Correct → Incorrect: {stats_test['correct_to_incorrect']:.2f}%\n"
        f"[INFO]       - Correct → Correct: {stats_test['correct_to_correct']:.2f}%\n"
        f"[INFO]       - Incorrect → Correct: {stats_test['incorrect_to_correct']:.2f}%"
    )
    new_test.save_to_disk(f"{ft_dataset_path}_test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--generation_model_path", type=str, required=True)
    parser.add_argument("--ft_dataset_path", type=str, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--initial_generation", action="store_true", help="Set this flag for initial generation"
    )
    args = parser.parse_args()

    iteration = args.iteration

    config = load_config(args.config_path)

    # Load dataset
    dataset = datasets.load_from_disk(str(config["data_path"]))
    train_data, test_data = dataset["train"], dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], cache_dir=config["cache_dir"])
    prompt_builder = get_prompt_builder(config["task_type"])
    reward_function = RewardEvaluator(config)

    initial_generation_few_shot = load_few_shot_prompts(
        config["few_shot_dir"], f"{config['task_type']}_initial"
    )
    correction_few_shot = load_few_shot_prompts(
        config["few_shot_dir"], f"{config['task_type']}_correction"
    )

    # Prompt functions
    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config["question_col"],
        few_shot_prompts=initial_generation_few_shot,
        context_col=config["context_col"],
    )

    correction_prompt_func = partial(
        prompt_builder.build_correction_prompt,
        tokenizer=tokenizer,
        question_col=config["question_col"],
        few_shot_prompts=correction_few_shot,
        initial_answer_col="star_correction_initial_generation",
        context_col=config["context_col"],
    )

    print(f"Generating from Model {args.generation_model_path}")

    # Initialize model (M0)
    model = LLM(
        args.generation_model_path,
        download_dir=config["cache_dir"],
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        max_model_len=config["max_model_len"],
        disable_log_stats=True,  # Disables logging statistics
        seed=config["random_seed"],
        # disable_log_requests=True,  # Disables logging requests
    )
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=config["temperature"],
        top_p=config["top_p"],
        max_tokens=config["max_tokens"],
        n=1,
        seed=config["random_seed"],
    )

    if not config["initial_answer_with_new_model"]:
        if args.initial_generation:
            new_train, new_test = branch_initial_generation(
                config=config,
                model=model,
                train_data=train_data,
                test_data=test_data,
                init_prompt_func=initial_generation_prompt_func,
                sampling_params=sampling_params,
                iteration=iteration,
                ft_dataset_path=args.ft_dataset_path,
                reward_function=reward_function,
            )
            dataset = DatasetDict({"train": new_train, "test": new_test})
            dataset.save_to_disk(args.ft_dataset_path)
            print(f"[INFO] Saving initial generations to {args.ft_dataset_path}")
        else:
            branch_correction_generation(
                config,
                model,
                train_data,
                test_data,
                correction_prompt_func,
                sampling_params,
                iteration,
                args.ft_dataset_path,
                reward_function=reward_function,
                prompt_builder=prompt_builder,
            )
    else:
        new_train, new_test = branch_initial_generation(
            config=config,
            model=model,
            train_data=train_data,
            test_data=test_data,
            init_prompt_func=initial_generation_prompt_func,
            sampling_params=sampling_params,
            iteration=iteration,
            ft_dataset_path=args.ft_dataset_path,
            reward_function=reward_function,
        )
        branch_correction_generation(
            config,
            model,
            new_train,
            new_test,
            correction_prompt_func,
            sampling_params,
            iteration,
            args.ft_dataset_path,
            reward_function=reward_function,
            prompt_builder=prompt_builder,
        )


if __name__ == "__main__":
    main()
