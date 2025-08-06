import argparse
import logging
import os
from functools import partial

import datasets
import torch
from transformers import AutoTokenizer  # pyright: ignore[reportPrivateImportUsage]
from vllm import LLM, SamplingParams

from prompts.prompt_schemas import load_few_shot_prompts
from prompts.self_refine_builder import SelfRefinePromptBuilder
from utils.eval_utils import RewardEvaluator
from utils.generation_utils import generate_for_dataset, load_config, store_generation_results
from utils.utils import KM, construct_run_name


def get_informative_candidate(
    sample,
    current_iteration,
    initial_gen_col="self_refine_initial_generation",
    feedback_prefix="self_refine_feedback_",
    refinement_prefix="self_refine_refinement_",
    trigger_phrase="!The Answer is correct!",
):
    """For a given sample and current iteration index (where the feedback contains the trigger),
    roll back along the candidate chain until we find the most recent informative candidate.

    Informative means that the candidate was obtained from an update (i.e. its iteration's feedback
    did NOT contain the trigger phrase). For iteration 0 or if no informative update is found,
    return the initial generation candidate.

    Returns the candidate answer string.
    """
    # For iteration 0, there is no roll-back available.
    if current_iteration == 0:
        init_ans = sample.get(initial_gen_col, [""])
        return init_ans[0] if isinstance(init_ans, list) and init_ans else init_ans

    # Iterate backward from current_iteration-1 down to 0.
    for j in range(current_iteration - 1, -1, -1):
        fb_key = f"{feedback_prefix}{j}"
        # If a feedback exists for iteration j, check if it does NOT contain the trigger.
        if fb_key in sample:
            feedback_j = sample.get(fb_key)
            if isinstance(feedback_j, list):
                feedback_j = feedback_j[0]
            # If this iteration j was not a kept‐same event, then candidate for iteration j is informative.
            if trigger_phrase not in feedback_j:
                # Get the candidate produced in iteration j.
                if j == 0:
                    cand = sample.get(initial_gen_col, [""])
                    return cand[0] if isinstance(cand, list) and cand else cand
                else:
                    ref_key = f"{refinement_prefix}{j}"
                    if ref_key in sample:
                        cand = sample.get(ref_key)
                        if isinstance(cand, list):
                            cand = cand[0]
                        return cand
    # Fallback: if no informative update is found, return the initial generation.
    init_ans = sample.get(initial_gen_col, [""])
    return init_ans[0] if isinstance(init_ans, list) and init_ans else init_ans


def compute_current_iteration_stats(
    ds,
    iteration,
    gt_col,
    evaluator,
    initial_gen_col="self_refine_initial_generation",
    feedback_prefix="self_refine_feedback_",
    refinement_prefix="self_refine_refinement_",
    trigger_phrase="is correct",
    threshold=0.5,
):
    """For a given iteration index (i), compute the following for samples that have that iteration:

    A) If the feedback for iteration i contains the trigger_phrase (i.e. kept_same event):
         - Instead of using the immediate previous candidate (which may also have been kept), roll back
           through earlier iterations (via get_informative_candidate) to get an informative candidate.
         - Evaluate whether that rolled-back candidate is correct or incorrect.

    B) If the feedback does NOT contain the trigger_phrase and a refinement exists, then treat this as an update:
         - Evaluate the transition from the previous candidate (from iteration i-1 or initial generation)
           to the new candidate (from refinement i).
         - Classify the transition as:
               - incorrect_to_correct (prev incorrect, new correct),
               - correct_to_incorrect (prev correct, new incorrect),
               - correct_to_correct (prev and new both correct).
         - (Transitions from incorrect to incorrect are not tracked.)

    Returns a dictionary with the following keys:
       - total_samples: # of samples with current iteration feedback.
       - kept_same_percentage: % of samples that used kept_same.
       - kept_same_correct_percentage: Among kept_same events, % whose informative candidate is correct.
       - kept_same_incorrect_percentage: Among kept_same events, % whose informative candidate is incorrect.
       - incorrect_to_correct_percentage: Among updates, % transitioning from incorrect to correct.
       - correct_to_incorrect_percentage: Among updates, % transitioning from correct to incorrect.
       - correct_to_correct_percentage: Among updates, % transitioning from correct to correct.
       - update_percentage: % of samples that underwent an update (refinement applied) at iteration i.
    """
    total_samples = 0

    # Counters for kept_same events.
    kept_same_count = 0
    kept_same_correct = 0
    kept_same_incorrect = 0

    # Counters for update events (where a refinement was applied).
    update_count = 0
    incorrect_to_correct = 0
    correct_to_incorrect = 0
    correct_to_correct = 0

    for sample in ds:
        # Determine previous candidate answer.
        total_samples += 1
        if iteration == 0:
            prev_ans_raw = sample.get(initial_gen_col, [""])
            prev_ans = (
                prev_ans_raw[0] if isinstance(prev_ans_raw, list) and prev_ans_raw else prev_ans_raw
            )
        else:
            prev_key = f"{refinement_prefix}{iteration - 1}"
            prev_raw = sample.get(prev_key)
            prev_ans = prev_raw[0] if isinstance(prev_raw, list) else prev_raw

        fb_key = f"{feedback_prefix}{iteration}"
        feedback_raw = sample.get(fb_key)
        feedback = feedback_raw[0] if isinstance(feedback_raw, list) else feedback_raw

        # Case A: Kept_same event if the feedback contains trigger_phrase.
        if trigger_phrase in feedback.lower():
            kept_same_count += 1
            # Instead of using prev_ans directly, roll back to get the informative candidate.
            baseline_ans = get_informative_candidate(
                sample,
                current_iteration=iteration,
                initial_gen_col=initial_gen_col,
                feedback_prefix=feedback_prefix,
                refinement_prefix=refinement_prefix,
                trigger_phrase=trigger_phrase,
            )
            ground_truth = sample[gt_col]
            score = evaluator(ground_truth=ground_truth, model_answer=baseline_ans)
            if score >= threshold:
                kept_same_correct += 1
            else:
                kept_same_incorrect += 1
        else:
            # Case B: Update event—check for corresponding refinement.
            ref_key = f"{refinement_prefix}{iteration}"
            if ref_key not in sample:
                continue  # Cannot process update if refinement is missing.
            update_count += 1
            new_raw = sample.get(ref_key)
            new_ans = new_raw[0] if isinstance(new_raw, list) else new_raw
            ground_truth = sample[gt_col]
            # Use immediate previous candidate (since update is occurring).
            prev_score = evaluator(ground_truth=ground_truth, model_answer=prev_ans)
            new_score = evaluator(ground_truth=ground_truth, model_answer=new_ans)
            prev_correct = prev_score >= threshold
            new_correct = new_score >= threshold
            if (not prev_correct) and new_correct:
                incorrect_to_correct += 1
            elif prev_correct and (not new_correct):
                correct_to_incorrect += 1
            elif prev_correct and new_correct:
                correct_to_correct += 1
            # Transitions from incorrect to incorrect are not recorded.

    # If no sample processed for current iteration, return zeros.
    if total_samples == 0:
        return {
            "total_samples": 0,
            "kept_same_percentage": 0,
            "kept_same_correct_percentage": 0,
            "kept_same_incorrect_percentage": 0,
            "incorrect_to_correct_percentage": 0,
            "correct_to_incorrect_percentage": 0,
            "correct_to_correct_percentage": 0,
            "update_percentage": 0,
        }

    kept_same_percentage = (kept_same_count / total_samples) * 100
    update_percentage = (update_count / total_samples) * 100

    if kept_same_count > 0:
        kept_same_correct_percentage = (kept_same_correct / kept_same_count) * 100
        kept_same_incorrect_percentage = (kept_same_incorrect / kept_same_count) * 100
    else:
        kept_same_correct_percentage = 0
        kept_same_incorrect_percentage = 0

    if update_count > 0:
        incorrect_to_correct_percentage = (incorrect_to_correct / total_samples) * 100
        correct_to_incorrect_percentage = (correct_to_incorrect / total_samples) * 100
        correct_to_correct_percentage = (correct_to_correct / total_samples) * 100
    else:
        incorrect_to_correct_percentage = 0
        correct_to_incorrect_percentage = 0
        correct_to_correct_percentage = 0

    return {
        "total_samples": total_samples,
        "kept_same_percentage": kept_same_percentage,
        "kept_same_correct_percentage": kept_same_correct_percentage,
        "kept_same_incorrect_percentage": kept_same_incorrect_percentage,
        "incorrect_to_correct_percentage": incorrect_to_correct_percentage,
        "correct_to_incorrect_percentage": correct_to_incorrect_percentage,
        "correct_to_correct_percentage": correct_to_correct_percentage,
        "update_percentage": update_percentage,
    }


def get_final_refined_answer(
    sample,
    initial_gen_col="self_refine_initial_generation",
    feedback_prefix="self_refine_feedback_",
    refinement_prefix="self_refine_refinement_",
    trigger_phrase="!The Answer is correct!",
):
    """Given a sample (a dictionary with keys for initial generation, feedback, and refinements),
    select the final answer according to the following rules:
      - Start with the initial generation.
      - For each iteration i:
          If the feedback message at iteration i contains the trigger_phrase,
          return the answer from the previous step (i-1 or the initial answer if i is 0).
          Else, if a refinement exists at iteration i, update the candidate answer and continue.
      - If no trigger phrase is found in any feedback, return the last available refinement (or the initial answer
        if no refinement exists).
    """
    # Extract initial answer (assumed stored as list with one element)
    try:
        candidate = sample.get(initial_gen_col, [""])[0]
    except Exception:
        candidate = sample.get(initial_gen_col, "")

    i = 0
    while True:
        fb_key = f"{feedback_prefix}{i}"
        ref_key = f"{refinement_prefix}{i}"

        # Retrieve feedback for iteration i
        feedback = sample.get(fb_key)
        if feedback is None:
            # No more feedback, so break the loop.
            break

        # If feedback is stored as a list, use its first element.
        feedback_text = feedback[0] if isinstance(feedback, list) else feedback

        # Check for the trigger phrase in the feedback. If found,
        # return the candidate answer from the previous iteration.
        if trigger_phrase in feedback_text.lower():
            return candidate

        # Otherwise, check if a corresponding refinement exists.
        refinement = sample.get(ref_key)
        if refinement is None:
            # No refinement to update candidate, so stop.
            break

        # If refinement is a list, use its first element.
        candidate = refinement[0] if isinstance(refinement, list) else refinement

        # Continue to next iteration.
        i += 1

    # After processing all iterations, return the candidate answer.
    return candidate


def KM_self_refine(
    ds,
    gt_col,
    evaluator,
    initial_gen_col="self_refine_initial_generation",
    feedback_prefix="self_refine_feedback_",
    refinement_prefix="self_refine_refinement_",
    trigger_phrase="is correct",
):
    """Compute an average evaluation score over a dataset of samples, where the final answer is selected
    from a series of iterations stored in separate columns.

    For each sample:
      - The ground truth answer is at key gt_col.
      - The answer chain is stored as:
          initial answer at key initial_gen_col (stored as a list containing a single string)
          feedbacks at keys f"{feedback_prefix}{i}" (stored as list of one string) for each iteration i
          refinements at keys f"{refinement_prefix}{i}" (stored as list of one string) for each iteration i
      - The final answer is determined as follows:
          * If any feedback message contains the trigger_phrase, the candidate answer from the previous iteration
            is taken as final.
          * Otherwise, the final answer is the last refinement, or if no refinement exists, the initial answer.
    The evaluator is applied on the ground truth versus the final answer for each sample.
    """
    total_score = 0.0
    total_count = 0

    for sample in ds:
        ground_truth = sample[gt_col]
        # Select final answer by iterating through the stored feedback and refinements.
        final_answer = get_final_refined_answer(
            sample,
            initial_gen_col=initial_gen_col,
            feedback_prefix=feedback_prefix,
            refinement_prefix=refinement_prefix,
            trigger_phrase=trigger_phrase,
        )
        total_score += evaluator(ground_truth=ground_truth, model_answer=final_answer)
        total_count += 1

    return total_score / total_count if total_count > 0 else 0.0


def setup_logger(run_name: str, log_file="star.log"):
    """Sets up a logger named "star_logger_{run_name}" that writes both
    to the console and to `log_file`.
    """
    logger_name = f"self_refine_logger_{run_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers if already set
    if not logger.handlers:
        # 1) Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)

        # 2) File handler
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


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
    parser = argparse.ArgumentParser(description="Run the Self-Refine Algorithm")
    parser.add_argument(
        "--model_config", type=str, required=True, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--data_config", type=str, required=True, help="Path to the YAML config file."
    )
    parser.add_argument(
        "--algo_config", type=str, required=True, help="Path to the YAML config file."
    )
    args = parser.parse_args()

    # Load configuration
    algo_config = load_config(args.algo_config)
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)

    run_name = construct_run_name(
        args.model_config, args.data_config, args.algo_config, algo_config["run_name_specification"]
    )
    algo_config["run_name"] = run_name

    config = {**algo_config, **model_config, **data_config}

    logger = setup_logger(config["run_name"], log_file=f"logs/general/{config['run_name']}.log")

    # Load dataset
    dataset = datasets.load_from_disk(str(config["data_path"]))
    _, test_data = dataset["train"], dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], cache_dir=config["cache_dir"])

    # few shots
    generation_few_shot_prompts = load_few_shot_prompts(
        config["few_shot_dir"], "self_refine_generation"
    )
    feedback_few_shot_prompts = load_few_shot_prompts(
        config["few_shot_dir"], "self_refine_feedback"
    )
    refine_few_shot_prompts = load_few_shot_prompts(
        config["few_shot_dir"], "self_refine_refinement"
    )

    prompt_builder = SelfRefinePromptBuilder(config)
    reward_function = RewardEvaluator(config)

    # Prompt functions
    # conversation_gather_func = partial(
    #     gather_full_conversation,
    #     question_col=config['question_col'],
    #     generation_col='self_refine_initial_generation',
    #     feedback_prefix='self_refine_feedback_',
    #     refinement_prefix='self_refine_refinement_')

    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=generation_few_shot_prompts,
    )

    feedback_prompt_func = partial(
        prompt_builder.build_feedback_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=feedback_few_shot_prompts,
    )
    refine_prompt_func = partial(
        prompt_builder.build_correction_prompt,
        tokenizer=tokenizer,
        few_shot_prompts=refine_few_shot_prompts,
    )

    save_dir = os.path.join(config["cache_dir"], "self_refine")
    os.makedirs(save_dir, exist_ok=True)
    run_dir = os.path.join(save_dir, config["run_name"])

    # Initialize model (M0)
    model = LLM(
        config["model_path"],
        download_dir=config["cache_dir"],
        dtype="bfloat16",
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=config["gpu_memory_utilization"],
        enforce_eager=config["enforce_eager"],
        max_model_len=config["max_model_len"],
        seed=config["random_seed"],
        # disable_log_stats=True,  # Disables logging statistics
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

    test_data = perform_generation(
        data=test_data,
        model=model,
        prompt_func=initial_generation_prompt_func,
        sampling_params=sampling_params,
        id_key=config["id_col"],
        output_col="self_refine_initial_generation",
    )
    acc = KM(
        test_data,
        target_col="self_refine_initial_generation",
        gt_col=config["gold_col"],
        evaluator=reward_function,
    )
    logger.info(f"[INFO] Initial Accuracy {acc}")

    # Outer loop: for n in 1...N
    for iteration in range(config["num_refine_iterations"]):
        logger.info(f"Starting feedback {iteration + 1}/{config['num_refine_iterations']}")

        test_data = perform_generation(
            data=test_data,
            model=model,
            prompt_func=feedback_prompt_func,
            sampling_params=sampling_params,
            id_key=config["id_col"],
            output_col=f"self_refine_feedback_{iteration}",  # store model's answer after rationale generation
        )

        logger.info(f"Starting refinement {iteration + 1}/{config['num_refine_iterations']}")

        test_data = perform_generation(
            data=test_data,
            model=model,
            prompt_func=refine_prompt_func,
            sampling_params=sampling_params,
            id_key=config["id_col"],
            output_col=f"self_refine_refinement_{iteration}",
        )

        acc = KM_self_refine(test_data, gt_col=config["gold_col"], evaluator=reward_function)
        logger.info(
            f"[INFO] Refinement Accuracy {acc} at iteration {iteration + 1}/{config['num_refine_iterations']}"
        )

        stats = compute_current_iteration_stats(
            test_data,
            iteration,
            gt_col=config["gold_col"],
            evaluator=reward_function,
        )
        logger.info(
            f"[INFO] Test Correction Statistics at step {iteration}:\n"
            f"[INFO]       - Correct → Incorrect: {stats['correct_to_incorrect_percentage']:.2f}%\n"
            f"[INFO]       - Correct → Correct: {stats['correct_to_correct_percentage']:.2f}%\n"
            f"[INFO]       - Incorrect → Correct: {stats['incorrect_to_correct_percentage']:.2f}%\n"
            f"[INFO]       - Kept Same: {stats['kept_same_percentage']:.2f}% "
            f"[INFO] (Correct: {stats['kept_same_correct_percentage']:.2f}%, "
            f"[INFO] Incorrect: {stats['kept_same_incorrect_percentage']:.2f}%)"
        )

    test_data.save_to_disk(run_dir)
    logger.info("Self-Refine algorithm completed.")


if __name__ == "__main__":
    main()
