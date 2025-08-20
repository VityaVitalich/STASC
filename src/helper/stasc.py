from datasets import Dataset

from prompts.stasc_builder import StascQABuilder


def filter_corrections(
    dataset,
    reward_function,
    prompt_builder: StascQABuilder,
    question_col="question",
    reference_col="reference",
    context_col="",
    init_answer_col="initial_answer",
    corr_answer_col="correction_answer",
    id_col="id",
    mode="improving",  # "improving" = I, "non_decreasing" = N
    threshold=0,  # for N-mode: treat answers as "correct" if reward ≥ threshold
):
    """Filter corrections based on reward function.
    - mode="improving": keep only strictly better corrections (Improving Filter, I).
    - mode="non_decreasing": keep corrections if they improve OR if initial is already correct (N).
    """
    new_ids, new_questions, new_refs = [], [], []
    new_inits, new_corrections, new_messages = [], [], []

    for row in dataset:
        row_id = row[id_col]
        question = row[question_col]
        reference = row[reference_col]
        all_context = row.get(context_col, [""])

        # Evaluate rewards
        init_reward = reward_function(ground_truth=reference, model_answer=row[init_answer_col])
        corr_reward = reward_function(ground_truth=reference, model_answer=row[corr_answer_col])

        # --- Filtering logic ---
        if mode == "improving":
            # Keep only strictly better corrections
            use_sample = corr_reward > init_reward
        elif mode == "non_decreasing":
            # Keep if correction improves OR init is already correct enough
            use_sample = (corr_reward > init_reward) or (init_reward >= threshold)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if use_sample:
            new_ids.append(f"{row_id}_gen")
            new_questions.append(question)
            new_refs.append(reference)
            new_inits.append(row[init_answer_col])
            new_corrections.append(row[corr_answer_col])

            # Messages for training
            messages = prompt_builder.build_correction_messages_with_final_answer(
                question, row[init_answer_col], row[corr_answer_col], all_context
            )
            new_messages.append(messages)

    # Build new dataset
    filtered_data = {
        id_col: new_ids,
        question_col: new_questions,
        reference_col: new_refs,
        init_answer_col: new_inits,
        corr_answer_col: new_corrections,
        "messages": new_messages,
    }

    print(f"[INFO] Filtered {len(new_ids)} corrections in mode={mode}")
    return Dataset.from_dict(filtered_data)
