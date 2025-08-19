from datasets import Dataset


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
    strict_improvement=True,
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
        # for init_answer in row[inital_answer_col]:
        #     for correction in flatten_predictions(row[correction_col]):
        # 3) Check if there is an improvement
        init_is_correct = reward_function(
            ground_truth=reference, model_answer=row[inital_answer_col]
        )
        correction_is_correct = reward_function(
            ground_truth=reference, model_answer=row[correction_col]
        )

        if strict_improvement:
            use_sample = correction_is_correct > init_is_correct
        else:
            use_sample = init_is_correct and correction_is_correct

        if use_sample:
            new_ids.append(f"{row_id}_gen")
            new_questions.append(question)
            new_refs.append(reference)
            new_answers.append(row[inital_answer_col])
            new_corrections.append(row[correction_col])

            messages = prompt_builder.build_correction_messages_with_final_answer(
                question, row[inital_answer_col], row[correction_col], all_context
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

    return Dataset.from_dict(flattened_data)
