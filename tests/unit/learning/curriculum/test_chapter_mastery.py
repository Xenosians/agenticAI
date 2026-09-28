from learning.curriculum.catalog import (
    get_curriculum,
)

from learning.curriculum.mastery import (
    apply_mastery_evaluation,
    evaluate_current_chapter,
    initial_curriculum_state,
)


def test_hub_curriculum_is_ordered_like_chapters():
    curriculum = get_curriculum("hub")

    assert [
        item.ordinal
        for item in curriculum.chapters
    ] == list(
        range(
            1,
            len(curriculum.chapters) + 1,
        )
    )


def test_hub_curriculum_contains_grounded_binding_chapter():
    curriculum = get_curriculum("hub")

    chapter = curriculum.chapter(
        "hub-05-grounded-bindings"
    )

    assert (
        "semantic_argument_unbound"
        in chapter.failure_codes
    )

    assert (
        "grounded_argument_recall"
        in chapter.mastery_thresholds
    )


def test_mastery_gate_refuses_missing_metrics():
    curriculum = get_curriculum("hub")
    state = initial_curriculum_state(
        curriculum
    )

    evaluation = evaluate_current_chapter(
        curriculum=curriculum,
        state=state,
        metrics={
            "instruction_following": 0.99,
        },
        reviewed_example_count=100,
        safety_pass_rate=1.0,
    )

    assert evaluation.mastered is False
    assert (
        "metric-below-threshold:json_contract_validity"
        in evaluation.block_reasons
    )


def test_mastery_gate_refuses_safety_regression():
    curriculum = get_curriculum("hub")
    state = initial_curriculum_state(
        curriculum
    )

    evaluation = evaluate_current_chapter(
        curriculum=curriculum,
        state=state,
        metrics={
            "instruction_following": 0.99,
            "json_contract_validity": 1.0,
        },
        reviewed_example_count=100,
        safety_pass_rate=0.99,
    )

    assert evaluation.mastered is False
    assert "safety-regression" in evaluation.block_reasons


def test_mastery_gate_advances_exactly_one_chapter():
    curriculum = get_curriculum("hub")
    state = initial_curriculum_state(
        curriculum
    )

    evaluation = evaluate_current_chapter(
        curriculum=curriculum,
        state=state,
        metrics={
            "instruction_following": 0.95,
            "json_contract_validity": 1.0,
        },
        reviewed_example_count=8,
        safety_pass_rate=1.0,
    )

    assert evaluation.mastered is True

    next_state = apply_mastery_evaluation(
        curriculum=curriculum,
        state=state,
        evaluation=evaluation,
    )

    assert next_state.mastered_chapter_ids == [
        "hub-01-structured-instructions"
    ]

    assert (
        next_state.current_chapter_id
        == "hub-02-resource-operation"
    )
