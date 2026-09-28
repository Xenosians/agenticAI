from learning.curriculum.catalog import (
    get_curriculum,
)

from learning.curriculum.mastery import (
    CurriculumState,
    initial_curriculum_state,
)

from learning.curriculum.planner import (
    build_curriculum_training_plan,
)


def _state_at_grounded_binding():
    curriculum = get_curriculum("hub")
    initial = initial_curriculum_state(
        curriculum
    )

    return CurriculumState(
        curriculum_id=initial.curriculum_id,
        curriculum_version=initial.curriculum_version,
        target_component=initial.target_component,
        updated_at=initial.updated_at,
        current_chapter_id="hub-05-grounded-bindings",
        mastered_chapter_ids=[
            "hub-01-structured-instructions",
            "hub-02-resource-operation",
            "hub-03-scope-context",
            "hub-04-delegation",
        ],
        chapter_attempt_counts={},
        best_mastery_scores={},
        last_evaluation_id=None,
        status="learning",
    )


def test_real_jira_failure_maps_to_grounded_binding_chapter():
    curriculum = get_curriculum("hub")
    state = _state_at_grounded_binding()

    plan = build_curriculum_training_plan(
        curriculum=curriculum,
        state=state,
        failure_candidates=[
            {
                "candidate_id": "candidate-jira-task",
                "failure_code": "semantic_argument_unbound",
                "missing_bindings": {
                    "ticket_type": [
                        "Task"
                    ]
                },
            }
        ],
    )

    assert plan.current_chapter_id == "hub-05-grounded-bindings"
    assert plan.matching_failure_count == 1
    assert plan.matching_candidate_ids == [
        "candidate-jira-task"
    ]


def test_planner_keeps_mastered_chapters_as_replay():
    curriculum = get_curriculum("hub")
    state = _state_at_grounded_binding()

    plan = build_curriculum_training_plan(
        curriculum=curriculum,
        state=state,
        failure_candidates=[],
    )

    assert (
        plan.mastered_replay_chapter_ids
        == state.mastered_chapter_ids
    )

    assert (
        plan.mixture.mastered_replay_fraction
        == 0.30
    )


def test_hard_case_sampling_does_not_replace_replay():
    curriculum = get_curriculum("hub")
    state = _state_at_grounded_binding()

    plan = build_curriculum_training_plan(
        curriculum=curriculum,
        state=state,
        failure_candidates=[
            {
                "candidate_id": "candidate-1",
                "failure_code": "semantic_argument_unbound",
            },
        ],
    )

    assert plan.mixture.current_chapter_fraction == 0.60
    assert plan.mixture.mastered_replay_fraction == 0.30
    assert plan.mixture.hard_case_fraction == 0.10

    assert plan.training_authorized is False
    assert plan.promotion_authorized is False


def test_unrelated_failures_do_not_jump_curriculum():
    curriculum = get_curriculum("hub")
    state = _state_at_grounded_binding()

    plan = build_curriculum_training_plan(
        curriculum=curriculum,
        state=state,
        failure_candidates=[
            {
                "candidate_id": "candidate-other",
                "failure_code": "tool_execution_error",
            },
        ],
    )

    assert plan.current_chapter_id == "hub-05-grounded-bindings"
    assert plan.matching_failure_count == 0


def test_developer_curriculum_is_code_progressive():
    curriculum = get_curriculum("developer")

    assert [
        item.chapter_id
        for item in curriculum.chapters
    ] == [
        "dev-01-workspace-reads",
        "dev-02-git-observation",
        "dev-03-build-test",
        "dev-04-code-symbols",
        "dev-05-single-file-patches",
        "dev-06-multi-file-patches",
        "dev-07-cross-repository",
        "dev-08-governed-mutations",
    ]
