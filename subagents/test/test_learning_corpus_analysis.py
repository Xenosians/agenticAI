import json

from pathlib import (
    Path,
)

import pytest

from learning.corpus_analysis import (
    analyze_corpus,
)

from learning.types import (
    CorrectionEvent,
    CorrectionValue,
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


def build_trajectory(
    *,
    trajectory_id: str,
    user_request: str,
    agent: str = (
        "developer-specialist"
    ),
    tool: str = (
        "workspace_git_status"
    ),
    arguments: dict | None = None,
    outcome_code: str = (
        "success"
    ),
) -> LearningTrajectory:

    if arguments is None:

        arguments = {
            "repository":
                "frontend"
        }

    return (
        LearningTrajectory(
            trajectory_id=(
                trajectory_id
            ),

            observed_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            job_id=(
                f"job-{trajectory_id}"
            ),

            attempt=1,

            user_request=(
                user_request
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "success"
            ),

            routes=[
                agent,
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        f"task-{trajectory_id}"
                    ),

                    agent=(
                        agent
                    ),

                    status=(
                        "success"
                    ),

                    outcome_code=(
                        outcome_code
                    ),

                    proposed_tool=(
                        tool
                    ),

                    proposed_arguments=(
                        arguments
                    ),
                )
            ],

            final_answer=(
                "Done."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,

                    route_count=1,

                    specialist_count=1,

                    specialist_success_count=1,

                    specialist_error_count=0,

                    approval_required_count=0,

                    tool_proposed_count=1,

                    tool_success_count=1,

                    overall_success=True,

                    had_error=False,

                    waiting_approval=False,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    components={
                        "success":
                            1.0
                    },

                    total=1.0,

                    quality_eligible=False,
                )
            ),

            quality=None,

            dataset_eligible=False,
        )
    )


def write_jsonl(
    path: Path,
    items,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for item in items:

            if hasattr(
                item,
                "model_dump"
            ):

                payload = (
                    item.model_dump(
                        mode="json",
                        by_alias=True,
                    )
                )

            else:

                payload = (
                    item
                )

            handle.write(
                json.dumps(
                    payload,
                    sort_keys=True,
                )
            )

            handle.write(
                "\n"
            )


def build_correction(
    *,
    correction_id: str,
    trajectory_id: str,
) -> CorrectionEvent:

    return (
        CorrectionEvent(
            correction_id=(
                correction_id
            ),

            observed_at=(
                "2026-09-15T00:00:01+00:00"
            ),

            trajectory_id=(
                trajectory_id
            ),

            task_id=(
                f"task-{trajectory_id}"
            ),

            correction_type=(
                "repository_scope"
            ),

            source=(
                "explicit_user"
            ),

            values=[
                CorrectionValue(
                    field=(
                        "repository"
                    ),

                    rejected_value=(
                        "ai"
                    ),

                    chosen_value=(
                        "frontend"
                    ),
                )
            ],

            dataset_eligible=False,
        )
    )


def test_corpus_analysis_counts_runtime_evidence(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id=(
                    "t1"
                ),

                user_request=(
                    "Show frontend status."
                ),
            ),

            build_trajectory(
                trajectory_id=(
                    "t2"
                ),

                user_request=(
                    "Show backend status."
                ),

                arguments={
                    "repository":
                        "backend"
                },
            ),
        ],
    )

    write_jsonl(
        correction_path,
        [
            build_correction(
                correction_id=(
                    "c1"
                ),

                trajectory_id=(
                    "t1"
                ),
            )
        ],
    )

    report = (
        analyze_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),
        )
    )

    assert (
        report.trajectory_count
        == 2
    )

    assert (
        report.correction_count
        == 1
    )

    assert (
        report.corrected_trajectory_count
        == 1
    )

    assert (
        report.tool_counts[
            "workspace_git_status"
        ]
        == 2
    )

    assert (
        report.domain_counts[
            "developer"
        ]
        == 2
    )


def test_duplicate_requests_are_normalized(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id=(
                    "t1"
                ),

                user_request=(
                    "Show frontend status"
                ),
            ),

            build_trajectory(
                trajectory_id=(
                    "t2"
                ),

                user_request=(
                    "  SHOW   FRONTEND STATUS  "
                ),
            ),
        ],
    )

    report = (
        analyze_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),
        )
    )

    assert (
        report.unique_request_count
        == 1
    )

    assert (
        report.duplicate_request_count
        == 1
    )

    assert (
        report.duplicate_request_rate
        == 0.5
    )


def test_held_out_eval_contamination_is_detected(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    eval_path = (
        tmp_path
        / "core.v2.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id=(
                    "t1"
                ),

                user_request=(
                    "Show me ticket KAN-1."
                ),

                agent=(
                    "ticket-specialist"
                ),

                tool=(
                    "ticket_get"
                ),

                arguments={
                    "ticket_key":
                        "KAN-1"
                },
            )
        ],
    )

    write_jsonl(
        eval_path,
        [
            {
                "schema":
                    "evaluation-case.v1",

                "case_id":
                    "ticket.kan1.get",

                "suite":
                    "core.v2",

                "user_request":
                    " show me ticket KAN-1. ",
            }
        ],
    )

    report = (
        analyze_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),

            eval_paths=[
                eval_path,
            ],
        )
    )

    assert (
        report
        .held_out_contamination_count
        == 1
    )

    assert (
        report
        .contamination_matches[
            0
        ]
        .eval_cases
        == [
            "core.v2:ticket.kan1.get"
        ]
    )

    assert any(
        warning.code
        == "held_out_contamination"

        for warning
        in report.warnings
    )


def test_orphan_correction_is_reported(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id=(
                    "t1"
                ),

                user_request=(
                    "Show frontend status."
                ),
            )
        ],
    )

    write_jsonl(
        correction_path,
        [
            build_correction(
                correction_id=(
                    "c1"
                ),

                trajectory_id=(
                    "missing"
                ),
            )
        ],
    )

    report = (
        analyze_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),
        )
    )

    assert (
        report.orphan_correction_count
        == 1
    )

    assert any(
        warning.code
        == "orphan_corrections"

        for warning
        in report.warnings
    )


def test_duplicate_trajectory_ids_are_rejected(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    correction_path = (
        tmp_path
        / "corrections.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id=(
                    "duplicate"
                ),

                user_request=(
                    "First request."
                ),
            ),

            build_trajectory(
                trajectory_id=(
                    "duplicate"
                ),

                user_request=(
                    "Second request."
                ),
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate trajectory_id"
        ),
    ):

        analyze_corpus(
            trajectory_path=(
                trajectory_path
            ),

            correction_path=(
                correction_path
            ),
        )