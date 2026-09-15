import json

from pathlib import (
    Path,
)

import pytest

from learning.curation import (
    CuratedTrajectoryReference,
    CurationReport,
)

from learning.diversity_gate import (
    DiversityGatePolicy,
    evaluate_diversity_gate,
)

from learning.types import (
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
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

            payload = (
                item.model_dump(
                    mode="json",
                    by_alias=True,
                )

                if hasattr(
                    item,
                    "model_dump"
                )

                else item
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


def build_trajectory(
    *,
    trajectory_id: str,
    user_request: str,
    agent: str,
    tool: str,
    arguments: dict | None = None,
) -> LearningTrajectory:

    if arguments is None:

        arguments = {}

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
                        "success"
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

            dataset_eligible=True,
        )
    )


def build_curation_report(
    eligible_ids: list[
        str
    ],
) -> CurationReport:

    return (
        CurationReport(
            curation_id=(
                "curation-test"
            ),

            generated_at=(
                "2026-09-15T00:00:00+00:00"
            ),

            candidate_trajectory_count=(
                len(
                    eligible_ids
                )
            ),

            eligible_trajectory_count=(
                len(
                    eligible_ids
                )
            ),

            excluded_trajectory_count=0,

            correction_count=0,

            usable_correction_count=0,

            orphan_correction_count=0,

            deduplicated_trajectory_count=0,

            held_out_contamination_count=0,

            exclusion_reason_counts={},

            eligible=[
                CuratedTrajectoryReference(
                    trajectory_id=(
                        trajectory_id
                    ),

                    evidence_fingerprint=(
                        f"fingerprint-{trajectory_id}"
                    ),

                    usable_correction_ids=[],
                )

                for trajectory_id
                in eligible_ids
            ],

            excluded=[],

            orphan_correction_ids=[],

            contamination_matches=[],
        )
    )


def relaxed_policy(
) -> DiversityGatePolicy:

    return (
        DiversityGatePolicy(
            min_eligible_trajectories=4,
            min_unique_requests=4,
            min_unique_behavior_patterns=4,
            min_unique_domains=2,
            min_unique_capabilities=2,
            max_duplicate_request_rate=0.25,
            max_dominant_domain_share=0.75,
            max_dominant_capability_share=0.75,
        )
    )


def balanced_trajectories(
) -> list[
    LearningTrajectory
]:

    return [
        build_trajectory(
            trajectory_id="t1",
            user_request=(
                "Check frontend status."
            ),
            agent=(
                "developer-specialist"
            ),
            tool=(
                "workspace_git_status"
            ),
            arguments={
                "repository":
                    "frontend"
            },
        ),

        build_trajectory(
            trajectory_id="t2",
            user_request=(
                "List backend branches."
            ),
            agent=(
                "developer-specialist"
            ),
            tool=(
                "workspace_git_branches"
            ),
            arguments={
                "repository":
                    "backend"
            },
        ),

        build_trajectory(
            trajectory_id="t3",
            user_request=(
                "Get current state for ticket DEV-10."
            ),
            agent=(
                "ticket-specialist"
            ),
            tool=(
                "ticket_get"
            ),
            arguments={
                "ticket_key":
                    "DEV-10"
            },
        ),

        build_trajectory(
            trajectory_id="t4",
            user_request=(
                "Read comments for ticket DEV-11."
            ),
            agent=(
                "ticket-specialist"
            ),
            tool=(
                "ticket_comments"
            ),
            arguments={
                "ticket_key":
                    "DEV-11",
                "limit":
                    5,
            },
        ),
    ]


def test_empty_curated_corpus_fails_gate(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [],
    )

    report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    []
                )
            ),
        )
    )

    assert (
        report.promotion_eligible
        is False
    )

    assert (
        report
        .metrics
        .eligible_trajectory_count
        == 0
    )

    assert (
        "minimum_eligible_trajectories"
        in report.failed_checks
    )

    assert (
        "minimum_unique_requests"
        in report.failed_checks
    )


def test_balanced_curated_corpus_passes(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    trajectories = (
        balanced_trajectories()
    )

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "t2",
                        "t3",
                        "t4",
                    ]
                )
            ),

            policy=(
                relaxed_policy()
            ),
        )
    )

    assert (
        report.promotion_eligible
        is True
    )

    assert (
        report.failed_checks
        == []
    )

    assert (
        report
        .metrics
        .unique_domain_count
        == 2
    )

    assert (
        report
        .metrics
        .unique_capability_count
        == 4
    )


def test_duplicate_request_rate_can_fail_gate(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    trajectories = (
        balanced_trajectories()
    )

    trajectories[
        1
    ].user_request = (
        "  CHECK   FRONTEND STATUS.  "
    )

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    policy = (
        relaxed_policy()
    )

    policy.min_unique_requests = 3

    report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "t2",
                        "t3",
                        "t4",
                    ]
                )
            ),

            policy=(
                policy
            ),
        )
    )

    assert (
        report
        .metrics
        .duplicate_request_count
        == 1
    )

    assert (
        report
        .metrics
        .duplicate_request_rate
        == 0.25
    )

    stricter_policy = (
        policy.model_copy(
            update={
                "max_duplicate_request_rate":
                    0.20,
            }
        )
    )

    stricter_report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "t2",
                        "t3",
                        "t4",
                    ]
                )
            ),

            policy=(
                stricter_policy
            ),
        )
    )

    assert (
        stricter_report.promotion_eligible
        is False
    )

    assert (
        "maximum_duplicate_request_rate"
        in stricter_report.failed_checks
    )


def test_dominant_domain_share_can_fail_gate(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    trajectories = [
        build_trajectory(
            trajectory_id="t1",
            user_request="Request one.",
            agent="developer-specialist",
            tool="workspace_git_status",
        ),

        build_trajectory(
            trajectory_id="t2",
            user_request="Request two.",
            agent="developer-specialist",
            tool="workspace_git_branches",
        ),

        build_trajectory(
            trajectory_id="t3",
            user_request="Request three.",
            agent="developer-specialist",
            tool="workspace_git_log",
        ),

        build_trajectory(
            trajectory_id="t4",
            user_request="Request four.",
            agent="ticket-specialist",
            tool="ticket_get",
        ),
    ]

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    policy = (
        relaxed_policy()
        .model_copy(
            update={
                "max_dominant_domain_share":
                    0.70,
            }
        )
    )

    report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "t2",
                        "t3",
                        "t4",
                    ]
                )
            ),

            policy=(
                policy
            ),
        )
    )

    assert (
        report
        .metrics
        .dominant_domain
        == "developer"
    )

    assert (
        report
        .metrics
        .dominant_domain_share
        == 0.75
    )

    assert (
        "maximum_dominant_domain_share"
        in report.failed_checks
    )


def test_dominant_capability_share_can_fail_gate(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    trajectories = [
        build_trajectory(
            trajectory_id="t1",
            user_request="Request one.",
            agent="developer-specialist",
            tool="workspace_git_status",
        ),

        build_trajectory(
            trajectory_id="t2",
            user_request="Request two.",
            agent="developer-specialist",
            tool="workspace_git_status",
        ),

        build_trajectory(
            trajectory_id="t3",
            user_request="Request three.",
            agent="ticket-specialist",
            tool="workspace_git_status",
        ),

        build_trajectory(
            trajectory_id="t4",
            user_request="Request four.",
            agent="ticket-specialist",
            tool="ticket_get",
        ),
    ]

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    policy = (
        relaxed_policy()
        .model_copy(
            update={
                "max_dominant_capability_share":
                    0.70,
            }
        )
    )

    report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "t2",
                        "t3",
                        "t4",
                    ]
                )
            ),

            policy=(
                policy
            ),
        )
    )

    assert (
        report
        .metrics
        .dominant_capability
        == "workspace_git_status"
    )

    assert (
        report
        .metrics
        .dominant_capability_share
        == 0.75
    )

    assert (
        "maximum_dominant_capability_share"
        in report.failed_checks
    )


def test_excluded_trajectory_does_not_affect_metrics(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    trajectories = (
        balanced_trajectories()
    )

    trajectories.append(
        build_trajectory(
            trajectory_id="excluded",
            user_request=(
                "Duplicate excluded request."
            ),
            agent=(
                "developer-specialist"
            ),
            tool=(
                "workspace_git_status"
            ),
        )
    )

    write_jsonl(
        trajectory_path,
        trajectories,
    )

    report = (
        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "t2",
                        "t3",
                        "t4",
                    ]
                )
            ),

            policy=(
                relaxed_policy()
            ),
        )
    )

    assert (
        report
        .metrics
        .eligible_trajectory_count
        == 4
    )

    assert (
        report
        .metrics
        .unique_request_count
        == 4
    )

    assert (
        report.promotion_eligible
        is True
    )


def test_missing_curated_trajectory_fails_closed(
    tmp_path: Path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    write_jsonl(
        trajectory_path,
        [
            build_trajectory(
                trajectory_id="t1",
                user_request="Request one.",
                agent="developer-specialist",
                tool="workspace_git_status",
            )
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "trajectory IDs absent"
        ),
    ):

        evaluate_diversity_gate(
            trajectory_path=(
                trajectory_path
            ),

            curation_report=(
                build_curation_report(
                    [
                        "t1",
                        "missing",
                    ]
                )
            ),
        )