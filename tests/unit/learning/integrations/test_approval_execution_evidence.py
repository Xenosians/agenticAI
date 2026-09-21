import json

from learning.context import (
    load_context_records,
)

from learning.integrations.approval_execution import (
    capture_approval_execution_evidence,
)

from learning.integrations.runtime_hooks import (
    ContinualLearningRuntimeHooks,
)


def build_trajectory(
    path,
):

    payload = {
        "trajectory_id":
            "trajectory-stage-1",

        "job_id":
            "job-stage-1",

        "user_request":
            (
                "Stage tools/git/catalog.py "
                "in the AI repository."
            ),

        "steps": [
            {
                "task_id":
                    "task-stage-1",

                "agent":
                    "developer-specialist",

                "task_instructions":
                    (
                        "Stage tools/git/catalog.py "
                        "in the AI Git repository."
                    ),

                "semantic_intent": {
                    "effect":
                        "mutation",

                    "allowed_tools": [
                        "workspace_git_stage_files",
                    ],
                },

                "proposed_tool":
                    "workspace_git_stage_files",

                "proposed_arguments": {
                    "repository":
                        "ai",

                    "paths": [
                        "tools/git/catalog.py",
                    ],
                },

                "approval_id":
                    "approval-stage-1",
            }
        ],
    }

    path.write_text(
        json.dumps(
            payload
        )
        + "\n",

        encoding="utf-8",
    )


def test_approved_git_execution_is_linked_to_original_trajectory(
    tmp_path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    context_path = (
        tmp_path
        / "context.jsonl"
    )

    build_trajectory(
        trajectory_path
    )

    hooks = (
        ContinualLearningRuntimeHooks(
            path=(
                context_path
            ),

            enabled=True,
        )
    )

    provider_result = {
        "ok":
            True,

        "status":
            "success",

        "repository":
            "ai",

        "requested_paths": [
            "tools/git/catalog.py",
        ],

        "files": [
            "tools/git/catalog.py",
        ],

        "staged_paths": [
            "tools/git/catalog.py",
        ],

        "staged_count":
            1,

        "verification_ok":
            True,
    }

    records = (
        capture_approval_execution_evidence(
            hooks=(
                hooks
            ),

            trajectory_path=(
                trajectory_path
            ),

            approval_id=(
                "approval-stage-1"
            ),

            approval_result={
                "ok":
                    True,

                "replayed":
                    False,

                "approval": {
                    "id":
                        "approval-stage-1",

                    "tool":
                        "workspace_git_stage_files",

                    "arguments": {
                        "repository":
                            "ai",

                        "paths": [
                            "tools/git/catalog.py",
                        ],
                    },

                    "risk":
                        "low",

                    "status":
                        "approved",
                },

                "result":
                    provider_result,
            },
        )
    )

    assert (
        len(
            records
        )
        == 2
    )

    persisted = (
        load_context_records(
            context_path
        )
    )

    assert (
        len(
            persisted
        )
        == 2
    )

    subjects = {
        item.subject

        for item
        in persisted
    }

    assert (
        "approval-execution:"
        "workspace_git_stage_files"
        in subjects
    )

    assert (
        "repository:ai"
        in subjects
    )

    for record in persisted:

        assert (
            record.trajectory_id
            == "trajectory-stage-1"
        )

        assert (
            record.job_id
            == "job-stage-1"
        )

        assert (
            record.task_id
            == "task-stage-1"
        )

        assert (
            record.authority_grant
            is False
        )

        assert (
            record.training_eligible
            is False
        )

    execution_record = next(
        record

        for record
        in persisted

        if record.subject.startswith(
            "approval-execution:"
        )
    )

    approval_outcome = (
        execution_record
        .payload[
            "approval_outcomes"
        ][
            0
        ]
    )

    assert (
        approval_outcome[
            "approval_id"
        ]
        == "approval-stage-1"
    )

    assert (
        approval_outcome[
            "status"
        ]
        == "approved"
    )

    assert (
        approval_outcome[
            "provider_ok"
        ]
        is True
    )

    assert (
        execution_record
        .payload[
            "trusted_provider_results"
        ][
            0
        ][
            "verification_ok"
        ]
        is True
    )

    git_record = next(
        record

        for record
        in persisted

        if (
            record.subject
            == "repository:ai"
        )
    )

    assert (
        git_record
        .payload[
            "changed_files"
        ]
        == [
            "tools/git/catalog.py",
        ]
    )


def test_replayed_approval_does_not_duplicate_evidence(
    tmp_path,
):

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    context_path = (
        tmp_path
        / "context.jsonl"
    )

    build_trajectory(
        trajectory_path
    )

    hooks = (
        ContinualLearningRuntimeHooks(
            path=(
                context_path
            ),

            enabled=True,
        )
    )

    records = (
        capture_approval_execution_evidence(
            hooks=(
                hooks
            ),

            trajectory_path=(
                trajectory_path
            ),

            approval_id=(
                "approval-stage-1"
            ),

            approval_result={
                "ok":
                    True,

                "replayed":
                    True,

                "approval": {
                    "id":
                        "approval-stage-1",
                },

                "result": {
                    "ok":
                        True,

                    "status":
                        "success",
                },
            },
        )
    )

    assert (
        records
        == []
    )

    assert not (
        context_path.exists()
    )
