import json

from learning.evidence.approval_lineage import (
    find_approval_lineage,
)


def test_find_approval_lineage_recovers_exact_task(
    tmp_path,
):

    path = (
        tmp_path
        / "trajectories.jsonl"
    )

    payload = {
        "trajectory_id":
            "trajectory-1",

        "job_id":
            "job-1",

        "user_request":
            (
                "Stage tools/git/catalog.py "
                "in the AI repository."
            ),

        "steps": [
            {
                "task_id":
                    "task-1",

                "agent":
                    "developer-specialist",

                "task_instructions":
                    (
                        "Stage tools/git/catalog.py "
                        "in the AI Git repository."
                    ),

                "proposed_tool":
                    "workspace_git_stage_files",

                "proposed_arguments": {
                    "repository":
                        "ai",

                    "paths": [
                        "tools/git/catalog.py",
                    ],
                },

                "semantic_intent": {
                    "effect":
                        "mutation",
                },

                "approval_id":
                    "approval-123",
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

    lineage = (
        find_approval_lineage(
            path=(
                path
            ),

            approval_id=(
                "approval-123"
            ),
        )
    )

    assert (
        lineage
        is not None
    )

    assert (
        lineage[
            "trajectory_id"
        ]
        == "trajectory-1"
    )

    assert (
        lineage[
            "job_id"
        ]
        == "job-1"
    )

    assert (
        lineage[
            "task_id"
        ]
        == "task-1"
    )

    assert (
        lineage[
            "proposed_tool"
        ]
        == "workspace_git_stage_files"
    )

    assert (
        lineage[
            "proposed_arguments"
        ][
            "paths"
        ]
        == [
            "tools/git/catalog.py",
        ]
    )


def test_find_approval_lineage_returns_none_for_unknown_id(
    tmp_path,
):

    path = (
        tmp_path
        / "trajectories.jsonl"
    )

    path.write_text(
        "",
        encoding="utf-8",
    )

    assert (
        find_approval_lineage(
            path=(
                path
            ),

            approval_id=(
                "missing"
            ),
        )
        is None
    )
