import json

from pathlib import (
    Path,
)

from learning.recorder import (
    TrajectoryRecorder,
)

from learning.rewards import (
    derive_execution_reward,
)

from learning.sanitizer import (
    sanitize_value,
)

from subagents.core.types import (
    AgentResult,
    HubResult,
)


def successful_hub_result(
) -> HubResult:

    return (
        HubResult(
            status="success",

            user_request=(
                "Show me the frontend git status."
            ),

            routes=[
                "developer-specialist",
            ],

            results=[
                AgentResult(
                    task_id="task-1",

                    agent_name=(
                        "developer-specialist"
                    ),

                    status="success",

                    proposed_tool=(
                        "workspace_git_status"
                    ),

                    proposed_arguments={
                        "repository":
                            "frontend",
                    },

                    tool_result={
                        "ok":
                            True,

                        "status":
                            "success",

                        "repository":
                            "frontend",

                        "branch":
                            "main",

                        "clean":
                            True,
                    },

                    answer=(
                        "frontend is on main."
                    ),
                )
            ],

            answer=(
                "The frontend repository "
                "is clean on main."
            ),
        )
    )


def test_sanitizer_redacts_sensitive_keys():
    payload = {
        "jira_api_token":
            "super-secret-value",

        "nested": {
            "password":
                "hunter2",

            "safe":
                "frontend",
        },
    }

    sanitized = (
        sanitize_value(
            payload
        )
    )

    assert (
        sanitized[
            "jira_api_token"
        ]
        == "<redacted>"
    )

    assert (
        sanitized[
            "nested"
        ][
            "password"
        ]
        == "<redacted>"
    )

    assert (
        sanitized[
            "nested"
        ][
            "safe"
        ]
        == "frontend"
    )


def test_sanitizer_redacts_secret_text():
    payload = (
        "Authorization: Bearer abc123 "
        "email=user@example.com "
        "password=hunter2"
    )

    sanitized = (
        sanitize_value(
            payload
        )
    )

    assert (
        "abc123"
        not in sanitized
    )

    assert (
        "user@example.com"
        not in sanitized
    )

    assert (
        "hunter2"
        not in sanitized
    )


def test_execution_reward_is_not_quality_label():
    result = (
        successful_hub_result()
    )

    reward = (
        derive_execution_reward(
            result
        )
    )

    assert (
        reward.total
        > 0
    )

    assert (
        reward.quality_eligible
        is False
    )


def test_recorder_writes_jsonl(
    tmp_path: Path,
):
    output = (
        tmp_path
        / "learning"
        / "trajectories.jsonl"
    )

    recorder = (
        TrajectoryRecorder(
            path=output,

            enabled=True,

            hub_model=(
                "hub-main"
            ),
        )
    )

    payload = (
        recorder.record(
            job_id="job-1",

            attempt=1,

            result=(
                successful_hub_result()
            ),
        )
    )

    assert (
        payload
        is not None
    )

    assert (
        output.exists()
    )

    lines = (
        output
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    assert (
        len(
            lines
        )
        == 1
    )

    stored = (
        json.loads(
            lines[
                0
            ]
        )
    )

    assert (
        stored[
            "schema"
        ]
        == "trajectory.v1"
    )

    assert (
        stored[
            "job_id"
        ]
        == "job-1"
    )

    assert (
        stored[
            "routes"
        ]
        == [
            "developer-specialist",
        ]
    )

    assert (
        stored[
            "steps"
        ][
            0
        ][
            "proposed_arguments"
        ][
            "repository"
        ]
        == "frontend"
    )

    assert (
        stored[
            "dataset_eligible"
        ]
        is False
    )


def test_disabled_recorder_writes_nothing(
    tmp_path: Path,
):
    output = (
        tmp_path
        / "trajectories.jsonl"
    )

    recorder = (
        TrajectoryRecorder(
            path=output,

            enabled=False,

            hub_model="hub-main",
        )
    )

    result = (
        recorder.record(
            job_id="job-1",

            attempt=1,

            result=(
                successful_hub_result()
            ),
        )
    )

    assert (
        result
        is None
    )

    assert not (
        output.exists()
    )