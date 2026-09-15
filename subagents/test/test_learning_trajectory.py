import json

from pathlib import (
    Path,
)

from learning.quality import (
    derive_trajectory_quality,
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

from learning.types import (
    TrajectoryStep,
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

                    outcome_code=(
                        "success"
                    ),

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


def test_success_quality_is_deterministic_but_unreviewed():
    quality = (
        derive_trajectory_quality(
            [
                TrajectoryStep(
                    task_id="task-1",
                    agent="developer-specialist",
                    status="success",
                    outcome_code="success",
                    proposed_tool=(
                        "workspace_git_status"
                    ),
                    proposed_arguments={
                        "repository":
                            "frontend",
                    },
                )
            ]
        )
    )

    assert (
        quality.grounding_valid
        is True
    )

    assert (
        quality.gateway_policy_passed
        is True
    )

    assert (
        quality.tool_execution_valid
        is True
    )

    assert (
        quality.route_correct
        is None
    )

    assert (
        quality.tool_correct
        is None
    )

    assert (
        quality.answer_grounded
        is None
    )

    assert (
        quality.quality_eligible
        is False
    )


def test_grounding_failure_is_classified():
    quality = (
        derive_trajectory_quality(
            [
                TrajectoryStep(
                    task_id="task-1",
                    agent="developer-specialist",
                    status="error",
                    outcome_code=(
                        "grounding_failed"
                    ),
                    proposed_tool=(
                        "workspace_git_status"
                    ),
                    proposed_arguments={
                        "repository":
                            "ai",
                    },
                )
            ]
        )
    )

    assert (
        quality.grounding_valid
        is False
    )

    assert (
        quality.gateway_policy_passed
        is None
    )

    assert (
        quality.tool_execution_valid
        is None
    )

    assert (
        quality.failure_types
        == [
            "grounding_failure"
        ]
    )


def test_policy_denial_is_classified():
    quality = (
        derive_trajectory_quality(
            [
                TrajectoryStep(
                    task_id="task-1",
                    agent="developer-specialist",
                    status="error",
                    outcome_code=(
                        "policy_denied"
                    ),
                )
            ]
        )
    )

    assert (
        quality.grounding_valid
        is True
    )

    assert (
        quality.gateway_policy_passed
        is False
    )

    assert (
        quality.failure_types
        == [
            "policy_denial"
        ]
    )


def test_recorder_writes_quality_and_alias_schema(
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

    lines = (
        output
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
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
            "execution_reward"
        ][
            "schema"
        ]
        == "execution-reward.v1"
    )

    assert (
        stored[
            "quality"
        ][
            "schema"
        ]
        == "trajectory-quality.v1"
    )

    assert (
        stored[
            "steps"
        ][
            0
        ][
            "outcome_code"
        ]
        == "success"
    )

    assert (
        stored[
            "quality"
        ][
            "grounding_valid"
        ]
        is True
    )

    assert (
        stored[
            "quality"
        ][
            "tool_correct"
        ]
        is None
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