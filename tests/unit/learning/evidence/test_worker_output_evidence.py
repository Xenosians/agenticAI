from pathlib import (
    Path,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from learning.evidence.types import (
    TrajectoryStep,
)

from subagents.core.definitions.types import (
    AgentResult,
    HubResult,
)


RAW_MULTI_CALL = """
[
    {
        "name": "reset_password",
        "arguments": {
            "user_id": "jdoe"
        }
    },
    {
        "name": "reset_password",
        "arguments": {
            "user_id": "alice"
        }
    }
]
"""


PARSED_MULTI_CALL = [
    {
        "name":
            "reset_password",

        "arguments": {
            "user_id":
                "jdoe",
        },
    },

    {
        "name":
            "reset_password",

        "arguments": {
            "user_id":
                "alice",
        },
    },
]


def build_multi_call_result(
) -> HubResult:

    return (
        HubResult(
            status=(
                "partial_error"
            ),

            user_request=(
                "Reset jdoe's password, "
                "not alice's."
            ),

            routes=[
                "account-specialist",
            ],

            results=[
                AgentResult(
                    task_id=(
                        "task-multi-call"
                    ),

                    agent_name=(
                        "account-specialist"
                    ),

                    status=(
                        "error"
                    ),

                    task_instructions=(
                        "Reset the password "
                        "for jdoe only."
                    ),

                    raw_model_output=(
                        RAW_MULTI_CALL
                    ),

                    proposed_tool_calls=(
                        PARSED_MULTI_CALL
                    ),

                    outcome_code=(
                        "invalid_tool_call_count"
                    ),

                    error=(
                        "Worker must return exactly "
                        "one tool call for this "
                        "runtime version."
                    ),
                )
            ],

            answer=(
                "Worker must return exactly "
                "one tool call for this "
                "runtime version."
            ),
        )
    )


def test_recorder_build_preserves_multi_call_evidence(
    tmp_path: Path,
):

    recorder = (
        TrajectoryRecorder(
            path=(
                tmp_path
                / "trajectories.jsonl"
            ),

            enabled=True,

            hub_model=(
                "hub-main"
            ),
        )
    )

    trajectory = (
        recorder.build(
            job_id=(
                "job-multi-call"
            ),

            attempt=1,

            result=(
                build_multi_call_result()
            ),
        )
    )

    assert (
        len(
            trajectory.steps
        )
        == 1
    )

    step = (
        trajectory.steps[
            0
        ]
    )

    assert (
        step.outcome_code
        == "invalid_tool_call_count"
    )

    assert (
        step.raw_model_output
        == RAW_MULTI_CALL
    )

    assert (
        step.proposed_tool_calls
        == PARSED_MULTI_CALL
    )

    assert (
        step.proposed_tool
        is None
    )

    assert (
        step.proposed_arguments
        is None
    )


def test_recorder_persists_multi_call_evidence(
    tmp_path: Path,
):

    output = (
        tmp_path
        / "trajectories.jsonl"
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                output
            ),

            enabled=True,

            hub_model=(
                "hub-main"
            ),
        )
    )

    payload = (
        recorder.record(
            job_id=(
                "job-multi-call"
            ),

            attempt=1,

            result=(
                build_multi_call_result()
            ),
        )
    )

    assert (
        payload
        is not None
    )

    step = (
        payload[
            "steps"
        ][
            0
        ]
    )

    assert (
        step[
            "raw_model_output"
        ]
        == RAW_MULTI_CALL
    )

    assert (
        step[
            "proposed_tool_calls"
        ]
        == PARSED_MULTI_CALL
    )

    assert (
        step[
            "proposed_tool"
        ]
        is None
    )

    assert (
        step[
            "proposed_arguments"
        ]
        is None
    )


def test_legacy_trajectory_step_remains_readable(
):

    step = (
        TrajectoryStep.model_validate(
            {
                "task_id":
                    "legacy-task",

                "agent":
                    "account-specialist",

                "status":
                    "error",

                "outcome_code":
                    "invalid_tool_call_count",

                "proposed_tool":
                    None,

                "proposed_arguments":
                    None,
            }
        )
    )

    assert (
        step.raw_model_output
        is None
    )

    assert (
        step.proposed_tool_calls
        is None
    )
