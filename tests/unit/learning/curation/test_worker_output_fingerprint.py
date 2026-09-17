from __future__ import annotations

import hashlib
import json

from copy import (
    deepcopy,
)

from learning.curation.engine import (
    evidence_fingerprint,
)

from learning.curation.corpus_analysis import (
    normalize_request,
)

from learning.evidence.types import (
    ExecutionReward,
    LearningTrajectory,
    TrajectorySignals,
    TrajectoryStep,
)


def build_trajectory(
    *,
    trajectory_id: str,
    step: TrajectoryStep,
) -> LearningTrajectory:

    return (
        LearningTrajectory(
            trajectory_id=(
                trajectory_id
            ),

            observed_at=(
                "2026-09-17T00:00:00+00:00"
            ),

            job_id=(
                f"job-{trajectory_id}"
            ),

            attempt=1,

            user_request=(
                "Reset jdoe's password, "
                "not alice's."
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "partial_error"
            ),

            routes=[
                "account-specialist",
            ],

            steps=[
                step
            ],

            final_answer=(
                "Worker must return exactly "
                "one tool call."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,
                    route_count=1,
                    specialist_count=1,
                    specialist_error_count=1,
                    had_error=True,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    total=-1.0,
                    quality_eligible=False,
                )
            ),

            dataset_eligible=True,
        )
    )


def build_multicall_step(
    *,
    second_user: str,
    raw_model_output: str,
) -> TrajectoryStep:

    return (
        TrajectoryStep(
            task_id=(
                "task-multi"
            ),

            task_instructions=(
                "Reset the password for jdoe only."
            ),

            agent=(
                "account-specialist"
            ),

            status=(
                "error"
            ),

            outcome_code=(
                "invalid_tool_call_count"
            ),

            raw_model_output=(
                raw_model_output
            ),

            proposed_tool_calls=[
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
                            second_user,
                    },
                },
            ],

            proposed_tool=None,

            proposed_arguments=None,

            error=(
                "Worker must return exactly "
                "one tool call."
            ),
        )
    )


def test_multicall_fingerprint_uses_complete_call_set(
):

    first = (
        build_trajectory(
            trajectory_id=(
                "trajectory-1"
            ),

            step=(
                build_multicall_step(
                    second_user=(
                        "alice"
                    ),

                    raw_model_output=(
                        '[{"name":"reset_password",'
                        '"arguments":{"user_id":"jdoe"}},'
                        '{"name":"reset_password",'
                        '"arguments":{"user_id":"alice"}}]'
                    ),
                )
            ),
        )
    )

    second = (
        build_trajectory(
            trajectory_id=(
                "trajectory-2"
            ),

            step=(
                build_multicall_step(
                    second_user=(
                        "bob"
                    ),

                    raw_model_output=(
                        '[{"name":"reset_password",'
                        '"arguments":{"user_id":"jdoe"}},'
                        '{"name":"reset_password",'
                        '"arguments":{"user_id":"bob"}}]'
                    ),
                )
            ),
        )
    )

    assert (
        evidence_fingerprint(
            first
        )
        !=
        evidence_fingerprint(
            second
        )
    )


def test_multicall_fingerprint_ignores_raw_json_formatting(
):

    compact = (
        build_trajectory(
            trajectory_id=(
                "trajectory-compact"
            ),

            step=(
                build_multicall_step(
                    second_user=(
                        "alice"
                    ),

                    raw_model_output=(
                        '[{"name":"reset_password",'
                        '"arguments":{"user_id":"jdoe"}},'
                        '{"name":"reset_password",'
                        '"arguments":{"user_id":"alice"}}]'
                    ),
                )
            ),
        )
    )

    pretty = (
        build_trajectory(
            trajectory_id=(
                "trajectory-pretty"
            ),

            step=(
                build_multicall_step(
                    second_user=(
                        "alice"
                    ),

                    raw_model_output=(
                        """
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
                    ),
                )
            ),
        )
    )

    assert (
        evidence_fingerprint(
            compact
        )
        ==
        evidence_fingerprint(
            pretty
        )
    )


def test_parse_error_fingerprint_uses_raw_worker_output(
):

    first = (
        build_trajectory(
            trajectory_id=(
                "parse-1"
            ),

            step=(
                TrajectoryStep(
                    task_id=(
                        "task-parse"
                    ),

                    task_instructions=(
                        "Check jdoe."
                    ),

                    agent=(
                        "account-specialist"
                    ),

                    status=(
                        "error"
                    ),

                    outcome_code=(
                        "tool_parse_error"
                    ),

                    raw_model_output=(
                        "Call account_status for jdoe."
                    ),

                    error=(
                        "Worker returned invalid JSON."
                    ),
                )
            ),
        )
    )

    second = (
        build_trajectory(
            trajectory_id=(
                "parse-2"
            ),

            step=(
                TrajectoryStep(
                    task_id=(
                        "task-parse"
                    ),

                    task_instructions=(
                        "Check jdoe."
                    ),

                    agent=(
                        "account-specialist"
                    ),

                    status=(
                        "error"
                    ),

                    outcome_code=(
                        "tool_parse_error"
                    ),

                    raw_model_output=(
                        "I cannot decide which tool to call."
                    ),

                    error=(
                        "Worker returned invalid JSON."
                    ),
                )
            ),
        )
    )

    assert (
        evidence_fingerprint(
            first
        )
        !=
        evidence_fingerprint(
            second
        )
    )


def test_single_tool_fingerprint_keeps_legacy_payload_shape(
):

    trajectory = (
        LearningTrajectory(
            trajectory_id=(
                "legacy-compatible"
            ),

            observed_at=(
                "2026-09-17T00:00:00+00:00"
            ),

            job_id=(
                "job-legacy-compatible"
            ),

            attempt=1,

            user_request=(
                "Is jdoe unlocked?"
            ),

            hub_model=(
                "hub-main"
            ),

            hub_status=(
                "partial_error"
            ),

            routes=[
                "account-specialist",
            ],

            steps=[
                TrajectoryStep(
                    task_id=(
                        "task-single"
                    ),

                    task_instructions=(
                        "Check whether jdoe is unlocked."
                    ),

                    agent=(
                        "account-specialist"
                    ),

                    status=(
                        "error"
                    ),

                    outcome_code=(
                        "agent_tool_not_allowed"
                    ),

                    proposed_tool=(
                        "unlock_user"
                    ),

                    proposed_arguments={
                        "user_id":
                            "jdoe",
                    },
                )
            ],

            final_answer=(
                "Denied."
            ),

            signals=(
                TrajectorySignals(
                    delegated=True,
                    route_count=1,
                    specialist_count=1,
                    specialist_error_count=1,
                    tool_proposed_count=1,
                    had_error=True,
                )
            ),

            execution_reward=(
                ExecutionReward(
                    total=-1.0,
                    quality_eligible=False,
                )
            ),

            dataset_eligible=True,
        )
    )

    legacy_payload = {
        "user_request":
            normalize_request(
                trajectory.user_request
            ),

        "hub_status":
            trajectory.hub_status,

        "routes":
            list(
                trajectory.routes
            ),

        "steps": [
            {
                "task_instructions":
                    normalize_request(
                        trajectory.steps[
                            0
                        ].task_instructions
                    ),

                "agent":
                    "account-specialist",

                "status":
                    "error",

                "outcome_code":
                    "agent_tool_not_allowed",

                "proposed_tool":
                    "unlock_user",

                "proposed_arguments": {
                    "user_id":
                        "jdoe",
                },
            }
        ],

        "final_answer":
            "Denied.",
    }

    serialized = (
        json.dumps(
            legacy_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )
    )

    expected = (
        hashlib
        .sha256(
            serialized.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )

    assert (
        evidence_fingerprint(
            trajectory
        )
        == expected
    )


def test_fingerprint_does_not_mutate_call_evidence(
):

    trajectory = (
        build_trajectory(
            trajectory_id=(
                "no-mutation"
            ),

            step=(
                build_multicall_step(
                    second_user=(
                        "alice"
                    ),

                    raw_model_output=(
                        "formatted raw output"
                    ),
                )
            ),
        )
    )

    before = (
        deepcopy(
            trajectory.steps[
                0
            ].proposed_tool_calls
        )
    )

    evidence_fingerprint(
        trajectory
    )

    assert (
        trajectory.steps[
            0
        ].proposed_tool_calls
        == before
    )
