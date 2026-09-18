from pathlib import (
    Path,
)

from learning.context import (
    load_context_records,
)

from learning.integrations.runtime_hooks import (
    ContinualLearningRuntimeHooks,
)


def test_exact_runtime_decisions_are_preferred(
    tmp_path: Path,
):

    context_path = (
        tmp_path
        / "context-records.jsonl"
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
        hooks.record_trajectory_context(
            {
                "trajectory_id":
                    "trajectory-exact-1",

                "job_id":
                    "job-exact-1",

                "user_request":
                    (
                        "Check the status of "
                        "the AI repository."
                    ),

                "steps": [
                    {
                        "task_id":
                            "task-exact-1",

                        "agent":
                            "developer-specialist",

                        "task_instructions":
                            (
                                "Inspect the AI repository "
                                "working tree."
                            ),

                        "semantic_intent": {
                            "summary":
                                (
                                    "Inspect the AI repository."
                                ),

                            "effect":
                                "read",

                            "allowed_tools": [
                                "workspace_git_status",
                            ],

                            "forbidden_tools":
                                [],

                            "allowed_arguments": {
                                "repository": [
                                    "ai",
                                ],
                            },

                            "forbidden_arguments":
                                {},

                            "max_tool_calls":
                                1,

                            "clarification_required":
                                False,
                        },

                        "proposed_tool":
                            "workspace_git_status",

                        "proposed_arguments": {
                            "repository":
                                "ai",
                        },

                        "semantic_guard_decision": {
                            "allowed":
                                True,

                            "decision_code":
                                "semantic_guard_allowed",

                            "error":
                                None,
                        },

                        "gateway_decision": {
                            "ok":
                                True,

                            "status":
                                "success",

                            "decision_code":
                                "success",

                            "tool":
                                "workspace_git_status",

                            "risk":
                                "read",
                        },

                        "status":
                            "success",

                        # Deliberately wrong historical-looking
                        # value.
                        #
                        # Exact gateway evidence must win.
                        "outcome_code":
                            "historical-wrong-code",

                        "approval_id":
                            None,

                        "error":
                            None,

                        "tool_result": {
                            "ok":
                                True,

                            "status":
                                "success",

                            "repository":
                                "ai",

                            "branch":
                                "main",

                            "clean":
                                True,

                            "ahead":
                                0,

                            "behind":
                                0,

                            "changes":
                                [],
                        },
                    }
                ],
            }
        )
    )

    assert (
        len(
            records
        )
        == 5
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
        == 5
    )

    subjects = [
        record.subject

        for record
        in persisted
    ]

    assert (
        "hub-routing-context"
        in subjects
    )

    assert (
        "specialist-proposal"
        in subjects
    )

    assert (
        "semantic-guard-outcome"
        in subjects
    )

    assert (
        "gateway:workspace_git_status"
        in subjects
    )

    guard_record = next(
        record

        for record
        in persisted

        if (
            record.subject
            == "semantic-guard-outcome"
        )
    )

    assert (
        guard_record.payload[
            "guard_outcomes"
        ][
            0
        ][
            "allowed"
        ]
        is True
    )

    assert (
        guard_record.payload[
            "guard_outcomes"
        ][
            0
        ][
            "code"
        ]
        == "semantic_guard_allowed"
    )

    gateway_record = next(
        record

        for record
        in persisted

        if (
            record.subject
            == "gateway:workspace_git_status"
        )
    )

    exact_gateway = (
        gateway_record.payload[
            "gateway_outcomes"
        ][
            0
        ]
    )

    assert (
        exact_gateway[
            "decision_code"
        ]
        == "success"
    )

    assert (
        exact_gateway[
            "risk"
        ]
        == "read"
    )

    # Most important compatibility assertion:
    #
    # Do not replace exact evidence with reconstructed outcome_code.
    assert (
        exact_gateway[
            "decision_code"
        ]
        != "historical-wrong-code"
    )


def test_historical_semantic_denial_still_reconstructs(
    tmp_path: Path,
):

    context_path = (
        tmp_path
        / "historical-context.jsonl"
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
        hooks.record_trajectory_context(
            {
                "trajectory_id":
                    "trajectory-old-1",

                "job_id":
                    "job-old-1",

                "user_request":
                    (
                        "Check the status of "
                        "the AI repository."
                    ),

                "steps": [
                    {
                        "task_id":
                            "task-old-1",

                        "agent":
                            "developer-specialist",

                        "task_instructions":
                            "Inspect the repository.",

                        "semantic_intent": {
                            "effect":
                                "read",

                            "allowed_tools": [
                                "workspace_git_status",
                            ],

                            "allowed_arguments": {
                                "repository": [
                                    "AI",
                                ],
                            },
                        },

                        "proposed_tool":
                            "workspace_git_status",

                        "proposed_arguments": {
                            "repository":
                                "ai",
                        },

                        # Historical trajectory:
                        #
                        # no semantic_guard_decision
                        # no gateway_decision
                        "status":
                            "error",

                        "outcome_code":
                            "semantic_argument_not_allowed",

                        "error":
                            (
                                "The proposed grounded argument "
                                "does not match the semantic "
                                "target."
                            ),

                        "tool_result":
                            None,
                    }
                ],
            }
        )
    )

    assert (
        len(
            records
        )
        == 3
    )

    persisted = (
        load_context_records(
            context_path
        )
    )

    guard_record = next(
        record

        for record
        in persisted

        if (
            record.subject
            == "semantic-guard-outcome"
        )
    )

    outcome = (
        guard_record.payload[
            "guard_outcomes"
        ][
            0
        ]
    )

    assert (
        outcome[
            "allowed"
        ]
        is False
    )

    assert (
        outcome[
            "code"
        ]
        == "semantic_argument_not_allowed"
    )

    # Historical semantic denial never invents a gateway event.
    assert not any(
        record.subject.startswith(
            "gateway:"
        )

        for record
        in persisted
    )


def test_exact_semantic_denial_does_not_create_gateway_event(
    tmp_path: Path,
):

    context_path = (
        tmp_path
        / "exact-denial-context.jsonl"
    )

    hooks = (
        ContinualLearningRuntimeHooks(
            path=(
                context_path
            ),

            enabled=True,
        )
    )

    hooks.record_trajectory_context(
        {
            "trajectory_id":
                "trajectory-denial-1",

            "job_id":
                "job-denial-1",

            "user_request":
                "Check bob.",

            "steps": [
                {
                    "task_id":
                        "task-denial-1",

                    "agent":
                        "account-specialist",

                    "task_instructions":
                        "Check bob.",

                    "semantic_intent": {
                        "effect":
                            "read",

                        "allowed_tools": [
                            "account_status",
                        ],

                        "allowed_arguments": {
                            "user_id": [
                                "bob",
                            ],
                        },
                    },

                    "proposed_tool":
                        "reset_password",

                    "proposed_arguments": {
                        "user_id":
                            "bob",
                    },

                    "semantic_guard_decision": {
                        "allowed":
                            False,

                        "decision_code":
                            "semantic_tool_not_allowed",

                        "error":
                            (
                                "The proposed capability does "
                                "not match the resolved semantic "
                                "intent."
                            ),
                    },

                    "gateway_decision":
                        None,

                    "status":
                        "error",

                    "outcome_code":
                        "semantic_tool_not_allowed",

                    "tool_result":
                        None,
                }
            ],
        }
    )

    persisted = (
        load_context_records(
            context_path
        )
    )

    guard_record = next(
        record

        for record
        in persisted

        if (
            record.subject
            == "semantic-guard-outcome"
        )
    )

    assert (
        guard_record.payload[
            "guard_outcomes"
        ][
            0
        ][
            "allowed"
        ]
        is False
    )

    assert not any(
        record.subject.startswith(
            "gateway:"
        )

        for record
        in persisted
    )