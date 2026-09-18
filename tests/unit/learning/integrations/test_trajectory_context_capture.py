from pathlib import (
    Path,
)

from learning.context import (
    load_context_records,
)

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from learning.integrations.runtime_hooks import (
    ContinualLearningRuntimeHooks,
)

from subagents.core.definitions.types import (
    AgentResult,
    HubResult,
    SemanticIntent,
)


def _ticket_intent(
) -> SemanticIntent:

    return (
        SemanticIntent(
            summary=(
                "Read ticket OPS-42."
            ),

            effect="read",

            allowed_tools=[
                "ticket_get",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "ticket_key": [
                    "OPS-42",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )


def _git_intent(
) -> SemanticIntent:

    return (
        SemanticIntent(
            summary=(
                "Inspect the AI repository status."
            ),

            effect="read",

            allowed_tools=[
                "workspace_git_status",
            ],

            forbidden_tools=[],

            allowed_arguments={
                "repository": [
                    "ai",
                ],
            },

            forbidden_arguments={},

            max_tool_calls=1,

            clarification_required=False,
        )
    )


def _hub_result(
) -> HubResult:

    return (
        HubResult(
            status="success",

            user_request=(
                "Check OPS-42 and then inspect "
                "the AI repo status."
            ),

            routes=[
                "ticket-specialist",
                "developer-specialist",
            ],

            results=[
                AgentResult(
                    task_id=(
                        "task-ticket"
                    ),

                    agent_name=(
                        "ticket-specialist"
                    ),

                    status="success",

                    task_instructions=(
                        "Retrieve ticket OPS-42."
                    ),

                    semantic_intent=(
                        _ticket_intent()
                    ),

                    proposed_tool=(
                        "ticket_get"
                    ),

                    proposed_arguments={
                        "ticket_key":
                            "OPS-42",
                    },

                    outcome_code=(
                        "success"
                    ),

                    tool_result={
                        "ok":
                            True,

                        "status":
                            "success",

                        "ticket": {
                            "provider":
                                "jira",

                            "key":
                                "OPS-42",

                            "summary":
                                "Investigate deployment failure",

                            "status":
                                "In Progress",

                            "assignee":
                                "jdoe",

                            "project_key":
                                "OPS",
                        },
                    },

                    answer=(
                        "OPS-42 retrieved."
                    ),
                ),

                AgentResult(
                    task_id=(
                        "task-git"
                    ),

                    agent_name=(
                        "developer-specialist"
                    ),

                    status="success",

                    task_instructions=(
                        "Inspect the AI repository "
                        "working tree state."
                    ),

                    semantic_intent=(
                        _git_intent()
                    ),

                    proposed_tool=(
                        "workspace_git_status"
                    ),

                    proposed_arguments={
                        "repository":
                            "ai",
                    },

                    outcome_code=(
                        "success"
                    ),

                    tool_result={
                        "ok":
                            True,

                        "status":
                            "success",

                        "repository":
                            "ai",

                        "branch":
                            "main",

                        "clean":
                            False,

                        "ahead":
                            1,

                        "behind":
                            0,

                        "changes": [
                            {
                                "code":
                                    "M",

                                "path":
                                    "learning/evidence/recorder.py",
                            }
                        ],

                        "change_count":
                            1,
                    },

                    answer=(
                        "AI repository inspected."
                    ),
                ),
            ],

            answer=(
                "Ticket and repository inspected."
            ),
        )
    )


def test_trajectory_capture_emits_structured_context(
    tmp_path: Path,
):

    context_path = (
        tmp_path
        / "learning"
        / "continual"
        / "context-records.jsonl"
    )

    trajectory_path = (
        tmp_path
        / "learning"
        / "trajectories.jsonl"
    )

    hooks = (
        ContinualLearningRuntimeHooks(
            path=(
                context_path
            ),

            enabled=True,
        )
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                trajectory_path
            ),

            enabled=True,

            hub_model=(
                "hub-main"
            ),

            context_hooks=(
                hooks
            ),
        )
    )

    payload = (
        recorder.record(
            job_id=(
                "job-context-1"
            ),

            attempt=1,

            result=(
                _hub_result()
            ),
        )
    )

    assert (
        payload
        is not None
    )

    trajectory_id = (
        payload[
            "trajectory_id"
        ]
    )

    records = (
        load_context_records(
            context_path
        )
    )

    assert records

    assert all(
        record.trajectory_id
        == trajectory_id

        for record
        in records
    )

    assert all(
        record.authority_grant
        is False

        for record
        in records
    )

    assert all(
        record.training_eligible
        is False

        for record
        in records
    )

    kinds = {
        record.kind

        for record
        in records
    }

    assert (
        "task"
        in kinds
    )

    assert (
        "tool"
        in kinds
    )

    assert (
        "jira"
        in kinds
    )

    assert (
        "git"
        in kinds
    )


def test_second_task_receives_previous_trusted_result_context(
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

            context_hooks=(
                hooks
            ),
        )
    )

    payload = (
        recorder.record(
            job_id=(
                "job-context-2"
            ),

            attempt=1,

            result=(
                _hub_result()
            ),
        )
    )

    assert (
        payload
        is not None
    )

    records = (
        load_context_records(
            context_path
        )
    )

    git_task_context = next(
        record

        for record
        in records

        if (
            record.kind
            == "task"

            and record.task_id
            == "task-git"
        )
    )

    previous_results = (
        git_task_context
        .payload[
            "previous_trusted_results"
        ]
    )

    assert (
        len(
            previous_results
        )
        == 1
    )

    previous = (
        previous_results[
            0
        ]
    )

    assert (
        previous[
            "task_id"
        ]
        == "task-ticket"
    )

    assert (
        previous[
            "agent"
        ]
        == "ticket-specialist"
    )

    assert (
        previous[
            "tool"
        ]
        == "ticket_get"
    )

    assert (
        previous[
            "result"
        ][
            "ticket"
        ][
            "key"
        ]
        == "OPS-42"
    )


def test_semantic_contract_is_preserved_in_context(
    tmp_path: Path,
):

    context_path = (
        tmp_path
        / "context-records.jsonl"
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                tmp_path
                / "trajectories.jsonl"
            ),

            enabled=True,

            hub_model="hub-main",

            context_hooks=(
                ContinualLearningRuntimeHooks(
                    path=(
                        context_path
                    ),

                    enabled=True,
                )
            ),
        )
    )

    payload = (
        recorder.record(
            job_id=(
                "job-context-3"
            ),

            attempt=1,

            result=(
                _hub_result()
            ),
        )
    )

    assert (
        payload
        is not None
    )

    records = (
        load_context_records(
            context_path
        )
    )

    ticket_task_context = next(
        record

        for record
        in records

        if (
            record.kind
            == "task"

            and record.task_id
            == "task-ticket"
        )
    )

    semantic_contract = (
        ticket_task_context
        .payload[
            "semantic_contract"
        ]
    )

    assert (
        semantic_contract[
            "effect"
        ]
        == "read"
    )

    assert (
        semantic_contract[
            "allowed_tools"
        ]
        == [
            "ticket_get",
        ]
    )

    assert (
        semantic_contract[
            "allowed_arguments"
        ][
            "ticket_key"
        ]
        == [
            "OPS-42",
        ]
    )


def test_context_failure_cannot_destroy_durable_trajectory(
    tmp_path: Path,
):

    class BrokenHooks:
        def record_trajectory_context(
            self,
            trajectory,
        ):
            raise RuntimeError(
                "synthetic context failure"
            )

    trajectory_path = (
        tmp_path
        / "trajectories.jsonl"
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                trajectory_path
            ),

            enabled=True,

            hub_model="hub-main",

            context_hooks=(
                BrokenHooks()
            ),
        )
    )

    payload = (
        recorder.record(
            job_id=(
                "job-context-failure"
            ),

            attempt=1,

            result=(
                _hub_result()
            ),
        )
    )

    assert (
        payload
        is not None
    )

    assert (
        trajectory_path
        .is_file()
    )

    lines = (
        trajectory_path
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