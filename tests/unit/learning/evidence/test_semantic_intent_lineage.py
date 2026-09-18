import asyncio

from learning.evidence.recorder import (
    TrajectoryRecorder,
)

from subagents.core.definitions.types import (
    AgentResult,
    SemanticIntent,
    SpecialistRequest,
)

from subagents.core.orchestration.orchestrator import (
    Orchestrator,
)


INTENT = (
    SemanticIntent(
        summary=(
            "Inspect the AI repository working tree."
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


class FakeRouter:
    async def route(
        self,
        user_request: str,
    ):
        return [
            SpecialistRequest(
                agent_name=(
                    "developer-specialist"
                ),

                instructions=(
                    "Inspect the AI repository "
                    "working tree state."
                ),

                semantic_intent=(
                    INTENT
                ),
            )
        ]


class FakeRuntime:
    async def run(
        self,
        task,
    ) -> AgentResult:

        return (
            AgentResult(
                task_id=(
                    task.task_id
                ),

                agent_name=(
                    task.agent_name
                ),

                status="success",

                proposed_tool=(
                    "workspace_git_status"
                ),

                proposed_arguments={
                    "repository":
                        "ai",
                },

                outcome_code="success",

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
                        True,
                },
            )
        )


class FakePrimaryAssistant:
    async def respond(
        self,
        user_request: str,
    ) -> str:

        return (
            "unused"
        )

    async def synthesize(
        self,
        user_request: str,
        results,
    ) -> str:

        return (
            "AI repository inspected."
        )


def test_semantic_intent_survives_orchestrator_and_trajectory(
    tmp_path,
):
    orchestrator = (
        Orchestrator(
            router=(
                FakeRouter()
            ),

            runtime=(
                FakeRuntime()
            ),

            primary_assistant=(
                FakePrimaryAssistant()
            ),

            require_semantic_intent=True,
        )
    )

    hub_result = (
        asyncio.run(
            orchestrator.run(
                "Check the AI repo status."
            )
        )
    )

    assert (
        len(
            hub_result.results
        )
        == 1
    )

    result = (
        hub_result.results[
            0
        ]
    )

    assert (
        result.semantic_intent
        == INTENT
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                tmp_path
                / "trajectories.jsonl"
            ),

            enabled=True,

            hub_model="hub-main",
        )
    )

    trajectory = (
        recorder.build(
            job_id=(
                "job-semantic-1"
            ),

            attempt=1,

            result=(
                hub_result
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
        step.semantic_intent
        is not None
    )

    assert (
        step.semantic_intent[
            "summary"
        ]
        == (
            "Inspect the AI repository "
            "working tree."
        )
    )

    assert (
        step.semantic_intent[
            "effect"
        ]
        == "read"
    )

    assert (
        step.semantic_intent[
            "allowed_tools"
        ]
        == [
            "workspace_git_status",
        ]
    )

    assert (
        step.semantic_intent[
            "allowed_arguments"
        ][
            "repository"
        ]
        == [
            "ai",
        ]
    )

    assert (
        step.semantic_intent[
            "max_tool_calls"
        ]
        == 1
    )

    assert (
        step.semantic_intent[
            "clarification_required"
        ]
        is False
    )

    assert (
        step.proposed_tool
        == "workspace_git_status"
    )

    assert (
        step.proposed_arguments
        == {
            "repository":
                "ai",
        }
    )


def test_semantic_intent_is_evidence_not_runtime_authority(
    tmp_path,
):
    orchestrator = (
        Orchestrator(
            router=(
                FakeRouter()
            ),

            runtime=(
                FakeRuntime()
            ),

            primary_assistant=(
                FakePrimaryAssistant()
            ),

            require_semantic_intent=True,
        )
    )

    hub_result = (
        asyncio.run(
            orchestrator.run(
                "Check the AI repo status."
            )
        )
    )

    recorder = (
        TrajectoryRecorder(
            path=(
                tmp_path
                / "trajectories.jsonl"
            ),

            enabled=True,

            hub_model="hub-main",
        )
    )

    trajectory = (
        recorder.build(
            job_id=(
                "job-semantic-2"
            ),

            attempt=1,

            result=(
                hub_result
            ),
        )
    )

    step = (
        trajectory.steps[
            0
        ]
    )

    assert (
        isinstance(
            step.semantic_intent,
            dict,
        )
    )

    assert (
        "authority_grant"
        not in step.semantic_intent
    )

    assert (
        trajectory.dataset_eligible
        is False
    )