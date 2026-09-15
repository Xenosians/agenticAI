import asyncio

from subagents.core.orchestrator import (
    Orchestrator,
)

from subagents.core.types import (
    AgentResult,
    SpecialistRequest,
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
                    "Inspect the frontend repository "
                    "working tree state."
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

                status=(
                    "success"
                ),

                outcome_code=(
                    "success"
                ),

                proposed_tool=(
                    "workspace_git_status"
                ),

                proposed_arguments={
                    "repository":
                        "frontend"
                },

                answer=(
                    "Frontend repository status retrieved."
                ),
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
            "Frontend repository status retrieved."
        )


def test_orchestrator_preserves_specialist_task_instructions(
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
        )
    )

    result = (
        asyncio.run(
            orchestrator.run(
                "What is the state of my frontend workspace?"
            )
        )
    )

    assert (
        len(
            result.results
        )
        == 1
    )

    assert (
        result.results[
            0
        ].task_instructions
        == (
            "Inspect the frontend repository "
            "working tree state."
        )
    )


def test_runtime_does_not_need_to_duplicate_task_context_logic(
):

    runtime_result = (
        AgentResult(
            task_id=(
                "task-1"
            ),

            agent_name=(
                "developer-specialist"
            ),

            status=(
                "success"
            ),
        )
    )

    assert (
        runtime_result.task_instructions
        is None
    )