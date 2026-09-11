import asyncio

from subagents.core.orchestrator import (
    Orchestrator,
)

from subagents.core.types import (
    AgentResult,
)


class FakeRouter:
    def __init__(
        self,
    ):
        self.routes = [
            "account-specialist"
        ]

    def route(
        self,
        user_request: str,
    ) -> list[str]:
        return self.routes


class FakeRuntime:
    def __init__(
        self,
    ):
        self.tasks = []

    async def run(
        self,
        task,
    ) -> AgentResult:
        self.tasks.append(
            task
        )

        return AgentResult(
            task_id=task.task_id,

            agent_name=(
                task.agent_name
            ),

            status="success",

            proposed_tool=(
                "account_status"
            ),

            proposed_arguments={
                "user_id":
                    "jdoe"
            },

            answer=(
                "{'ok': True, "
                "'user_id': 'jdoe', "
                "'locked': False}"
            ),
        )


class FakePrimaryAssistant:
    def __init__(
        self,
    ):
        self.requests = []

    def respond(
        self,
        user_request: str,
    ) -> str:
        self.requests.append(
            user_request
        )

        return (
            "This is a primary "
            "assistant response."
        )


def build_orchestrator(
    router=None,
    runtime=None,
    primary_assistant=None,
):
    if router is None:
        router = FakeRouter()

    if runtime is None:
        runtime = FakeRuntime()

    if primary_assistant is None:
        primary_assistant = (
            FakePrimaryAssistant()
        )

    orchestrator = Orchestrator(
        router=router,

        runtime=runtime,

        primary_assistant=(
            primary_assistant
        ),
    )

    return (
        orchestrator,
        router,
        runtime,
        primary_assistant,
    )


def test_orchestrator_routes_to_worker():
    (
        orchestrator,
        _router,
        runtime,
        primary_assistant,
    ) = build_orchestrator()

    result = asyncio.run(
        orchestrator.run(
            "Is jdoe locked?"
        )
    )

    assert (
        result.status
        == "success"
    )

    assert result.routes == [
        "account-specialist"
    ]

    assert (
        len(
            result.results
        )
        == 1
    )

    assert (
        result.results[
            0
        ].agent_name
        == "account-specialist"
    )

    assert (
        result.results[
            0
        ].proposed_tool
        == "account_status"
    )

    assert (
        len(
            runtime.tasks
        )
        == 1
    )

    assert (
        runtime.tasks[
            0
        ].user_request
        == "Is jdoe locked?"
    )

    # A routed request should not use
    # the primary conversational path.
    assert (
        primary_assistant.requests
        == []
    )


def test_orchestrator_handles_no_route():
    router = FakeRouter()

    router.routes = []

    runtime = FakeRuntime()

    primary_assistant = (
        FakePrimaryAssistant()
    )

    (
        orchestrator,
        _router,
        _runtime,
        _primary_assistant,
    ) = build_orchestrator(
        router=router,

        runtime=runtime,

        primary_assistant=(
            primary_assistant
        ),
    )

    result = asyncio.run(
        orchestrator.run(
            "Tell me a joke."
        )
    )

    # No specialist route now means:
    #
    # primary conversational assistant
    # handles the request successfully.
    assert (
        result.status
        == "success"
    )

    assert (
        result.routes
        == []
    )

    assert (
        result.results
        == []
    )

    assert (
        result.answer
        == (
            "This is a primary "
            "assistant response."
        )
    )

    assert (
        primary_assistant.requests
        == [
            "Tell me a joke."
        ]
    )

    # No specialist work should run.
    assert (
        runtime.tasks
        == []
    )


def test_orchestrator_handles_multiple_workers():
    router = FakeRouter()

    router.routes = [
        "account-specialist",
        "access-specialist",
    ]

    runtime = FakeRuntime()

    primary_assistant = (
        FakePrimaryAssistant()
    )

    (
        orchestrator,
        _router,
        _runtime,
        _primary_assistant,
    ) = build_orchestrator(
        router=router,

        runtime=runtime,

        primary_assistant=(
            primary_assistant
        ),
    )

    result = asyncio.run(
        orchestrator.run(
            "Check whether jdoe is locked "
            "and whether he has VPN access."
        )
    )

    assert (
        len(
            result.routes
        )
        == 2
    )

    assert (
        len(
            result.results
        )
        == 2
    )

    assert (
        len(
            runtime.tasks
        )
        == 2
    )

    # Routed requests should not use
    # the primary conversational path.
    assert (
        primary_assistant.requests
        == []
    )


def test_orchestrator_propagates_approval():
    router = FakeRouter()

    class ApprovalRuntime:
        async def run(
            self,
            task,
        ):
            return AgentResult(
                task_id=(
                    task.task_id
                ),

                agent_name=(
                    task.agent_name
                ),

                status=(
                    "approval_required"
                ),

                proposed_tool=(
                    "unlock_user"
                ),

                proposed_arguments={
                    "user_id":
                        "jdoe"
                },

                answer=(
                    "Approval required: "
                    "approval-123"
                ),
            )

    primary_assistant = (
        FakePrimaryAssistant()
    )

    (
        orchestrator,
        _router,
        _runtime,
        _primary_assistant,
    ) = build_orchestrator(
        router=router,

        runtime=(
            ApprovalRuntime()
        ),

        primary_assistant=(
            primary_assistant
        ),
    )

    result = asyncio.run(
        orchestrator.run(
            "Unlock jdoe"
        )
    )

    assert (
        result.status
        == "approval_required"
    )

    assert (
        result.results[
            0
        ].proposed_tool
        == "unlock_user"
    )

    assert (
        result.results[
            0
        ].proposed_arguments
        == {
            "user_id":
                "jdoe"
        }
    )

    # Approval-producing specialist requests
    # should not fall through to the primary
    # conversational assistant.
    assert (
        primary_assistant.requests
        == []
    )