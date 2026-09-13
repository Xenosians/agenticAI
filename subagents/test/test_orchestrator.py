import asyncio

from subagents.core.orchestrator import (
    Orchestrator,
)

from subagents.core.types import (
    AgentResult,
    SpecialistRequest,
)


class FakeRouter:
    def __init__(
        self,
    ):
        self.delegations = [
            SpecialistRequest(
                agent_name=(
                    "account-specialist"
                ),
                instructions=(
                    "Determine whether "
                    "jdoe is locked."
                ),
            )
        ]

        self.requests = []

    async def route(
        self,
        user_request: str,
    ):
        self.requests.append(
            user_request
        )

        return self.delegations


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
            task_id=(
                task.task_id
            ),

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
                "Account jdoe is enabled "
                "and is not locked."
            ),
        )


class FakePrimaryAssistant:
    def __init__(
        self,
    ):
        self.requests = []
        self.synthesis_calls = []

        self.synthesis_response = (
            "jdoe is not locked."
        )

    async def respond(
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

    async def synthesize(
        self,
        user_request,
        results,
    ) -> str:
        self.synthesis_calls.append(
            {
                "user_request":
                    user_request,

                "results":
                    results,
            }
        )

        return (
            self.synthesis_response
        )


def build_orchestrator(
    router=None,
    runtime=None,
    primary_assistant=None,
):
    if router is None:
        router = (
            FakeRouter()
        )

    if runtime is None:
        runtime = (
            FakeRuntime()
        )

    if primary_assistant is None:
        primary_assistant = (
            FakePrimaryAssistant()
        )

    orchestrator = (
        Orchestrator(
            router=router,
            runtime=runtime,
            primary_assistant=(
                primary_assistant
            ),
        )
    )

    return (
        orchestrator,
        router,
        runtime,
        primary_assistant,
    )


def test_orchestrator_runs_main_specialist_main_flow():
    (
        orchestrator,
        router,
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
        runtime.tasks[
            0
        ].agent_name
        == "account-specialist"
    )

    assert (
        runtime.tasks[
            0
        ].user_request
        == "Is jdoe locked?"
    )

    assert (
        runtime.tasks[
            0
        ].instructions
        == (
            "Determine whether "
            "jdoe is locked."
        )
    )

    assert router.requests == [
        "Is jdoe locked?"
    ]

    # Specialist success must return through the Main model.
    assert (
        len(
            primary_assistant
            .synthesis_calls
        )
        == 1
    )

    assert (
        primary_assistant
        .synthesis_calls[
            0
        ][
            "user_request"
        ]
        == "Is jdoe locked?"
    )

    assert (
        result.answer
        == "jdoe is not locked."
    )

    # Direct-response mode was not used.
    assert (
        primary_assistant.requests
        == []
    )


def test_orchestrator_handles_no_route():
    router = (
        FakeRouter()
    )

    router.delegations = []

    runtime = (
        FakeRuntime()
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

    assert (
        result.status
        == "success"
    )

    assert result.routes == []
    assert result.results == []

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

    assert (
        primary_assistant
        .synthesis_calls
        == []
    )

    assert runtime.tasks == []


def test_orchestrator_handles_multiple_specialists():
    router = (
        FakeRouter()
    )

    router.delegations = [
        SpecialistRequest(
            agent_name=(
                "account-specialist"
            ),
            instructions=(
                "Check account jdoe."
            ),
        ),

        SpecialistRequest(
            agent_name=(
                "access-specialist"
            ),
            instructions=(
                "Check VPN access "
                "for jdoe."
            ),
        ),
    ]

    runtime = (
        FakeRuntime()
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

    assert result.routes == [
        "account-specialist",
        "access-specialist",
    ]

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

    assert (
        len(
            primary_assistant
            .synthesis_calls
        )
        == 1
    )


def test_orchestrator_does_not_synthesize_approval():
    router = (
        FakeRouter()
    )

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

                approval_id=(
                    "approval-123"
                ),

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
        ].approval_id
        == "approval-123"
    )

    assert (
        result.answer
        == (
            "Approval required: "
            "approval-123"
        )
    )

    assert (
        primary_assistant
        .synthesis_calls
        == []
    )


def test_orchestrator_falls_back_if_main_synthesis_fails():
    class BrokenPrimaryAssistant(
        FakePrimaryAssistant
    ):
        async def synthesize(
            self,
            user_request,
            results,
        ):
            raise RuntimeError(
                "main synthesis unavailable"
            )

    primary_assistant = (
        BrokenPrimaryAssistant()
    )

    (
        orchestrator,
        _router,
        _runtime,
        _primary_assistant,
    ) = build_orchestrator(
        primary_assistant=(
            primary_assistant
        )
    )

    result = asyncio.run(
        orchestrator.run(
            "Is jdoe locked?"
        )
    )

    assert (
        result.status
        == "success"
    )

    assert (
        result.answer
        == (
            "Account jdoe is enabled "
            "and is not locked."
        )
    )