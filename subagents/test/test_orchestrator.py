import asyncio

from subagents.core.orchestrator import Orchestrator
from subagents.core.types import AgentResult


class FakeRouter:
    def __init__(self):
        self.routes = [
            "account-specialist"
        ]

    def route(
        self,
        user_request: str,
    ) -> list[str]:
        return self.routes


class FakeRuntime:
    def __init__(self):
        self.tasks = []

    async def run(
        self,
        task,
    ) -> AgentResult:
        self.tasks.append(task)

        return AgentResult(
            task_id=task.task_id,
            agent_name=task.agent_name,
            status="success",
            proposed_tool="account_status",
            proposed_arguments={
                "user_id": "jdoe"
            },
            answer=(
                "{'ok': True, "
                "'user_id': 'jdoe', "
                "'locked': False}"
            ),
        )


def test_orchestrator_routes_to_worker():
    router = FakeRouter()
    runtime = FakeRuntime()

    orchestrator = Orchestrator(
        router=router,
        runtime=runtime,
    )

    result = asyncio.run(
        orchestrator.run(
            "Is jdoe locked?"
        )
    )

    assert result.status == "success"

    assert result.routes == [
        "account-specialist"
    ]

    assert len(result.results) == 1

    assert (
        result.results[0].agent_name
        == "account-specialist"
    )

    assert (
        result.results[0].proposed_tool
        == "account_status"
    )

    assert len(runtime.tasks) == 1

    assert (
        runtime.tasks[0].user_request
        == "Is jdoe locked?"
    )


def test_orchestrator_handles_no_route():
    router = FakeRouter()
    router.routes = []

    runtime = FakeRuntime()

    orchestrator = Orchestrator(
        router=router,
        runtime=runtime,
    )

    result = asyncio.run(
        orchestrator.run(
            "Tell me a joke."
        )
    )

    assert result.status == "no_route"
    assert result.routes == []
    assert result.results == []


def test_orchestrator_handles_multiple_workers():
    router = FakeRouter()

    router.routes = [
        "account-specialist",
        "access-specialist",
    ]

    runtime = FakeRuntime()

    orchestrator = Orchestrator(
        router=router,
        runtime=runtime,
    )

    result = asyncio.run(
        orchestrator.run(
            "Check whether jdoe is locked "
            "and whether he has VPN access."
        )
    )

    assert len(result.routes) == 2
    assert len(result.results) == 2

    assert len(runtime.tasks) == 2


def test_orchestrator_propagates_approval():
    router = FakeRouter()

    class ApprovalRuntime:
        async def run(self, task):
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="approval_required",
                proposed_tool="unlock_user",
                proposed_arguments={
                    "user_id": "jdoe"
                },
                answer=(
                    "Approval required: "
                    "approval-123"
                ),
            )

    orchestrator = Orchestrator(
        router=router,
        runtime=ApprovalRuntime(),
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
        result.results[0].proposed_tool
        == "unlock_user"
    )