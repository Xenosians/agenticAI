"""Independent read steps of one request run side by side (bounded); every
other step keeps the request's order.

A read step has no condition, no dependency and a validated read intent, so
running it early cannot change what it reads or what runs after it. Mutations,
conditional steps and dependent steps are barriers: nothing overlaps them.
"""
import asyncio

import pytest

from subagents.core.definitions.types import (
    AgentResult,
    ResultCondition,
    SemanticIntent,
    SpecialistRequest,
)
from subagents.core.orchestration.orchestrator import Orchestrator


def read_step(agent: str, tool: str = "account_status") -> SpecialistRequest:
    return SpecialistRequest(
        agent_name=agent,
        instructions=f"Read with {agent}.",
        semantic_intent=SemanticIntent(summary=f"Read with {agent}.", effect="read", allowed_tools=[tool]),
    )


def mutation_step(agent: str) -> SpecialistRequest:
    return SpecialistRequest(
        agent_name=agent,
        instructions=f"Change with {agent}.",
        semantic_intent=SemanticIntent(summary=f"Change with {agent}.", effect="mutation",
                                       allowed_tools=["account_unlock"]),
    )


class PlannedRouter:
    def __init__(self, delegations):
        self.delegations = delegations

    async def route(self, user_request):
        return list(self.delegations)


class TrackingRuntime:
    """Records how many specialist steps were running at the same time."""

    def __init__(self, *, fail: str | None = None, results: dict | None = None):
        self.active = 0
        self.peak = 0
        self.order: list[str] = []
        self.finished: list[str] = []
        self.cancelled: list[str] = []
        self.fail = fail
        self.results = results or {}

    async def run(self, task) -> AgentResult:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.order.append(task.agent_name)
        try:
            await asyncio.sleep(0.05)
            if task.agent_name == self.fail:
                raise RuntimeError("specialist failed")
            self.finished.append(task.agent_name)
            return AgentResult(
                task_id=task.task_id,
                agent_name=task.agent_name,
                status="success",
                answer=f"{task.agent_name} answered",
                tool_result=self.results.get(task.agent_name),
            )
        except asyncio.CancelledError:
            self.cancelled.append(task.agent_name)
            raise
        finally:
            self.active -= 1


class Assistant:
    def __init__(self):
        self.synthesized = []

    async def respond(self, user_request):
        return "direct"

    async def synthesize(self, user_request, results):
        self.synthesized.append([result.agent_name for result in results])
        return "combined"


def run(delegations, runtime, *, limit, outbound=lambda agent_name: True):
    assistant = Assistant()
    orchestrator = Orchestrator(
        router=PlannedRouter(delegations),
        runtime=runtime,
        primary_assistant=assistant,
        max_parallel_specialists=limit,
        runs_outbound=outbound,
    )
    return asyncio.run(orchestrator.run("Check everything.")), assistant


def test_independent_reads_run_side_by_side_within_the_limit_and_keep_their_order():
    runtime = TrackingRuntime()
    steps = [read_step("account-specialist"), read_step("asset-specialist"), read_step("ticket-specialist")]

    result, assistant = run(steps, runtime, limit=2)

    assert runtime.peak == 2
    assert result.status == "success"
    assert result.routes == ["account-specialist", "asset-specialist", "ticket-specialist"]
    assert [item.agent_name for item in result.results] == result.routes
    assert assistant.synthesized == [result.routes]


def test_a_limit_of_one_keeps_the_serial_behaviour():
    runtime = TrackingRuntime()

    result, _ = run([read_step("account-specialist"), read_step("asset-specialist")], runtime, limit=1)

    assert runtime.peak == 1
    assert runtime.order == ["account-specialist", "asset-specialist"]
    assert [item.agent_name for item in result.results] == ["account-specialist", "asset-specialist"]


def test_a_mutation_step_is_a_barrier_nothing_overlaps_it():
    runtime = TrackingRuntime()
    steps = [read_step("account-specialist"), mutation_step("access-specialist"), read_step("asset-specialist")]

    result, _ = run(steps, runtime, limit=2)

    assert runtime.peak == 1
    assert runtime.order == ["account-specialist", "access-specialist", "asset-specialist"]
    assert result.routes == runtime.order


def test_a_conditional_step_waits_for_the_read_it_depends_on():
    runtime = TrackingRuntime(results={"access-specialist": {"ok": True, "has_access": False}})
    check = read_step("access-specialist", tool="check_access")
    grant = SpecialistRequest(
        agent_name="access-specialist",
        instructions="Grant access if missing.",
        semantic_intent=SemanticIntent(summary="Grant access.", effect="mutation", allowed_tools=["grant_access"]),
        condition=ResultCondition(source_agent="access-specialist", result_field="has_access", equals=False),
    )

    result, _ = run([check, grant], runtime, limit=2)

    assert runtime.peak == 1
    assert runtime.order == ["access-specialist", "access-specialist"]
    assert len(result.results) == 2


def test_a_failing_read_cancels_its_unfinished_neighbour_and_raises():
    runtime = TrackingRuntime(fail="account-specialist")
    steps = [read_step("account-specialist"), read_step("asset-specialist"), read_step("ticket-specialist")]

    async def scenario():
        orchestrator = Orchestrator(
            router=PlannedRouter(steps),
            runtime=runtime,
            primary_assistant=Assistant(),
            max_parallel_specialists=2,
            runs_outbound=lambda agent_name: True,
        )
        with pytest.raises(RuntimeError, match="specialist failed"):
            await orchestrator.run("Check everything.")
        # Nothing the request started is left running after it ends.
        await asyncio.sleep(0)
        current = asyncio.current_task()
        return [task for task in asyncio.all_tasks() if task is not current and not task.done()]

    assert asyncio.run(scenario()) == []
    assert "ticket-specialist" not in runtime.finished


@pytest.mark.parametrize("limit", [0, 3, -1, True])
def test_the_parallel_limit_is_one_or_two(limit):
    with pytest.raises(ValueError):
        Orchestrator(
            router=PlannedRouter([]),
            runtime=TrackingRuntime(),
            primary_assistant=Assistant(),
            max_parallel_specialists=limit,
        )


def test_steps_on_the_local_model_take_turns_so_the_gpu_never_swaps_models():
    """Two local specialists at once would load two models into one small GPU."""
    local = {"account-specialist", "asset-specialist"}
    outbound = lambda agent_name: agent_name not in local  # noqa: E731
    steps = [read_step("account-specialist"), read_step("ticket-specialist"), read_step("asset-specialist")]

    runtime = TrackingRuntime()
    result, _ = run(steps, runtime, limit=2, outbound=outbound)

    # The outbound step overlaps a local one; the two local steps never run together.
    assert runtime.peak == 2
    assert result.routes == ["account-specialist", "ticket-specialist", "asset-specialist"]

    runtime = TrackingRuntime()
    run([read_step("account-specialist"), read_step("asset-specialist")], runtime, limit=2, outbound=outbound)
    assert runtime.peak == 1


def test_without_knowing_where_models_run_every_step_is_treated_as_local():
    runtime = TrackingRuntime()
    orchestrator = Orchestrator(
        router=PlannedRouter([read_step("account-specialist"), read_step("asset-specialist")]),
        runtime=runtime,
        primary_assistant=Assistant(),
        max_parallel_specialists=2,
    )

    asyncio.run(orchestrator.run("Check both."))

    assert runtime.peak == 1
