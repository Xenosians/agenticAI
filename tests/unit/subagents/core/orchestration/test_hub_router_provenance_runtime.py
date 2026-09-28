import asyncio

from learning.evidence.hub_routing import (
    consume_hub_routing_trace,
    reset_hub_routing_trace,
)
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.router import (
    LLMRouter,
)


class SequencedInference:
    def __init__(self, responses):
        self.responses = list(responses)

    async def generate(
        self,
        *,
        model_key,
        messages,
        max_new_tokens,
        priority,
    ):
        if not self.responses:
            raise AssertionError("Unexpected extra router generation")

        return self.responses.pop(0)


def _router(responses):
    return LLMRouter(
        registry=AgentRegistry(),
        inference=SequencedInference(responses),
        model_key="hub-main",
        strict_contract=True,
        max_new_tokens=512,
    )


def test_router_captures_accepted_empty_route():
    async def scenario():
        reset_hub_routing_trace()

        router = _router(
            [
                '{"delegations":[]}',
            ]
        )

        result = await router.route(
            "Hello"
        )

        # ContextVar values are task-local by design.
        #
        # Production execute_job() awaits hub.run() and then calls the
        # trajectory recorder in this SAME asyncio task. Therefore the test
        # must consume the routing trace before asyncio.run() exits instead
        # of reading the parent synchronous Context after the child task is
        # destroyed.
        trace = consume_hub_routing_trace()

        return result, trace

    result, trace = asyncio.run(
        scenario()
    )

    assert result == []

    assert len(trace) == 1
    assert trace[0].mode == "normal"
    assert trace[0].validation_status == "accepted"
    assert trace[0].validated_delegation_count == 0


def test_router_preserves_rejected_normal_and_accepted_repair():
    async def scenario():
        reset_hub_routing_trace()

        router = _router(
            [
                '{"not_delegations":[]}',
                '{"delegations":[]}',
            ]
        )

        result = await router.route_with_repair(
            "Hello"
        )

        trace = consume_hub_routing_trace()

        return result, trace

    result, trace = asyncio.run(
        scenario()
    )

    assert result == []

    assert len(trace) == 2
    assert trace[0].mode == "normal"
    assert trace[0].validation_status == "rejected"
    assert trace[1].mode == "repair"
    assert trace[1].validation_status == "accepted"
