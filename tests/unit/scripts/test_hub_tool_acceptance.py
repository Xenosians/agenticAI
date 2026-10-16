import asyncio
import json
from types import SimpleNamespace

import pytest

from scripts.hub_tool_acceptance import (
    ObservedInference, ProposalBoundary, observe_cases, score_case,
)
from scripts.tool_acceptance_matrix import load_inventory, load_proposal_cases, validate_case
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.orchestrator import Orchestrator
from subagents.core.orchestration.router import LLMRouter
from subagents.core.orchestration.runtime import AgentRuntime


@pytest.fixture(scope="module")
def corpus():
    contracts, agents = load_inventory()
    case = next(item for item in load_proposal_cases()
                if item.expected_tool == "asset_get" and not item.training_overlap
                and not validate_case(item, contracts, agents))
    return contracts, agents, case


def real_graph(corpus, *, worker_arguments=None, router_tool=None):
    contracts, agents, case = corpus
    registry = AgentRegistry()
    registry.register(agents[case.agent_name])
    routed_tool = router_tool or case.expected_tool
    routing = json.dumps({"delegations": [{"agent": case.agent_name,
        "instructions": case.user, "intent": {
            "summary": case.user, "effect": "read", "allowed_tools": [routed_tool],
            "allowed_arguments": case.allowed_arguments, "forbidden_tools": [],
            "forbidden_arguments": {}, "max_tool_calls": 1,
            "clarification_required": False}}]})
    proposal = json.dumps([{"name": case.expected_tool,
                           "arguments": case.expected_arguments if worker_arguments is None else worker_arguments}])

    class Serving:
        async def generate(self, **kwargs):
            return proposal if kwargs["stage"].value == "SPECIALIST_REASONING" else routing

    class NoConversation:
        async def respond(self, *args, **kwargs):
            raise AssertionError("A conversational response cannot prove tool selection")

        async def synthesize(self, *args, **kwargs):
            raise AssertionError("No tool result exists to synthesize")

    inference = ObservedInference(Serving())
    boundary = ProposalBoundary()
    hub = Orchestrator(
        router=LLMRouter(registry=registry, inference=inference, model_key="test",
                         strict_contract=True),
        runtime=AgentRuntime(agent_registry=registry, inference=inference, tool_gateway=boundary),
        primary_assistant=NoConversation(), require_semantic_intent=True)
    return hub, inference, boundary


def observe(corpus, **kwargs):
    contracts, agents, case = corpus
    hub, inference, boundary = real_graph(corpus, **kwargs)
    rows = []
    complete = asyncio.run(observe_cases(
        hub=hub, inference=inference, boundary=boundary, cases=[case],
        contracts=contracts, agents=agents, metrics=[], timeout=3, record=rows.append))
    return complete, rows, boundary


def test_actual_router_and_worker_reach_disabled_execution_boundary(corpus):
    complete, rows, boundary = observe(corpus)
    assert complete and rows[0]["passed"]
    assert rows[0]["decision_code"] == "hub_and_worker_proposal_passed"
    assert rows[0]["tool_executed"] is False
    assert rows[0]["approval_created"] is False
    assert len(boundary.calls) == 1


def test_hub_wrong_route_cannot_be_rescued_by_correct_worker_fixture(corpus):
    _, rows, boundary = observe(corpus, router_tool="invented-tool")
    assert not rows[0]["passed"]
    assert boundary.calls == []


def test_worker_argument_drift_cannot_be_hidden_by_runtime_binding(corpus):
    _, rows, boundary = observe(corpus, worker_arguments={"asset_id": "INVENTED-999"})
    assert not rows[0]["passed"]
    # Production binding can safely restore the one authorized target, but this
    # model-understanding score must still reject the original hallucination.
    assert len(boundary.calls) == 1
    assert boundary.calls[0]["arguments"] == {"asset_id": "AST-204"}


def test_extra_specialist_route_cannot_count_as_success(corpus):
    contracts, agents, case = corpus
    worker = SimpleNamespace(agent_name=case.agent_name)
    result = score_case(case, SimpleNamespace(results=[worker, worker]), [], [],
                        contracts=contracts, agents=agents)
    assert result == {"passed": False, "decision_code": "wrong_specialist_or_route_count"}


def test_timeout_stops_batch_without_retry_or_gold_injection(corpus):
    contracts, agents, case = corpus
    submitted = []

    class Hub:
        async def run(self, request):
            submitted.append(request)
            await asyncio.Event().wait()

    rows = []
    complete = asyncio.run(observe_cases(
        hub=Hub(), inference=SimpleNamespace(responses=[]), boundary=ProposalBoundary(),
        cases=[case, case], contracts=contracts, agents=agents, metrics=[],
        timeout=0.01, record=rows.append))
    assert complete is False
    assert submitted == [case.user]
    assert len(rows) == 1 and rows[0]["error_type"] == "TimeoutError"
    assert not rows[0]["passed"]
