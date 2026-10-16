import asyncio
import json
from pathlib import Path

import pytest

from services.extensions.budget import estimate_messages_tokens, prepare_external_request
from services.extensions.context import execution_profile
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.candidate_selector import Candidate, CandidateSet, CandidateValidationError
from subagents.core.orchestration.router import LLMRouter, RoutingContractError
from subagents.llm.runtime.stages import InferenceStage


class RecordingInference:
    def __init__(self, *responses):
        self.responses = iter(responses or ['{"delegations":[]}'])
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


class FixedSelector:
    def __init__(self, selection):
        self.selection = selection

    def select(self, request, registry):
        return self.selection


@pytest.fixture
def registry(monkeypatch):
    # Match production callable registration without depending on test order.
    from mcp_server import create_mcp_server
    from tools import runtime_schema
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    create_mcp_server()
    result = AgentRegistry()
    result.register_many(load_agent_directory(Path(__file__).resolve().parents[5] / "subagents/agents"))
    return result


@pytest.fixture
def selected():
    token = execution_profile.set({"backend": "http", "model": "test-provider"})
    try:
        yield
    finally:
        execution_profile.reset(token)


def account_selection():
    return CandidateSet((Candidate("account-specialist", ("account_status",)),), confidence=1)


def account_plan(tool="account_status", value="alice", effect="read"):
    return json.dumps({"delegations": [{
        "agent": "account-specialist", "instructions": "Check account status for alice",
        "intent": {"summary": "Check account status", "effect": effect,
                   "allowed_tools": [tool], "forbidden_tools": [],
                   "allowed_arguments": {"user_id": [value]}, "forbidden_arguments": {},
                   "max_tool_calls": 1, "clarification_required": False},
    }]})


def router_for(registry, *responses, selection=None):
    inference = RecordingInference(*responses)
    router = LLMRouter(registry, inference, "hub-main", strict_contract=True, max_new_tokens=512,
                       candidate_selector=FixedSelector(selection) if selection is not None else None)
    return router, inference


@pytest.mark.parametrize("user_request", ["Show Jira projects", "Check account status for alice", "Get Jira ticket OPS-12"])
def test_real_registry_small_request_discloses_compact_packet(registry, selected, user_request):
    router, inference = router_for(registry)
    with pytest.raises(RoutingContractError, match="No supported action"):
        asyncio.run(router.route(user_request))
    call = inference.calls[0]
    assert call["stage"] == InferenceStage.SEMANTIC_ROUTING
    assert estimate_messages_tokens(call["messages"]) + 512 < 5000
    assert len(call["messages"][0]["content"]) < len(router._build_system_prompt()) / 2
    assert call["messages"][-1] == {"role": "user", "content": user_request}


def test_default_local_prompt_and_stage_are_unchanged(registry):
    router, inference = router_for(registry)
    asyncio.run(router.route("Show Jira projects"))
    assert inference.calls[0]["stage"] == InferenceStage.CONTROL_ROUTING
    assert inference.calls[0]["messages"][0]["content"] == router._build_system_prompt()


@pytest.mark.parametrize("selection", [CandidateSet(fallback_required=True), CandidateSet()])
def test_candidate_miss_uses_full_local_control_never_external_catalog(registry, selected, selection):
    router, inference = router_for(registry, selection=selection)
    asyncio.run(router.route("Unfamiliar request"))
    assert inference.calls[0]["stage"] == InferenceStage.CONTROL_ROUTING
    assert inference.calls[0]["messages"][0]["content"] == router._build_system_prompt()


def test_oversized_mandatory_packet_falls_back_without_truncating_user(registry, selected):
    router, inference = router_for(registry, selection=account_selection())
    request = "Check account status for alice " + "x" * 22000
    asyncio.run(router.route(request))
    assert inference.calls[0]["stage"] == InferenceStage.CONTROL_ROUTING
    assert inference.calls[0]["messages"][-1]["content"] == request


def test_invalid_selector_identity_rejected_before_any_model_call(registry, selected):
    selection = CandidateSet((Candidate("made-up-specialist", ("account_status",)),))
    router, inference = router_for(registry, selection=selection)
    with pytest.raises(CandidateValidationError):
        asyncio.run(router.route("Check account status for alice"))
    assert inference.calls == []


def test_compact_route_still_validates_exact_grounding_with_full_registry(registry, selected):
    router, inference = router_for(registry, account_plan(value="mallory"), selection=account_selection())
    routes = asyncio.run(router.route("Check account status for alice"))
    # Existing trusted normalization removes invented values, then requires
    # clarification instead of inventing a replacement target.
    assert routes[0].semantic_intent.clarification_required
    assert routes[0].semantic_intent.allowed_arguments == {}
    assert len(inference.calls) == 1


def test_disclosed_hint_cannot_expand_to_another_valid_capability(registry, selected):
    router, _ = router_for(registry, account_plan(tool="unlock_user", effect="mutation"), selection=account_selection())
    with pytest.raises(RoutingContractError, match="undisclosed"):
        asyncio.run(router.route("Check account status for alice"))


def test_valid_compact_route_uses_full_semantic_contract(registry, selected):
    router, _ = router_for(registry, account_plan(), selection=account_selection())
    routes = asyncio.run(router.route("Check account status for alice"))
    assert len(routes) == 1
    assert routes[0].semantic_intent.allowed_tools == ["account_status"]


def test_one_bounded_repair_preserves_user_and_feedback_under_history_budget(registry, selected):
    router, inference = router_for(registry, "invalid JSON", account_plan(), selection=account_selection())
    request = "Check account status for alice"
    routes = asyncio.run(router.route_with_repair(request, context=[{"role": "user", "content": "old " * 9000}]))
    assert len(routes) == 1 and len(inference.calls) == 2
    repair = inference.calls[1]
    assert repair["stage"] == InferenceStage.REPAIR
    assert repair["repair_of"] == InferenceStage.SEMANTIC_ROUTING
    trimmed = prepare_external_request({}, repair["messages"], 512, default_request_budget=5000)
    assert trimmed.messages[-1] == {"role": "user", "content": request}
    assert any(message["role"] == "system" and "VALIDATION FEEDBACK" in message["content"] for message in trimmed.messages)
    assert all("old old" not in message["content"] for message in trimmed.messages)


def test_rejected_repair_is_not_retried_again(registry, selected):
    router, inference = router_for(registry, "invalid JSON", "invalid JSON", selection=account_selection())
    with pytest.raises(RoutingContractError):
        asyncio.run(router.route_with_repair("Check account status for alice"))
    assert len(inference.calls) == 2


def test_empty_compact_plan_cannot_become_a_successful_conversational_job(registry, selected):
    router, inference = router_for(registry, '{"delegations":[]}', '{"delegations":[]}',
                                   selection=account_selection())
    with pytest.raises(RoutingContractError, match="No supported action"):
        asyncio.run(router.route_with_repair("Check account status for alice"))
    assert len(inference.calls) == 2
