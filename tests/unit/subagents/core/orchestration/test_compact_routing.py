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
    from mcp_server import register_capability_signatures
    from tools import runtime_schema
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    register_capability_signatures()
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


def router_for(registry, *responses, selection=None, prompt_profile="standard", max_new_tokens=512,
               model_profile_resolver=None):
    inference = RecordingInference(*responses)
    router = LLMRouter(registry, inference, "hub-main", strict_contract=True, max_new_tokens=max_new_tokens,
                       router_prompt_profile=prompt_profile, model_profile_resolver=model_profile_resolver,
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
def test_candidate_miss_uses_full_local_control_when_selected_budget_is_small(registry, selected, selection):
    router, inference = router_for(registry, selection=selection)
    asyncio.run(router.route("Unfamiliar request"))
    assert inference.calls[0]["stage"] == InferenceStage.CONTROL_ROUTING
    assert inference.calls[0]["messages"][0]["content"] == router._build_system_prompt()


@pytest.fixture
def large_budget_selected():
    token = execution_profile.set({"backend": "http", "model": "claude-test", "url": "https://api.anthropic.com",
                                   "api_format": "anthropic"})
    try:
        yield
    finally:
        execution_profile.reset(token)


@pytest.mark.parametrize("selection", [CandidateSet(fallback_required=True), CandidateSet()])
def test_candidate_miss_routes_full_catalog_on_selected_connection_when_budget_admits(
        registry, large_budget_selected, selection):
    router, inference = router_for(registry, selection=selection)
    asyncio.run(router.route("Unfamiliar request"))
    call = inference.calls[0]
    assert call["stage"] == InferenceStage.SEMANTIC_ROUTING_BROAD
    system = call["messages"][0]["content"]
    for agent in registry.list_agents():
        for tool in agent.tools:
            assert f'"{tool}"' in system
    assert estimate_messages_tokens(call["messages"]) + 512 <= 24000


def test_broad_route_conversation_is_not_a_routing_failure(registry, large_budget_selected):
    router, inference = router_for(registry, selection=CandidateSet(fallback_required=True))
    assert asyncio.run(router.route("hello there")) == []


def test_broad_route_still_validates_with_full_registry(registry, large_budget_selected):
    router, inference = router_for(registry, account_plan(), selection=CandidateSet(fallback_required=True))
    routes = asyncio.run(router.route("Check account status for alice"))
    assert [route.agent_name for route in routes] == ["account-specialist"]


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


def test_candidate_miss_repair_reroutes_against_the_full_catalog(registry, large_budget_selected):
    narrow = CandidateSet((Candidate("account-specialist", ("account_status",)),), confidence=1)
    router, inference = router_for(registry, '{"delegations":[]}', account_plan(), selection=narrow)
    routes = asyncio.run(router.route_with_repair("Check account status for alice"))
    assert [call["stage"] for call in inference.calls] == [InferenceStage.SEMANTIC_ROUTING, InferenceStage.REPAIR]
    assert inference.calls[1]["repair_of"] == InferenceStage.SEMANTIC_ROUTING_BROAD
    repaired_catalog = inference.calls[1]["messages"][0]["content"]
    assert '"process_exec"' in repaired_catalog and '"ticket_create"' in repaired_catalog
    assert [route.agent_name for route in routes] == ["account-specialist"]


def test_candidate_miss_without_large_budget_keeps_existing_repair(registry, selected):
    narrow = CandidateSet((Candidate("account-specialist", ("account_status",)),), confidence=1)
    router, inference = router_for(registry, '{"delegations":[]}', '{"delegations":[]}', selection=narrow)
    with pytest.raises(RoutingContractError):
        asyncio.run(router.route_with_repair("Check account status for alice"))
    assert inference.calls[1]["repair_of"] == InferenceStage.SEMANTIC_ROUTING


@pytest.mark.parametrize("user_request", ["Show Jira projects", "Check account status for alice", "Get Jira ticket OPS-12"])
def test_explicit_local_compact_profile_uses_bounded_local_candidate_packet(registry, user_request):
    router, inference = router_for(registry, prompt_profile="compact")
    with pytest.raises(RoutingContractError, match="No supported action"):
        asyncio.run(router.route(user_request))
    call = inference.calls[0]
    assert call["stage"] is InferenceStage.CONTROL_ROUTING
    assert call["model_key"] == "hub-main"
    assert estimate_messages_tokens(call["messages"]) + call["max_new_tokens"] <= 5000
    assert call["messages"][-1] == {"role": "user", "content": user_request}
    assert len(call["messages"][0]["content"]) < len(router._build_system_prompt()) / 2


@pytest.mark.parametrize("selection", [CandidateSet(fallback_required=True), CandidateSet()])
def test_unresolved_local_compact_request_discloses_full_trusted_catalog(registry, selection):
    router, inference = router_for(registry, prompt_profile="compact", selection=selection)
    assert asyncio.run(router.route("hello there")) == []
    call = inference.calls[0]
    assert call["stage"] is InferenceStage.CONTROL_ROUTING
    assert estimate_messages_tokens(call["messages"]) + call["max_new_tokens"] <= 24000
    system = call["messages"][0]["content"]
    for agent in registry.list_agents():
        for tool in agent.tools:
            assert f'"{tool}"' in system


def test_local_candidate_miss_widens_once_and_keeps_feedback_and_adapter(registry):
    router, inference = router_for(registry, '{"delegations":[]}', account_plan(),
                                   prompt_profile="compact", selection=account_selection())
    request = "Check account status for alice"
    routes = asyncio.run(router.route_with_repair(request))
    assert [route.agent_name for route in routes] == ["account-specialist"]
    assert len(inference.calls) == 2
    first, repair = inference.calls
    assert first["stage"] is InferenceStage.CONTROL_ROUTING
    assert repair["stage"] is InferenceStage.REPAIR
    assert repair["repair_of"] is InferenceStage.CONTROL_ROUTING
    assert first["model_key"] == repair["model_key"] == "hub-main"
    assert '"process_exec"' not in first["messages"][0]["content"]
    assert '"process_exec"' in repair["messages"][0]["content"]
    assert repair["messages"][-1] == {"role": "user", "content": request}
    assert any("TRUSTED VALIDATION FEEDBACK" in item["content"] for item in repair["messages"])
    assert "Conditional workflow generation is disabled" in repair["messages"][0]["content"]
    assert estimate_messages_tokens(repair["messages"]) + repair["max_new_tokens"] <= 24000


def test_local_compact_format_repair_never_adds_a_third_attempt(registry):
    router, inference = router_for(registry, "invalid JSON", "invalid JSON",
                                   prompt_profile="compact", selection=account_selection())
    with pytest.raises(RoutingContractError, match="invalid JSON"):
        asyncio.run(router.route_with_repair("Check account status for alice"))
    assert len(inference.calls) == 2
    assert inference.calls[1]["repair_of"] is InferenceStage.CONTROL_ROUTING
    assert '"process_exec"' not in inference.calls[1]["messages"][0]["content"]


def test_local_compact_candidate_cannot_select_an_undisclosed_registered_tool(registry):
    router, _ = router_for(registry, account_plan(tool="unlock_user", effect="mutation"),
                           prompt_profile="compact", selection=account_selection())
    with pytest.raises(RoutingContractError, match="undisclosed"):
        asyncio.run(router.route("Check account status for alice"))


def test_local_compact_preserves_grounding_rejection(registry):
    router, _ = router_for(registry, account_plan(value="mallory"),
                           prompt_profile="compact", selection=account_selection())
    routes = asyncio.run(router.route("Check account status for alice"))
    assert routes[0].semantic_intent.clarification_required
    assert routes[0].semantic_intent.allowed_arguments == {}


def test_local_compact_invalid_selector_is_rejected_before_generation(registry):
    router, inference = router_for(registry, prompt_profile="compact",
        selection=CandidateSet((Candidate("made-up-specialist", ("account_status",)),)))
    with pytest.raises(CandidateValidationError):
        asyncio.run(router.route("Check account status for alice"))
    assert inference.calls == []


def test_prior_turns_are_quoted_so_an_earlier_answer_cannot_set_the_reply_format(registry, selected):
    # Replayed as chat turns, an earlier prose answer made a selected chat model
    # continue in prose, and the job failed as invalid routing JSON.
    router, inference = router_for(registry, account_plan(), selection=account_selection())
    history = [{"role": "user", "content": "can you write code?"},
               {"role": "assistant", "content": "Yes, I can help with programming."}]
    request = "Check account status for alice"

    assert asyncio.run(router.route(request, context=history))

    messages = inference.calls[0]["messages"]
    assert messages[-1] == {"role": "user", "content": request}
    assert [message["role"] for message in messages] == ["system", "user", "user", "user"]
    assert [json.loads(message["content"].split("\n", 1)[1]) for message in messages[1:-1]] == history


def test_local_compact_trims_history_without_changing_mandatory_request(registry):
    router, inference = router_for(registry, account_plan(), prompt_profile="compact", selection=account_selection())
    request = "Check account status for alice"
    context = [{"role": "user", "content": "old " * 9000}]
    original = list(context)
    assert asyncio.run(router.route(request, context=context))
    call = inference.calls[0]
    assert call["messages"][-1] == {"role": "user", "content": request}
    assert all("old old" not in item["content"] for item in call["messages"])
    assert context == original
    assert estimate_messages_tokens(call["messages"]) + call["max_new_tokens"] <= 5000


def test_local_compact_oversized_mandatory_input_fails_before_any_generation(registry):
    from services.extensions.budget import ExternalInferenceBudgetError
    router, inference = router_for(registry, prompt_profile="compact", selection=account_selection())
    with pytest.raises(ExternalInferenceBudgetError) as failure:
        asyncio.run(router.route_with_repair("Check account status for alice " + "x" * 100000))
    assert inference.calls == []
    assert failure.value.request_token_budget == 24000
    assert failure.value.mandatory_tokens > failure.value.input_budget


def test_local_compact_budgets_ignore_external_provider_environment(registry, monkeypatch):
    monkeypatch.setenv("EXTERNAL_INFERENCE_REQUEST_TOKEN_BUDGET", "512")
    monkeypatch.setenv("EXTERNAL_INFERENCE_MAX_OUTPUT_TOKENS", "32")
    router, inference = router_for(registry, account_plan(), prompt_profile="compact", selection=account_selection(),
                                   max_new_tokens=900)
    assert asyncio.run(router.route("Check account status for alice"))
    assert inference.calls[0]["max_new_tokens"] == 768
    assert estimate_messages_tokens(inference.calls[0]["messages"]) + 768 <= 5000


def test_local_compact_capture_has_actual_prompt_without_standard_training_identity(registry):
    from learning.evidence.hub_routing import consume_hub_routing_trace
    def forbidden_resolver(_):
        raise AssertionError("Compact local routing is not admitted to standard Hub training")
    router, inference = router_for(registry, account_plan(), prompt_profile="compact", selection=account_selection(),
                                   model_profile_resolver=forbidden_resolver)
    async def scenario():
        await router.route("Check account status for alice")
        return consume_hub_routing_trace()
    trace = asyncio.run(scenario())
    assert len(trace) == 1
    assert trace[0].messages == inference.calls[0]["messages"]
    assert trace[0].model_identity.complete is False
    assert trace[0].model_identity.model_weights_sha256 is None
    assert trace[0].training_eligible is False


def test_local_compact_setting_preserves_explicit_remote_adapter_selection(registry, selected):
    router, inference = router_for(registry, account_plan(), prompt_profile="compact", selection=account_selection())
    assert asyncio.run(router.route("Check account status for alice"))
    assert inference.calls[0]["stage"] is InferenceStage.SEMANTIC_ROUTING


@pytest.mark.parametrize("value", [None, "", "unknown", 1])
def test_invalid_local_router_profile_fails_at_construction(registry, value):
    with pytest.raises(ValueError, match="prompt profile"):
        router_for(registry, prompt_profile=value)
