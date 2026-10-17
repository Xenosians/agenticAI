"""Success-ordering metadata survives disclosure and repair without normalization."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from agent.onboarding import account_ticket_plan
from services.extensions.context import execution_profile
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentDefinition, SemanticIntent
from subagents.core.orchestration.candidate_selector import (
    Candidate, CandidateSet, build_candidate_agent_specs,
)
from subagents.core.orchestration.intent_contract import build_router_semantic_agent_spec
from subagents.core.orchestration.router import LLMRouter, RoutingContractError


@pytest.fixture
def registry():
    registry = AgentRegistry()
    for agent, tool in (("account-specialist", "account_create"), ("jira-specialist", "ticket_create")):
        registry.register(AgentDefinition(name=agent, description=agent, model="test", tools=[tool]))
    return registry


def test_compact_metadata_retains_the_same_declared_successors_as_full_catalog(registry):
    selection = CandidateSet((Candidate("account-specialist", ("account_create",)),), confidence=1)
    compact = build_candidate_agent_specs(selection, registry)[0]["capabilities"][0]
    full = build_router_semantic_agent_spec(registry.get("account-specialist"))["capabilities"][0]
    assert compact["intent_metadata"]["success_continuations"] == ["ticket_create"]
    assert compact["intent_metadata"]["success_continuations"] == full["intent_metadata"]["success_continuations"]


def test_compact_metadata_rejects_unregistered_successors(registry, monkeypatch):
    import subagents.core.orchestration.candidate_selector as module
    original = module.get_tool

    def lookup(name):
        tool = original(name)
        return {**tool, "success_continuations": ["not_registered"]} if name == "account_create" else tool

    monkeypatch.setattr(module, "get_tool", lookup)
    with pytest.raises(ValueError, match="success_continuations"):
        build_candidate_agent_specs(
            CandidateSet((Candidate("account-specialist", ("account_create",)),)), registry,
        )


def plan():
    rows = []
    for agent, tool, values in (
        ("account-specialist", "account_create", {"given_name": ["Mvp"], "family_name": ["Test"]}),
        ("jira-specialist", "ticket_create", {"project_key": ["KAN"]}),
    ):
        intent = asdict(SemanticIntent(summary="Perform the requested operation", effect="mutation",
                                      allowed_tools=[tool], allowed_arguments=values))
        intent.pop("missing_required_arguments")
        rows.append({"agent": agent, "instructions": "Perform the requested operation", "intent": intent})
    rows[1]["after_success_of"] = rows[0]["agent"]
    return {"delegations": rows}


@pytest.mark.parametrize("compact", [False, True])
def test_misnested_dependency_is_rejected_and_one_fresh_repair_uses_same_strict_contract(registry, compact):
    good = plan()
    bad = deepcopy(good)
    follow_up = bad["delegations"][1]
    follow_up["intent"]["after_success_of"] = follow_up.pop("after_success_of")

    class Inference:
        calls = []

        async def generate(self, **kwargs):
            self.calls.append(kwargs)
            # The model must emit a new valid plan. Production does not move fields.
            return json.dumps(bad if len(self.calls) == 1 else good)

    class Selector:
        def select(self, *args):
            return CandidateSet((Candidate("account-specialist", ("account_create",)),
                                 Candidate("jira-specialist", ("ticket_create",))), confidence=1)

    inference = Inference()
    router = LLMRouter(registry, inference, "test", strict_contract=True, candidate_selector=Selector())
    request = "Create account given name Mvp family name Test. After success create Jira ticket in KAN for that account."
    token = execution_profile.set({"backend": "http", "model": "test"} if compact else None)
    try:
        result = asyncio.run(router.route_with_repair(request))
    finally:
        execution_profile.reset(token)
    assert account_ticket_plan(result) is not None
    assert len(inference.calls) == 2
    for call in inference.calls:
        assert "after_success_of is a sibling of agent, instructions and intent" in call["messages"][0]["content"]
        assert "It is NEVER\ninside intent" in call["messages"][0]["content"]
    assert any("unsupported fields: after_success_of" in message["content"]
               for message in inference.calls[1]["messages"] if message["role"] == "system")


def test_repeated_invalid_dependency_remains_an_error_with_exactly_one_repair(registry):
    bad = plan()
    follow_up = bad["delegations"][1]
    follow_up["intent"]["after_success_of"] = follow_up.pop("after_success_of")

    class Inference:
        calls = 0

        async def generate(self, **kwargs):
            self.calls += 1
            return json.dumps(bad)

    inference = Inference()
    router = LLMRouter(registry, inference, "test", strict_contract=True)
    with pytest.raises(RoutingContractError, match="unsupported fields: after_success_of"):
        asyncio.run(router.route_with_repair("Create account Mvp Test then Jira ticket KAN after success"))
    assert inference.calls == 2
