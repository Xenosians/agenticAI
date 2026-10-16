from pathlib import Path

import pytest

from services.knowledge import MockKnowledgeService
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.orchestration.intent_contract import parse_semantic_intent
from subagents.core.orchestration.semantic_guard import SemanticGuard
from tools import runtime_schema
from tools.capability_contract import build_capability_contract
from tools.knowledge.mcp import register_knowledge_tools
from tools.registry import get_tool
from tools.runtime_schema import ContractAwareMCPServer


@pytest.fixture
def knowledge_agent(monkeypatch):
    # Requiredness comes from the real callable signature, independently of
    # whether another test imported mcp_server and populated global schemas.
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    register_knowledge_tools(
        ContractAwareMCPServer("Knowledge contract test"),
        MockKnowledgeService(),
    )
    agents_dir = Path(__file__).resolve().parents[5] / "subagents" / "agents"
    return next(
        agent for agent in load_agent_directory(agents_dir)
        if agent.name == "knowledge-specialist"
    )


def search_intent(allowed_arguments=None):
    return {
        "summary": "Search knowledge for VPN troubleshooting.",
        "effect": "read",
        "allowed_tools": ["knowledge_search"],
        "forbidden_tools": [],
        "allowed_arguments": allowed_arguments or {},
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": False,
    }


def test_non_grounded_capability_accepts_empty_semantic_argument_maps(knowledge_agent):
    contract = build_capability_contract("knowledge_search", get_tool("knowledge_search"))
    assert contract.required_arguments == ("query",)
    assert contract.derived_arguments == ("query",)
    assert contract.requiredness_source == "executable_signature"

    intent = parse_semantic_intent(search_intent(), agent=knowledge_agent)
    assert intent is not None
    assert intent.allowed_arguments == {}
    assert intent.forbidden_arguments == {}
    assert not intent.clarification_required


def test_non_grounded_capability_rejects_untrusted_argument_binding(knowledge_agent):
    with pytest.raises(ValueError, match="untrusted or irrelevant grounded argument"):
        parse_semantic_intent(
            search_intent({"unrecognized_target": ["VPN troubleshooting"]}),
            agent=knowledge_agent,
        )


def test_derived_search_phrase_is_not_promoted_to_authority(knowledge_agent):
    intent = parse_semantic_intent(
        search_intent({"query": ["VPN troubleshooting"]}),
        agent=knowledge_agent,
    )
    assert intent.allowed_arguments == {}
    assert not intent.clarification_required


@pytest.mark.parametrize("arguments,decision_code", [
    ({}, "semantic_required_argument_missing"),
    ({"query": ""}, "semantic_derived_argument_invalid"),
    ({"query": "  "}, "semantic_derived_argument_invalid"),
])
def test_search_still_requires_nonempty_query(knowledge_agent, arguments, decision_code):
    intent = parse_semantic_intent(search_intent(), agent=knowledge_agent)
    decision = SemanticGuard().evaluate(
        agent=knowledge_agent,
        intent=intent,
        tool_name="knowledge_search",
        arguments=arguments,
    )
    assert not decision.allowed
    assert decision.decision_code == decision_code


def test_specialist_can_compose_read_only_search_query(knowledge_agent):
    intent = parse_semantic_intent(search_intent(), agent=knowledge_agent)
    decision = SemanticGuard().evaluate(
        agent=knowledge_agent,
        intent=intent,
        tool_name="knowledge_search",
        arguments={"query": "VPN connection troubleshooting", "limit": 5},
    )
    assert decision.allowed
