from __future__ import annotations

from subagents.core.definitions.types import AgentDefinition
from subagents.core.orchestration.intent_contract import (
    _normalize_argument_map,
    parse_semantic_intent,
)
from subagents.core.orchestration.semantic_guard import SemanticGuard
from subagents.core.tooling.capability_contract import build_model_capability_contract


def create_tool():
    return {
        "description": "Create one thing.",
        "risk": "medium",
        "requires_approval": True,
        "required_arguments": ["target", "summary"],
        "grounded_arguments": ["target", "kind"],
        "derived_arguments": ["summary"],
        "parameters": {
            "target": {"type": "str"},
            "summary": {"type": "str"},
            "kind": {"type": "str"},
        },
    }


def lookup(name: str):
    return create_tool() if name == "create_thing" else None


def agent():
    return AgentDefinition(
        name="thing-specialist",
        description="Creates things.",
        model="test-model",
        tools=["create_thing"],
    )


def payload(target_values):
    return {
        "summary": "Create the requested thing.",
        "effect": "mutation",
        "allowed_tools": ["create_thing"],
        "forbidden_tools": [],
        "allowed_arguments": {
            "target": target_values,
            "kind": [],
        },
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": False,
    }


def test_empty_and_null_optional_bindings_are_absent():
    assert _normalize_argument_map(
        {"target": ["alpha"], "kind": [], "other": None},
        field_name="allowed_arguments",
    ) == {"target": ["alpha"]}


def test_capability_contract_exposes_argument_roles():
    contract = build_model_capability_contract(
        tool_name="create_thing",
        tool=create_tool(),
        description="Create one thing.",
        argument_schema=create_tool()["parameters"],
    )
    assert contract["required_arguments"] == ["target", "summary"]
    assert contract["grounded_arguments"] == ["target", "kind"]
    assert contract["derived_arguments"] == ["summary"]


def test_hallucinated_required_grounded_value_becomes_clarification():
    semantic = parse_semantic_intent(
        payload(["INTERNAL"]),
        agent=agent(),
        tool_lookup=lookup,
        user_request="create an internal meeting ticket for next sunday",
    )
    assert semantic is not None
    assert semantic.clarification_required is True
    assert semantic.allowed_arguments == {}
    assert semantic.missing_required_arguments == ["target"]


def test_explicit_required_grounded_value_remains_executable():
    semantic = parse_semantic_intent(
        payload(["KAN"]),
        agent=agent(),
        tool_lookup=lookup,
        user_request="create the meeting ticket in KAN",
    )
    assert semantic is not None
    assert semantic.clarification_required is False
    assert semantic.missing_required_arguments == []
    assert semantic.allowed_arguments == {"target": ["KAN"]}


def test_guard_allows_derived_content_but_keeps_target_exact():
    semantic = parse_semantic_intent(
        payload(["KAN"]),
        agent=agent(),
        tool_lookup=lookup,
        user_request="create the internal meeting ticket in KAN next sunday",
    )
    guard = SemanticGuard(tool_lookup=lookup)

    allowed = guard.evaluate(
        agent=agent(),
        intent=semantic,
        tool_name="create_thing",
        arguments={
            "target": "KAN",
            "summary": "Internal meeting next Sunday",
        },
    )
    assert allowed.allowed is True

    denied = guard.evaluate(
        agent=agent(),
        intent=semantic,
        tool_name="create_thing",
        arguments={
            "target": "INTERNAL",
            "summary": "Internal meeting next Sunday",
        },
    )
    assert denied.allowed is False
    assert denied.decision_code == "semantic_argument_not_allowed"


def test_guard_requires_required_derived_argument():
    semantic = parse_semantic_intent(
        payload(["KAN"]),
        agent=agent(),
        tool_lookup=lookup,
        user_request="create the internal meeting ticket in KAN next sunday",
    )
    decision = SemanticGuard(tool_lookup=lookup).evaluate(
        agent=agent(),
        intent=semantic,
        tool_name="create_thing",
        arguments={"target": "KAN"},
    )
    assert decision.allowed is False
    assert decision.decision_code == "semantic_required_argument_missing"
