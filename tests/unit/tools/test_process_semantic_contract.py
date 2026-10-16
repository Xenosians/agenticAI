from pathlib import Path

import pytest

from subagents.core.definitions.loader import load_agent_directory
from subagents.core.orchestration.intent_contract import parse_semantic_intent
from subagents.core.orchestration.semantic_guard import SemanticGuard
from tools import runtime_schema
from tools.capability_contract import build_capability_contract
from tools.registry import get_tool


@pytest.fixture
def developer_agent(monkeypatch):
    # Register real executable signatures, without invoking any tool/provider.
    from mcp_server import create_mcp_server

    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    create_mcp_server()
    agents_dir = Path(__file__).resolve().parents[3] / "subagents" / "agents"
    return next(
        agent for agent in load_agent_directory(agents_dir)
        if agent.name == "developer-specialist"
    )


def process_intent(executable=None):
    return {
        "summary": "Run the explicitly requested native process.",
        "effect": "mutation",
        "allowed_tools": ["process_exec"],
        "forbidden_tools": [],
        "allowed_arguments": {"executable": [executable]} if executable else {},
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": False,
    }


def test_process_executable_requires_exact_grounding(developer_agent):
    contract = build_capability_contract("process_exec", get_tool("process_exec"))
    assert contract.required_arguments == ("executable",)
    assert "executable" in contract.grounded_arguments
    assert "executable" not in contract.derived_arguments
    assert contract.requires_approval

    intent = parse_semantic_intent(
        process_intent("pwd"), agent=developer_agent, user_request="Run native process pwd"
    )
    assert not intent.clarification_required
    assert intent.allowed_arguments == {"executable": ["pwd"]}

    decision = SemanticGuard().evaluate(
        agent=developer_agent, intent=intent, tool_name="process_exec",
        arguments={"executable": "pwd"},
    )
    assert decision.allowed


@pytest.mark.parametrize("executable,user_request", [
    (None, "Execute a native process"),
    ("pwd", "Execute a native process"),
    ("ls", "Run native process pwd"),
])
def test_missing_or_invented_executable_requires_clarification(
    developer_agent, executable, user_request
):
    intent = parse_semantic_intent(
        process_intent(executable), agent=developer_agent, user_request=user_request
    )
    assert intent.clarification_required
    assert intent.allowed_arguments == {}
    assert intent.missing_required_arguments == ["executable"]


def test_specialist_cannot_replace_explicit_executable(developer_agent):
    intent = parse_semantic_intent(
        process_intent("pwd"), agent=developer_agent, user_request="Run native process pwd"
    )
    decision = SemanticGuard().evaluate(
        agent=developer_agent, intent=intent, tool_name="process_exec",
        arguments={"executable": "ls"},
    )
    assert not decision.allowed
    assert decision.decision_code == "semantic_argument_not_allowed"
