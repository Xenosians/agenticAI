"""A read the validated semantic contract fixes completely needs no specialist model step."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.definitions.types import AgentTask, SemanticIntent
from subagents.core.orchestration.runtime import AgentRuntime
from subagents.core.orchestration.semantic_binding import determined_read_call
from subagents.core.orchestration.semantic_guard import SemanticGuard

AGENTS = Path(__file__).resolve().parents[5] / "subagents/agents"


@pytest.fixture(scope="module")
def registry():
    agents = AgentRegistry()
    agents.register_many(load_agent_directory(AGENTS))
    return agents


def intent(tools, arguments=None, *, effect="read", **overrides):
    values = {"summary": "test", "effect": effect, "allowed_tools": list(tools),
              "allowed_arguments": dict(arguments or {}), "forbidden_tools": [], "forbidden_arguments": {},
              "max_tool_calls": 1, "clarification_required": False}
    values.update(overrides)
    return SemanticIntent(**values)


@pytest.mark.parametrize("agent,contract,expected", [
    ("networking-specialist", intent(["network_tailnet_devices"]),
     {"name": "network_tailnet_devices", "arguments": {}}),
    ("account-specialist", intent(["account_status"], {"user_id": ["jdoe"]}),
     {"name": "account_status", "arguments": {"user_id": "jdoe"}}),
    ("networking-specialist", intent(["network_dns_integrity_check"], {"names": ["vimeo.com", "example.com"]}),
     {"name": "network_dns_integrity_check", "arguments": {"names": ["vimeo.com", "example.com"]}}),
])
def test_a_read_the_contract_fixes_is_determined(registry, agent, contract, expected):
    assert determined_read_call(intent=contract, agent=registry.get(agent)) == expected


@pytest.mark.parametrize("agent,contract", [
    # A change always goes through the model step and approval.
    ("account-specialist", intent(["unlock_user"], {"user_id": ["jdoe"]}, effect="mutation")),
    # A choice between tools or between targets is the model's to make.
    ("networking-specialist", intent(["network_tailnet_devices", "network_tailnet_dns"])),
    ("account-specialist", intent(["account_status"], {"user_id": ["jdoe", "asmith"]})),
    # Unbound or free-text arguments are proposal content, never invented here.
    ("networking-specialist", intent(["network_dns_integrity_check"])),
    ("knowledge-specialist", intent(["knowledge_search"], {"query": ["vpn"]})),
    # A contract the guard would refuse keeps its normal path and refusal.
    ("account-specialist", intent(["account_status"], {"user_id": ["jdoe"]}, clarification_required=True)),
    ("account-specialist", intent(["account_status"], {"user_id": ["jdoe"]}, forbidden_arguments={"user_id": ["jdoe"]})),
    ("account-specialist", intent(["account_status"], {"user_id": ["jdoe"]}, effect="mutation")),
    ("networking-specialist", intent(["account_status"], {"user_id": ["jdoe"]})),
    ("networking-specialist", intent(["network_tailnet_devices"], forbidden_tools=["network_tailnet_devices"])),
])
def test_anything_short_of_a_fixed_read_keeps_the_model_step(registry, agent, contract):
    assert determined_read_call(intent=contract, agent=registry.get(agent)) is None


def test_no_contract_means_no_determined_call(registry):
    assert determined_read_call(intent=None, agent=registry.get("networking-specialist")) is None
