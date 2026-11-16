"""A tool description is routing evidence; two tools of one specialist must not claim the same thing.

Live, the database inventory's "Use this for database servers, their backups
and capacity" pulled backup questions away from proxmox_backups, although the
deterministic selector ranked proxmox_backups first. Within one specialist, a
"Use this ... for <scope>" directive may not claim another tool's resource
noun unless it names that tool (which states where its scope ends). A phrase
qualified by the tool's own resource ("tailnet health", "database hosts") is
its own scope, not a claim.
"""
import re
from pathlib import Path

import pytest

from mcp_server import register_capability_signatures
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.definitions.registry import AgentRegistry
from subagents.core.orchestration.candidate_selector import _terms
from tools import runtime_schema
from tools.registry import TOOLS

ROOT = Path(__file__).resolve().parents[3]
# "Do NOT use this for ..." states where a scope ends; it claims nothing.
DIRECTIVE = re.compile(r"(?<!not )(?<!never )\buse this\b[^.;:]*?\bfor\b([^.;:(]*)", re.IGNORECASE)
PHRASE_BREAK = re.compile(r",|\band\b|\bor\b", re.IGNORECASE)


def _own_terms(resource_type: str) -> set[str]:
    # The first segment of a dotted resource type is its domain ("proxmox",
    # "network"), shared by every tool of the specialist; it qualifies nothing.
    segments = resource_type.split(".")
    return _terms(" ".join(segments[1:] if len(segments) > 1 else segments))


def scope_collisions(agents, tools) -> list[tuple[str, str, str, str]]:
    found = []
    for agent in agents:
        for name in agent.tools:
            description = tools[name]["description"]
            own = _own_terms(tools[name]["resource_type"])
            phrases = [_terms(phrase) for clause in DIRECTIVE.findall(description)
                       for phrase in PHRASE_BREAK.split(clause)]
            claims = [phrase for phrase in phrases if phrase and not phrase & own]
            for other in agent.tools:
                if other == name or other in description:
                    continue
                noun = _terms(tools[other]["resource_type"].rsplit(".", 1)[-1])
                for phrase in claims:
                    if noun and noun <= phrase and not noun <= own:
                        found.append((agent.name, name, other, " ".join(sorted(phrase))))
    return found


class Agent:
    def __init__(self, name, tools):
        self.name, self.tools = name, tools


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    register_capability_signatures()
    result = AgentRegistry()
    result.register_many(load_agent_directory(ROOT / "subagents/agents"))
    return result


def test_the_description_that_misrouted_backup_questions_is_caught():
    tools = dict(TOOLS, proxmox_database_inventory=dict(
        TOOLS["proxmox_database_inventory"],
        description="List database hosts. Use this for database servers, their backups and capacity."))
    agents = [Agent("proxmox-specialist", ["proxmox_database_inventory", "proxmox_backups"])]
    assert scope_collisions(agents, tools) == [
        ("proxmox-specialist", "proxmox_database_inventory", "proxmox_backups", "backup")]


def test_naming_the_other_tool_or_qualifying_by_the_own_resource_is_not_a_claim():
    tools = dict(TOOLS, proxmox_database_inventory=dict(
        TOOLS["proxmox_database_inventory"],
        description="Use this for database hosts and their backups; for backup ages use proxmox_backups."))
    agents = [Agent("proxmox-specialist", ["proxmox_database_inventory", "proxmox_backups"])]
    assert scope_collisions(agents, tools) == []
    tailnet = [Agent("networking-specialist", ["network_tailnet_devices", "network_router_health"])]
    assert scope_collisions(tailnet, TOOLS) == []


def test_a_negated_directive_states_where_a_scope_ends():
    tools = dict(TOOLS, proxmox_database_inventory=dict(
        TOOLS["proxmox_database_inventory"], description="Do not use this for backups."))
    agents = [Agent("proxmox-specialist", ["proxmox_database_inventory", "proxmox_backups"])]
    assert scope_collisions(agents, tools) == []


def test_no_two_tools_of_one_specialist_claim_the_same_resource(registry):
    assert scope_collisions(registry.list_agents(), TOOLS) == []
