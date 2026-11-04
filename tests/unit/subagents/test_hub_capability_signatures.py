"""The production hub knows which arguments every capability requires.

Required arguments come from each tool's executable signature (the registered
MCP callable), never from the model. The AI service builds its hub in its own
process, separate from the MCP server process, so building the hub must capture
those signatures there too. Without them a request that leaves out a required
value (an account without a family name) reaches the specialist instead of
asking for it, and the clarification cannot name what is missing.
"""
from pathlib import Path

import pytest

from config import ModelProfileSettings, Settings
from subagents.core.definitions.loader import load_agent_directory
from subagents.core.orchestration.intent_contract import parse_semantic_intent
from tools import runtime_schema
from tools.capability_contract import build_capability_contract
from tools.registry import TOOLS

AGENTS_DIR = Path(__file__).resolve().parents[3] / "subagents" / "agents"


@pytest.fixture
def no_captured_signatures(monkeypatch):
    # A fresh AI service process: nothing has captured executable signatures yet.
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})


def build_production_hub(tmp_path, monkeypatch):
    import subagents.hub as hub_module
    from subagents.hub import build_hub

    monkeypatch.setattr(hub_module, "load_agent_directory", lambda _directory: [])
    settings = Settings(_env_file=None, asset_backend="mock", knowledge_backend="mock", agents_dir=tmp_path,
                        model_profiles={"hub-main": ModelProfileSettings(backend="ministral", model_path=tmp_path)})

    class Models:
        def exists(self, _key):
            return True

        def model_profile(self, key):
            return settings.require_model_profile(key)

    return build_hub(settings=settings, model_manager=Models(), inference=object(), tool_gateway=object())


def test_building_the_hub_captures_every_capabilitys_executable_signature(
    no_captured_signatures, tmp_path, monkeypatch,
):
    build_production_hub(tmp_path, monkeypatch)

    assert set(runtime_schema.runtime_registered_tools()) == set(TOOLS)
    unresolved = [name for name in sorted(TOOLS)
                  if build_capability_contract(name, TOOLS[name]).requiredness_source == "unresolved"]
    assert unresolved == []
    contract = build_capability_contract("account_create", TOOLS["account_create"])
    assert contract.required_arguments == ("given_name", "family_name")


def test_an_account_request_without_a_family_name_asks_for_it(no_captured_signatures, tmp_path, monkeypatch):
    build_production_hub(tmp_path, monkeypatch)
    account = next(agent for agent in load_agent_directory(AGENTS_DIR) if agent.name == "account-specialist")

    intent = parse_semantic_intent({
        "summary": "Create a directory account for John.",
        "effect": "mutation",
        "allowed_tools": ["account_create"],
        "forbidden_tools": [],
        "allowed_arguments": {"given_name": ["John"], "role": ["software developer"]},
        "forbidden_arguments": {},
        "max_tool_calls": 1,
        "clarification_required": False,
    }, agent=account, user_request="can you make an account for John as a software developer")

    assert intent.clarification_required is True
    assert intent.missing_required_arguments == ["family_name"]
