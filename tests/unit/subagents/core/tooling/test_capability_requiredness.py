"""A specialist sees the same required and optional arguments the guard enforces.

Which arguments a capability requires comes from its executable signature. The
model-facing catalog read only the catalog's own `required_arguments` key, so most
tools (account creation among them) showed no required arguments and no way to
tell optional ones apart. The account specialist then stopped to ask for an
optional department instead of proposing the account.
"""
import pytest

from mcp_server import register_capability_signatures
from subagents.core.tooling.capabilities import build_capability_spec
from tools import runtime_schema
from tools.capability_contract import build_capability_contract
from tools.registry import TOOLS


@pytest.fixture
def signatures(monkeypatch):
    monkeypatch.setattr(runtime_schema, "_SCHEMAS", {})
    register_capability_signatures()


def test_every_capability_shows_the_trusted_required_and_optional_arguments(signatures):
    mismatched = {}
    for name in sorted(TOOLS):
        spec = build_capability_spec(name, include_arguments=True)
        trusted = build_capability_contract(name, TOOLS[name])
        shown = (spec.get("required_arguments"), spec.get("optional_arguments"))
        expected = (list(trusted.required_arguments), list(trusted.optional_arguments))
        if shown != expected:
            mismatched[name] = {"shown": shown, "trusted": expected}

    assert mismatched == {}


def test_account_creation_needs_only_the_two_names(signatures):
    spec = build_capability_spec("account_create", include_arguments=True)

    assert spec["required_arguments"] == ["given_name", "family_name"]
    assert spec["optional_arguments"] == ["department", "role"]
