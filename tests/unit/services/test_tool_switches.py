"""Owner switches for MCP tools: a turned-off tool is never run for that owner."""
import asyncio

import pytest

from services.extensions.context import disabled_tools, owner_execution
from services.extensions.store import ExtensionStore
from services.extensions.tool_switches import ToolSwitchError, ToolSwitches
from subagents.core.definitions.types import AgentDefinition
from subagents.core.tooling.gateway import ToolGateway


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: store)
    return store


def test_a_tool_is_on_until_its_owner_turns_it_off_and_only_for_that_owner(store):
    switches = ToolSwitches(store)
    assert switches.disabled("alice") == frozenset()
    assert switches.set_enabled("alice", "account_status", False) == {"id": "account_status", "enabled": False}
    assert switches.disabled("alice") == frozenset({"account_status"})
    assert switches.disabled("bob") == frozenset()
    assert switches.set_enabled("alice", "account_status", True) == {"id": "account_status", "enabled": True}
    assert switches.disabled("alice") == frozenset()


def test_only_registered_tools_and_real_booleans_are_accepted(store):
    switches = ToolSwitches(store)
    for name in ("no_such_tool", "", None, 7, "account_status; DROP"):
        with pytest.raises(ToolSwitchError) as refused:
            switches.set_enabled("alice", name, False)
        assert refused.value.code == "tool_not_found"
    for value in ("false", 0, None):
        with pytest.raises(ToolSwitchError) as refused:
            switches.set_enabled("alice", "account_status", value)
        assert refused.value.code == "tool_enabled_invalid"
    assert switches.disabled("alice") == frozenset()


def test_the_request_carries_the_owners_turned_off_tools(store):
    ToolSwitches(store).set_enabled("alice", "account_status", False)
    assert disabled_tools.get() == frozenset()
    with owner_execution("alice", "Is jdoe locked out?"):
        assert disabled_tools.get() == frozenset({"account_status"})
        with owner_execution("bob", "Is jdoe locked out?"):
            assert disabled_tools.get() == frozenset()
        assert disabled_tools.get() == frozenset({"account_status"})
    assert disabled_tools.get() == frozenset()


def gateway_for_test():
    class MCP:
        def __init__(self):
            self.calls = []

        async def call_tool(self, *args):
            self.calls.append(args)
            return {"ok": True, "status": "success", "result": {"user_id": "jdoe", "locked": False}}

    def approval(*args, **kwargs):
        pytest.fail("A read tool must not create an approval")

    mcp = MCP()
    return ToolGateway(approval_creator=approval, mcp=mcp), mcp


AGENT = AgentDefinition(name="account-specialist", description="Accounts", model="test", tools=["account_status"])


def test_the_gateway_refuses_a_turned_off_tool_before_policy_approval_or_provider(store):
    ToolSwitches(store).set_enabled("alice", "account_status", False)
    gateway, mcp = gateway_for_test()
    with owner_execution("alice", "Is jdoe locked out?"):
        result = asyncio.run(gateway.execute(AGENT, "Is jdoe locked out?", "account_status", {"user_id": "jdoe"}))
    assert result["ok"] is False and result["status"] == "denied"
    assert result["decision_code"] == "tool_disabled_by_owner"
    assert "turned off" in result["error"] and mcp.calls == []


def test_another_owners_switch_does_not_affect_this_request(store):
    ToolSwitches(store).set_enabled("bob", "account_status", False)
    gateway, mcp = gateway_for_test()
    with owner_execution("alice", "Is jdoe locked out?"):
        result = asyncio.run(gateway.execute(AGENT, "Is jdoe locked out?", "account_status", {"user_id": "jdoe"}))
    assert result["decision_code"] != "tool_disabled_by_owner"
    assert len(mcp.calls) == 1


def test_learning_records_an_owner_switch_as_the_owners_choice_not_a_model_failure():
    from learning.continual.automation.canary import model_attributable_failure
    from learning.curation.engine import KNOWN_OUTCOME_CODES
    from learning.evidence.quality import FAILURE_CATEGORY_BY_CODE

    assert "tool_disabled_by_owner" in KNOWN_OUTCOME_CODES
    category = FAILURE_CATEGORY_BY_CODE["tool_disabled_by_owner"]
    assert category == "owner_disabled_tool"
    assert not model_attributable_failure(outcome_codes=["tool_disabled_by_owner"], failure_types=[category])
