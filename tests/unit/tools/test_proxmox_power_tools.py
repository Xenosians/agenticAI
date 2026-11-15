"""Proxmox guest power tools: approval-gated, exact-target policy, honest receipts."""
import asyncio

import pytest

from subagents.core.definitions.types import AgentDefinition
from subagents.core.tooling.gateway import ToolGateway
from tools.proxmox import catalog as proxmox_catalog
from tools.proxmox.catalog import PROXMOX_TOOLS
from tools.proxmox.mcp import register_proxmox_tools
from tools.registry import TOOLS, format_approval_required, format_tool_result

POWER_TOOLS = {"proxmox_guest_start": "start", "proxmox_guest_shutdown": "shutdown",
               "proxmox_guest_reboot": "reboot", "proxmox_guest_force_stop": "force_stop"}


class Target:
    """Answers power_target like ProxmoxService for one stopped LXC guest."""

    def __init__(self, status="stopped", lock=None):
        self.status = status
        self.lock = lock
        self.calls = []

    def power_target(self, node, vmid, action):
        self.calls.append((node, vmid, action))
        if (node, vmid) != ("hummingbird", 990):
            from services.provider_errors import ProviderResourceNotFound
            raise ProviderResourceNotFound(f"Proxmox VMID {vmid} was not found on node {node}.")
        allowed = (action == "start" and self.status == "stopped" and not self.lock) or (
            action != "start" and self.status == "running" and not self.lock)
        return {"ok": True, "status": "success", "provider": "proxmox", "data": {
            "node": node, "vmid": vmid, "type": "lxc", "name": "janus-lab", "status": self.status,
            "lock": self.lock, "template": False, "action": action, "allowed": allowed,
            "reason": None if allowed else f"LXC {vmid} (janus-lab) is {self.status}."}}

    def close(self):
        pass


@pytest.fixture
def target(monkeypatch):
    provider = Target()
    monkeypatch.setattr(proxmox_catalog, "_policy_service", lambda: provider)
    return provider


def test_four_power_tools_are_approval_gated_mutations_owned_by_the_proxmox_specialist():
    from subagents.core.definitions.loader import load_agent_definition

    agent = load_agent_definition("subagents/agents/proxmox-specialist.md")
    for name, action in POWER_TOOLS.items():
        tool = TOOLS[name]
        assert name in PROXMOX_TOOLS and name in agent.tools
        assert (tool["effect"], tool["requires_approval"], tool["operation_kind"]) == ("mutation", True, "action")
        assert tool["risk"] == ("high" if action == "force_stop" else "medium")
        assert tool["grounded_arguments"] == ["node", "vmid"]
        assert tool["trusted_policy_arguments"] == ["expected_type", "expected_name"]
        assert set(tool["parameters"]) == {"node", "vmid"}
        assert tool["parameters"]["vmid"]["type"] == "int"
        assert tool["policy_owns_preconditions"] is True


def test_policy_freezes_the_exact_guest_before_approval(target):
    policy = TOOLS["proxmox_guest_start"]["policy_resolver"]
    result = policy(node="hummingbird", vmid=990)
    assert result["ok"] is True and result["requires_approval"] is True
    assert result["execution_arguments"] == {"node": "hummingbird", "vmid": 990,
                                             "expected_type": "lxc", "expected_name": "janus-lab"}
    assert target.calls == [("hummingbird", 990, "start")]


def test_policy_refuses_an_action_that_cannot_apply_and_a_guest_that_is_not_there(target):
    refused = TOOLS["proxmox_guest_shutdown"]["policy_resolver"](node="hummingbird", vmid=990)
    assert (refused["ok"], refused["status"]) == (False, "denied")
    assert "stopped" in refused["error"]
    missing = TOOLS["proxmox_guest_start"]["policy_resolver"](node="hummingbird", vmid=989)
    assert (missing["ok"], missing["status"]) == (False, "denied")
    assert "989" in missing["error"]


def test_policy_without_a_configured_proxmox_says_so(monkeypatch):
    monkeypatch.setattr(proxmox_catalog, "_policy_service", lambda: None)
    result = TOOLS["proxmox_guest_start"]["policy_resolver"](node="hummingbird", vmid=990)
    assert (result["ok"], result["status"]) == (False, "error")
    assert "not set up" in result["error"]


def test_approval_text_names_the_exact_action_and_guest():
    arguments = {"node": "hummingbird", "vmid": 990, "expected_type": "lxc", "expected_name": "janus-lab"}
    text = format_approval_required("proxmox_guest_force_stop", arguments, "a1b2c3d4")
    assert "Force stop" in text and "LXC 990 (janus-lab)" in text and "hummingbird" in text
    assert "unsaved data" in text and "a1b2c3d4" in text


def test_result_text_reports_the_read_back_not_the_request():
    success = {"ok": True, "status": "executed", "action": "start", "node": "hummingbird", "vmid": 990,
               "type": "lxc", "name": "janus-lab", "observed_status": "running", "upid": "UPID:x",
               "verification_ok": True}
    assert "is now running" in format_tool_result("proxmox_guest_start", success)
    unknown = {"ok": False, "status": "outcome_unknown", "error": "Proxmox did not answer.", "action": "start",
               "vmid": 990, "node": "hummingbird"}
    assert "Do not repeat" in format_tool_result("proxmox_guest_start", unknown)


def test_mcp_registers_power_tools_that_pass_the_frozen_target_through():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    class Provider:
        calls = []

        def guest_power(self, node, vmid, action, *, expected_type, expected_name):
            self.calls.append((node, vmid, action, expected_type, expected_name))
            return {"ok": True, "status": "executed", "provider": "proxmox", "operation": "guest_power",
                    "action": action, "node": node, "vmid": vmid, "type": expected_type, "name": expected_name,
                    "expected_status": "running", "observed_status": "running", "upid": "UPID:x",
                    "task_exit_status": "OK", "changed": True, "verification_ok": True, "retry_safe": False,
                    "error": None, "observed_at": "2026-11-16T00:00:00+00:00"}

    server = Server()
    provider = Provider()
    register_proxmox_tools(server, provider)
    assert set(server.tools) == set(PROXMOX_TOOLS)
    result = server.tools["proxmox_guest_reboot"]("hummingbird", 990, expected_type="lxc", expected_name="janus-lab")
    assert (result.ok, result.status, result.verification_ok) == (True, "executed", True)
    assert provider.calls == [("hummingbird", 990, "reboot", "lxc", "janus-lab")]


def test_an_unconfigured_power_tool_reports_not_configured_and_never_acts():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    server = Server()
    register_proxmox_tools(server, None, missing_settings=("PROXMOX_BASE_URL",))
    result = server.tools["proxmox_guest_start"]("hummingbird", 990, expected_type="lxc", expected_name="janus-lab")
    assert (result.ok, result.status, result.error_code) == (False, "error", "not_configured")


def test_the_gateway_stops_a_power_proposal_at_approval_with_the_frozen_target(target):
    created = []

    def approval_creator(tool_name, arguments, risk=None):
        created.append((tool_name, dict(arguments), risk))
        return {"id": "a1b2c3d4"}

    class NoMCP:
        async def call_tool(self, *args):
            raise AssertionError("a power action must not run before approval")

    agent = AgentDefinition(name="proxmox-specialist", description="", model="hub-main",
                            tools=list(POWER_TOOLS), max_steps=3, system_prompt="")
    gateway = ToolGateway(approval_creator=approval_creator, mcp=NoMCP())
    result = asyncio.run(gateway.execute(agent, "Start container 990 on node hummingbird.",
                                         "proxmox_guest_start", {"node": "hummingbird", "vmid": 990}))
    assert result["status"] == "approval_required"
    assert created == [("proxmox_guest_start", {"node": "hummingbird", "vmid": 990, "expected_type": "lxc",
                                                "expected_name": "janus-lab"}, "medium")]
    ungrounded = asyncio.run(gateway.execute(agent, "Start container 990 on node hummingbird.",
                                             "proxmox_guest_start", {"node": "hummingbird", "vmid": 991}))
    assert ungrounded["status"] == "denied" and len(created) == 1
