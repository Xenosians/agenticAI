"""Dashboard power actions go through the same governed gateway as chat, for the authenticated owner."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from api import proxmox_actions
from subagents.core.definitions.types import AgentDefinition

HEADERS = {"x-internal-token": "test-internal-token"}
SPECIALIST = AgentDefinition(name="proxmox-specialist", description="", model="hub-main",
                             tools=["proxmox_guest_start", "proxmox_guest_shutdown", "proxmox_guest_reboot",
                                    "proxmox_guest_force_stop"], max_steps=3, system_prompt="")


class Gateway:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def execute(self, agent, user_input, tool_name, arguments):
        from services.extensions.context import disabled_tools, request_owner
        self.calls.append((agent.name, user_input, tool_name, dict(arguments), request_owner.get(),
                           frozenset(disabled_tools.get())))
        return self.result


class Registry:
    def get(self, name):
        assert name == "proxmox-specialist"
        return SPECIALIST


def client_for(monkeypatch, gateway, disabled=frozenset()):
    runtime = SimpleNamespace(tool_gateway=gateway, hub=SimpleNamespace(runtime=SimpleNamespace(
        agent_registry=Registry())), settings=SimpleNamespace())
    monkeypatch.setattr(proxmox_actions, "require_ready_runtime", lambda request: runtime)
    monkeypatch.setattr(proxmox_actions, "owner_disabled_tools", lambda owner: disabled)
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: "test-internal-token"))
    application = FastAPI()
    application.include_router(proxmox_actions.router)
    return TestClient(application)


APPROVAL = {"ok": True, "status": "approval_required", "decision_code": "approval_required",
            "tool": "proxmox_guest_start", "risk": "medium", "approval_id": "a1b2c3d4",
            "approval_arguments": {"node": "hummingbird", "vmid": 990, "expected_type": "lxc",
                                   "expected_name": "janus-lab"}}


def test_proposal_requires_internal_authentication(monkeypatch):
    client = client_for(monkeypatch, Gateway(APPROVAL))
    response = client.post("/internal/v1/proxmox/actions/propose",
                           json={"owner": "user-1", "node": "hummingbird", "vmid": 990, "action": "start"})
    assert response.status_code == 401


def test_a_proposal_creates_one_governed_approval_for_the_exact_target(monkeypatch):
    gateway = Gateway(APPROVAL)
    client = client_for(monkeypatch, gateway, disabled=frozenset({"ticket_create"}))
    response = client.post("/internal/v1/proxmox/actions/propose", headers=HEADERS,
                           json={"owner": "user-1", "node": "hummingbird", "vmid": 990, "action": "start"})
    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "approval_required", "approval_id": "a1b2c3d4", "tool": "proxmox_guest_start",
                    "action": "start", "risk": "medium", "node": "hummingbird", "vmid": 990, "type": "lxc",
                    "name": "janus-lab", "approval_text": body["approval_text"]}
    assert body["approval_text"].startswith("Start LXC 990 (janus-lab) on Proxmox node hummingbird.")
    agent, user_input, tool, arguments, owner, disabled = gateway.calls[0]
    assert (agent, tool, arguments, owner) == ("proxmox-specialist", "proxmox_guest_start",
                                               {"node": "hummingbird", "vmid": 990}, "user-1")
    assert "990" in user_input and "hummingbird" in user_input
    assert disabled == frozenset({"ticket_create"})


@pytest.mark.parametrize("result,status,code", [
    ({"ok": False, "status": "denied", "decision_code": "policy_denied",
      "error": "LXC 990 (janus-lab) is already running."}, 409, "proxmox_action_refused"),
    ({"ok": False, "status": "denied", "decision_code": "tool_disabled_by_owner",
      "error": "The tool 'proxmox_guest_start' is turned off in your MCP Tool Store."}, 403, "proxmox_action_disabled"),
    ({"ok": False, "status": "error", "decision_code": "policy_denied",
      "error": "Proxmox VE is not set up on this Janus host."}, 503, "proxmox_action_unavailable"),
])
def test_a_refused_proposal_says_why_and_creates_nothing(monkeypatch, result, status, code):
    client = client_for(monkeypatch, Gateway(result))
    response = client.post("/internal/v1/proxmox/actions/propose", headers=HEADERS,
                           json={"owner": "user-1", "node": "hummingbird", "vmid": 990, "action": "start"})
    assert response.status_code == status
    assert response.json()["detail"] == {"code": code, "message": result["error"]}


@pytest.mark.parametrize("payload", [
    {"owner": "user-1", "node": "hummingbird", "vmid": 990, "action": "delete"},
    {"owner": "user-1", "node": "hummingbird", "vmid": "990", "action": "start"},
    {"owner": "user-1", "node": "../x", "vmid": 990, "action": "start"},
    {"owner": "user-1", "node": "hummingbird", "vmid": 990, "action": "start", "expected_type": "qemu"},
])
def test_only_an_exact_node_vmid_and_one_of_four_actions_are_accepted(monkeypatch, payload):
    gateway = Gateway(APPROVAL)
    client = client_for(monkeypatch, gateway)
    response = client.post("/internal/v1/proxmox/actions/propose", headers=HEADERS, json=payload)
    assert response.status_code == 422
    assert gateway.calls == []
