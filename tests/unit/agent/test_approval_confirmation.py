"""A destructive Proxmox change needs the guest's exact name typed (owner decision, 17 November).

The check runs in the approval authority before the execution claim: a missing
or different name leaves the approval pending, executes nothing, and never
consumes the owner's one decision.
"""
import asyncio

import pytest

import agent.approvals as approvals_module
from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager, approval_to_dict


class RecordingMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool, arguments, *, owner=None):
        self.calls.append((tool, arguments, owner))
        return {"ok": True, "status": "executed", "verification_ok": True}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    def tool(name):
        if name == "proxmox_guest_delete":
            return {"risk": "high", "requires_approval": True,
                    "confirmation_argument": "expected_name"}
        return {"risk": "medium", "requires_approval": True}

    monkeypatch.setattr(approvals_module, "get_tool", tool)
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()
    mcp = RecordingMCP()
    return ApprovalManager(store=store, mcp=mcp, tool_lookup=tool), store, mcp


def create(manager, name="janus-lab"):
    return manager.create_approval(
        "proxmox_guest_delete",
        {"node": "hummingbird", "vmid": 990, "expected_type": "lxc", "expected_name": name},
    )


def test_a_destructive_approval_names_what_the_owner_must_type(manager):
    approval_manager, _, _ = manager
    approval = create(approval_manager)
    assert approval["confirmation_text"] == "janus-lab"
    assert approval_to_dict(approval_manager.store.get(approval["id"]))["confirmation_text"] == "janus-lab"


def test_without_the_typed_name_nothing_runs_and_the_approval_stays_pending(manager):
    approval_manager, store, mcp = manager
    approval = create(approval_manager)

    result = asyncio.run(approval_manager.approve_approval(approval["id"]))

    assert result["ok"] is False
    assert result["error_code"] == "confirmation_required"
    assert "exact name" in result["error"]
    assert store.get(approval["id"]).status == "pending"
    assert mcp.calls == []


def test_a_different_name_never_executes_and_keeps_the_decision_open(manager):
    approval_manager, store, mcp = manager
    approval = create(approval_manager)

    result = asyncio.run(approval_manager.approve_approval(approval["id"], confirm_name="janus-lab-2"))

    assert result["ok"] is False
    assert result["error_code"] == "confirmation_mismatch"
    assert store.get(approval["id"]).status == "pending"
    assert mcp.calls == []


def test_the_exact_name_executes_once_and_a_replay_does_not_ask_again(manager):
    approval_manager, store, mcp = manager
    approval = create(approval_manager)

    result = asyncio.run(approval_manager.approve_approval(approval["id"], confirm_name="janus-lab"))
    assert result["ok"] is True
    assert store.get(approval["id"]).status == "approved"
    assert mcp.calls == [("proxmox_guest_delete",
                          {"node": "hummingbird", "vmid": 990, "expected_type": "lxc",
                           "expected_name": "janus-lab"}, None)]

    replay = asyncio.run(approval_manager.approve_approval(approval["id"]))
    assert replay["replayed"] is True
    assert len(mcp.calls) == 1


def test_a_destructive_approval_without_its_frozen_name_cannot_run_at_all(manager):
    approval_manager, store, mcp = manager
    approval = approval_manager.create_approval(
        "proxmox_guest_delete", {"node": "hummingbird", "vmid": 990})

    result = asyncio.run(approval_manager.approve_approval(approval["id"], confirm_name=""))

    assert result["error_code"] == "confirmation_required"
    assert store.get(approval["id"]).status == "pending"
    assert mcp.calls == []
