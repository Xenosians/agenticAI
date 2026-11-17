"""An approved action runs for the account that proposed it (owner decision, 17 November).

Each account connects its own Proxmox and Tailscale. The proposal records its
owner with the approval, durably, and the approved action runs with that
owner's connection, whichever request carries the decision. Approvals stored
before owners were recorded keep working (no owner: the machine default).
"""
import asyncio
import sqlite3

from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager
from services.extensions.context import request_owner


class RecordingMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool, arguments, *, owner=None):
        self.calls.append((tool, arguments, owner))
        return {"ok": True, "status": "success", "message": "Executed exact action"}


def test_a_proposal_records_the_proposing_owner_durably(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()
    manager = ApprovalManager(store=store, mcp=RecordingMCP())
    token = request_owner.set("alice")
    try:
        approval = manager.create_approval("workspace_mkdir", {"path": "reviewed"})
    finally:
        request_owner.reset(token)
    restarted = ApprovalStore(store.path)
    restarted.initialize()
    assert restarted.get(approval["id"]).owner == "alice"
    assert approval["owner"] == "alice"


def test_the_approved_action_runs_for_the_proposing_owner_not_the_deciding_request(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()
    mcp = RecordingMCP()
    manager = ApprovalManager(store=store, mcp=mcp)
    token = request_owner.set("alice")
    try:
        approval = manager.create_approval("workspace_mkdir", {"path": "reviewed"})
    finally:
        request_owner.reset(token)
    token = request_owner.set("bob")
    try:
        result = asyncio.run(manager.approve_approval(approval["id"]))
    finally:
        request_owner.reset(token)
    assert result["ok"] is True
    assert mcp.calls == [("workspace_mkdir", {"path": "reviewed"}, "alice")]


def test_an_approval_without_an_owner_runs_with_none(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()
    mcp = RecordingMCP()
    manager = ApprovalManager(store=store, mcp=mcp)
    approval = manager.create_approval("workspace_mkdir", {"path": "reviewed"})
    asyncio.run(manager.approve_approval(approval["id"]))
    assert mcp.calls == [("workspace_mkdir", {"path": "reviewed"}, None)]
    assert approval["owner"] is None


def test_a_store_from_before_owners_gains_the_column_and_keeps_every_row(tmp_path):
    path = tmp_path / "approvals.db"
    legacy = ApprovalStore(path)
    legacy.initialize()
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE approvals DROP COLUMN owner") if "owner" in {
            row[1] for row in connection.execute("PRAGMA table_info(approvals)")} else None
        connection.execute("INSERT INTO approvals (approval_id, tool, arguments, risk, status) "
                           "VALUES ('old1', 'workspace_mkdir', '{\"path\":\"x\"}', 'medium', 'pending')")
    upgraded = ApprovalStore(path)
    upgraded.initialize()
    upgraded.initialize()
    entry = upgraded.get("old1")
    assert entry.owner is None
    assert entry.arguments == {"path": "x"}
    assert entry.status == "pending"
