import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import subprocess
import sys
import threading

import pytest

from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager


ARGUMENTS = {"path": "reviewed folder", "nested": {"value": "exact Unicode ✓"}}


class RecordingMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {"ok": True, "status": "success", "message": "Executed exact action"}


def manager_for(store, mcp=None):
    return ApprovalManager(store=store, mcp=mcp or RecordingMCP())


def create_store(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()
    store.create("decision", "workspace_mkdir", ARGUMENTS, "medium")
    return store


def test_rejection_is_durable_and_later_approval_cannot_execute_after_restart(tmp_path):
    store = create_store(tmp_path)
    manager = manager_for(store)
    first = asyncio.run(manager.reject_approval("decision"))
    assert first["approval"]["status"] == "rejected"
    assert first["approval"]["arguments"] == ARGUMENTS
    assert first["result"] is None
    assert first["replayed"] is False
    assert "rejected by the user" in first["error"]
    assert manager.mcp.calls == []
    assert store.list_pending() == []

    restarted = ApprovalStore(store.path)
    restarted.initialize()
    restarted_manager = manager_for(restarted)
    for decision in (restarted_manager.reject_approval, restarted_manager.approve_approval):
        replay = asyncio.run(decision("decision"))
        assert replay == {**first, "replayed": True}
    assert restarted_manager.mcp.calls == []
    assert restarted.claim_for_execution("decision") is False
    with pytest.raises(ValueError, match="not in executing state"):
        restarted.mark_approved("decision", {"ok": True})


def test_unknown_rejection_does_not_create_authority(tmp_path):
    store = create_store(tmp_path)
    result = asyncio.run(manager_for(store).reject_approval("missing"))
    assert result == {"ok": False, "error": "Approval 'missing' not found."}
    assert store.count() == 1


@pytest.mark.parametrize("status", ["executing", "approved", "failed"])
def test_late_rejection_reports_existing_authority_without_overwriting_evidence(tmp_path, status):
    store = create_store(tmp_path)
    assert store.claim_for_execution("decision")
    evidence = {"ok": status == "approved", "status": "success" if status == "approved" else "error"}
    if status == "approved":
        store.mark_approved("decision", evidence)
    elif status == "failed":
        store.mark_failed("decision", evidence)
    else:
        store.record_execution_result("decision", evidence)
    before = store.get("decision")
    manager = manager_for(store)
    result = asyncio.run(manager.reject_approval("decision"))
    assert result["approval"]["status"] == status
    assert result["result"] == evidence
    assert result["replayed"] is True
    assert result["ok"] is (status == "approved")
    assert "rejected" not in result.get("error", "")
    assert store.get("decision") == before
    assert manager.mcp.calls == []


def test_execution_and_rejection_compete_atomically_across_store_instances(tmp_path):
    store = create_store(tmp_path)
    other = ApprovalStore(store.path)
    other.initialize()
    barrier = threading.Barrier(2)

    def decide(function):
        barrier.wait(timeout=3)
        return function("decision")

    with ThreadPoolExecutor(max_workers=2) as executor:
        approved = executor.submit(decide, store.claim_for_execution)
        rejected = executor.submit(decide, other.reject_pending)
        approval_won, rejection_won = approved.result(timeout=5), rejected.result(timeout=5)
    assert approval_won != rejection_won
    assert store.get("decision").status == ("executing" if approval_won else "rejected")
    assert store.get("decision").arguments == ARGUMENTS


def test_approve_observes_rejection_that_wins_between_read_and_execution_claim(tmp_path, monkeypatch):
    store = create_store(tmp_path)
    other = ApprovalStore(store.path)
    other.initialize()
    original_claim = store.claim_for_execution

    def lose_claim(approval_id):
        assert other.reject_pending(approval_id)
        return original_claim(approval_id)

    monkeypatch.setattr(store, "claim_for_execution", lose_claim)
    manager = manager_for(store)
    result = asyncio.run(manager.approve_approval("decision"))
    assert result["approval"]["status"] == "rejected"
    assert result["replayed"] is True
    assert manager.mcp.calls == []


def test_reject_while_another_manager_executes_cannot_claim_to_stop_the_tool(tmp_path):
    store = create_store(tmp_path)
    other = ApprovalStore(store.path)
    other.initialize()

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        class BlockedMCP(RecordingMCP):
            async def call_tool(self, tool, arguments):
                started.set()
                await release.wait()
                return await super().call_tool(tool, arguments)

        mcp = BlockedMCP()
        executor = manager_for(store, mcp)
        rejector = manager_for(other)
        execution = asyncio.create_task(executor.approve_approval("decision"))
        await asyncio.wait_for(started.wait(), 2)
        try:
            rejected = await rejector.reject_approval("decision")
            assert rejected["approval"]["status"] == "executing"
            assert "rejected" not in rejected["error"]
            assert rejected["replayed"] is True
        finally:
            release.set()
        completed = await execution
        assert completed["approval"]["status"] == "approved"
        assert mcp.calls == [("workspace_mkdir", ARGUMENTS)]
        late = await rejector.reject_approval("decision")
        assert late["approval"]["status"] == "approved"
        assert late["result"] == completed["result"]
        assert rejector.mcp.calls == []

    asyncio.run(scenario())


def legacy_database(path):
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("""
            CREATE TABLE approvals (
                approval_id TEXT PRIMARY KEY, tool TEXT NOT NULL, arguments TEXT NOT NULL,
                risk TEXT NOT NULL, status TEXT NOT NULL, result TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CHECK (status IN ('pending', 'executing', 'approved', 'failed'))
            )
        """)
        connection.execute("CREATE INDEX approval_status_index ON approvals(status)")
        connection.execute("CREATE TABLE audit (approval_id TEXT)")
        connection.execute("""
            CREATE TRIGGER approval_decision_audit AFTER UPDATE OF status ON approvals
            BEGIN INSERT INTO audit VALUES (NEW.approval_id); END
        """)
        for status in ("pending", "executing", "approved", "failed"):
            connection.execute(
                "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (status, "workspace_mkdir", '{ "path": "exact ✓" }', "medium", status,
                 None if status == "pending" else '{ "ok": false, "message": "exact" }',
                 "2020-01-01 01:02:03", "2020-01-02 04:05:06"),
            )


def raw_rows(path):
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT * FROM approvals ORDER BY approval_id").fetchall()


def test_legacy_check_migration_preserves_every_row_and_schema_dependents(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    before = raw_rows(path)
    store = ApprovalStore(path)
    store.initialize()
    store.initialize()  # Repeated startup must not rebuild or alter evidence.
    assert raw_rows(path) == before
    assert store.reject_pending("pending")
    assert store.get("executing").status == "executing"
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT * FROM audit").fetchall() == [("pending",)]
        indexes = connection.execute("PRAGMA index_list(approvals)").fetchall()
        assert any(row[1] == "approval_status_index" for row in indexes)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE approvals SET status='invented' WHERE approval_id='failed'")


def test_interrupted_migration_rolls_back_all_ddl_and_rows(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    before = raw_rows(path)

    class InterruptedStore(ApprovalStore):
        def _migrate_rejection_constraint(self, connection):
            super()._migrate_rejection_constraint(connection)
            raise RuntimeError("Interrupted before commit")

    with pytest.raises(RuntimeError, match="Interrupted before commit"):
        InterruptedStore(path).initialize()
    assert raw_rows(path) == before
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE approvals SET status='rejected' WHERE approval_id='pending'")
    restarted = ApprovalStore(path)
    restarted.initialize()
    assert restarted.reject_pending("pending")


def test_process_crash_before_migration_commit_recovers_old_then_migrates(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    before = raw_rows(path)
    script = """
import os
import sys
from agent.approval_store import ApprovalStore
class CrashStore(ApprovalStore):
    def _migrate_rejection_constraint(self, connection):
        super()._migrate_rejection_constraint(connection)
        os._exit(23)
CrashStore(sys.argv[1]).initialize()
"""
    process = subprocess.run([sys.executable, "-c", script, str(path)], check=False, timeout=10)
    assert process.returncode == 23
    assert raw_rows(path) == before
    restarted = ApprovalStore(path)
    restarted.initialize()
    assert raw_rows(path) == before
    assert restarted.reject_pending("pending")


def test_concurrent_startup_migrates_once_without_losing_existing_authority(tmp_path):
    path = tmp_path / "legacy.db"
    legacy_database(path)
    before = raw_rows(path)
    barrier = threading.Barrier(2)

    def initialize():
        barrier.wait(timeout=3)
        ApprovalStore(path).initialize()

    with ThreadPoolExecutor(max_workers=2) as executor:
        tasks = [executor.submit(initialize) for _ in range(2)]
        for task in tasks:
            task.result(timeout=5)
    assert raw_rows(path) == before
    assert ApprovalStore(path).reject_pending("pending")
