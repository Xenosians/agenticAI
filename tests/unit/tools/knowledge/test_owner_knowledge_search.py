"""knowledge_search reads the reviewed public snapshot and the requesting owner's reference data.

The owner comes from the trusted request context on the host, never from the model;
owner passages are quoted untrusted reference data with their citation.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from services.extensions.context import answer_references, owner_execution
from services.extensions.store import ExtensionStore
from services.knowledge import MockKnowledgeService
from tools.knowledge.owner_search import owner_knowledge_search

FREEZE = "Payroll cluster maintenance. The quarterly change freeze starts on the second Tuesday at 18:00 UTC."


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: store)
    return store


def add(store, owner="alice", text=FREEZE):
    return store.dataset(owner, name="Payroll runbook", source="Payroll runbook (owned)", license="Owned",
                         purpose="rag", reviewed=True, text=text)


def test_the_owners_reference_passages_are_added_to_the_tool_result(store):
    row = add(store)
    with owner_execution("alice", "payroll change freeze"):
        result = owner_knowledge_search({"ok": True, "status": "success", "hits": [], "count": 0},
                                        {"query": "payroll change freeze"})
        references = answer_references()
    assert result["ok"] and result["status"] == "success"
    assert result["reference_passages"] == [{"source": "Payroll runbook (owned)", "text": FREEZE,
                                             "trust": "untrusted_reference"}]
    assert result["count"] == 1
    assert [item["locator"] for item in references] == [f"dataset:{row['id']}#chunk:1@{row['sha256']}"]


def test_another_owners_data_is_never_returned(store):
    add(store, owner="bob")
    with owner_execution("alice", "payroll change freeze"):
        result = owner_knowledge_search({"ok": True, "status": "success", "hits": [], "count": 0},
                                        {"query": "payroll change freeze"})
    assert "reference_passages" not in result and result["count"] == 0


def test_outside_a_request_nothing_is_added(store):
    add(store)
    result = owner_knowledge_search({"ok": True, "status": "success", "hits": [], "count": 0},
                                    {"query": "payroll change freeze"})
    assert "reference_passages" not in result


def test_a_runbook_filter_never_returns_owner_passages(store):
    add(store)
    with owner_execution("alice", "payroll change freeze"):
        result = owner_knowledge_search({"ok": True, "status": "success", "hits": [], "count": 0},
                                        {"query": "payroll change freeze", "kind": "runbook"})
    assert "reference_passages" not in result


def test_the_result_formatter_quotes_owner_passages_as_reference_text():
    from tools.registry import format_tool_result
    text = format_tool_result("knowledge_search", {
        "ok": True, "status": "success", "count": 1, "hits": [],
        "reference_passages": [{"source": "Payroll runbook (owned)", "text": FREEZE,
                                "trust": "untrusted_reference"}]})
    assert "Payroll runbook (owned)" in text and "second Tuesday" in text
    assert "reference" in text.lower()


def test_the_gateway_adds_owner_passages_to_a_successful_search(store):
    from subagents.core.tooling.gateway import ToolGateway
    add(store)

    class Mcp:
        async def call_tool(self, name, arguments):
            return {"ok": True, "status": "success", "hits": [], "count": 0, "truncated": False}

    gateway = ToolGateway(approval_creator=lambda *args, **kwargs: None, mcp=Mcp())
    agent = SimpleNamespace(name="knowledge-specialist", tools=["knowledge_search"])
    with owner_execution("alice", "When does the payroll change freeze start?"):
        result = asyncio.run(gateway.execute(agent, "When does the payroll change freeze start?",
                                             "knowledge_search", {"query": "payroll change freeze start"}))
    assert result["ok"] and result["result"]["reference_passages"][0]["text"] == FREEZE
    assert json.dumps(result["result"]).count("sha256") == 0
