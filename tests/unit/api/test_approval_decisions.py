import importlib
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from agent.approval_store import ApprovalStore
from agent.approvals import ApprovalManager
from api import extensions


api_module = importlib.import_module("api.app")
HEADERS = {"x-internal-token": "test-decision-token"}


class RecordingMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(self, tool, arguments):
        self.calls.append((tool, arguments))
        return {"ok": True, "status": "success"}


@pytest.fixture
def approval_api(tmp_path, monkeypatch):
    store = ApprovalStore(tmp_path / "approvals.db")
    store.initialize()
    store.create("decision", "workspace_mkdir", {"path": "exact reviewed folder"}, "medium")
    mcp = RecordingMCP()
    manager = ApprovalManager(store=store, mcp=mcp)
    runtime = SimpleNamespace(
        ready=True, approvals=manager, learning_hooks=None,
        trajectory_recorder=SimpleNamespace(path=tmp_path / "trajectory.jsonl"),
    )
    monkeypatch.setattr(api_module, "application_runtime", lambda _: runtime)
    monkeypatch.setattr(api_module, "capture_approval_execution_evidence", lambda **_: [])
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: HEADERS["x-internal-token"],
    ))
    # These route tests inject only their owned approval manager; do not start
    # the application's real model/GPU lifespan.
    client = TestClient(api_module.app, raise_server_exceptions=True)
    try:
        yield client, store, mcp
    finally:
        client.close()


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("headers", [{}, {"x-internal-token": "wrong-token"}])
def test_approval_decisions_require_internal_auth_before_touching_authority(
    approval_api, decision, headers,
):
    client, store, mcp = approval_api
    response = client.post(f"/v1/approvals/decision/{decision}", headers=headers)
    assert response.status_code == 401
    assert store.get("decision").status == "pending"
    assert mcp.calls == []


def test_authenticated_rejection_and_later_approval_return_same_durable_human_decision(approval_api):
    client, store, mcp = approval_api
    first = client.post("/v1/approvals/decision/reject", headers=HEADERS)
    assert first.status_code == 200
    assert first.json() == {
        "approval_id": "decision",
        "protocol_status": "processed",
        "execution_status": "rejected",
        "approval_status": "rejected",
        "replayed": False,
        "result": None,
        "error": "Action rejected by the user. No tool was executed.",
    }
    for decision in ("reject", "approve"):
        replay = client.post(f"/v1/approvals/decision/{decision}", headers=HEADERS)
        assert replay.status_code == 200
        assert replay.json() == {**first.json(), "replayed": True}
    assert store.get("decision").arguments == {"path": "exact reviewed folder"}
    assert mcp.calls == []


def test_authenticated_approval_still_executes_exact_persisted_action_once(approval_api):
    client, store, mcp = approval_api
    response = client.post("/v1/approvals/decision/approve", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["execution_status"] == "succeeded"
    late = client.post("/v1/approvals/decision/reject", headers=HEADERS)
    assert late.status_code == 200
    assert late.json() == {**response.json(), "replayed": True}
    assert mcp.calls == [("workspace_mkdir", {"path": "exact reviewed folder"})]
    assert store.get("decision").status == "approved"


@pytest.mark.parametrize("status, execution_status", [
    ("executing", "unresolved"), ("failed", "failed"),
])
def test_rejection_preserves_existing_execution_authority(approval_api, status, execution_status):
    client, store, mcp = approval_api
    assert store.claim_for_execution("decision")
    evidence = {"ok": False, "status": "error", "message": "Persisted provider detail"}
    if status == "failed":
        store.mark_failed("decision", evidence)
    else:
        store.record_execution_result("decision", evidence)
    response = client.post("/v1/approvals/decision/reject", headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["protocol_status"] == "processed"
    assert response.json()["execution_status"] == execution_status
    assert response.json()["approval_status"] == status
    assert response.json()["result"] == evidence
    assert response.json()["replayed"] is True
    assert "rejected" not in response.json()["error"]
    assert mcp.calls == []


@pytest.mark.parametrize("decision", ["reject", "approve"])
def test_unknown_approval_is_a_protocol_failure(approval_api, decision):
    client, store, mcp = approval_api
    response = client.post(f"/v1/approvals/missing/{decision}", headers=HEADERS)
    assert response.status_code == 409
    assert store.get("decision").status == "pending"
    assert mcp.calls == []


def test_openapi_declares_rejection_as_a_processed_typed_outcome():
    schema = api_module.app.openapi()
    response = schema["paths"]["/v1/approvals/{approval_id}/reject"]["post"]["responses"]["200"]
    assert response["content"]["application/json"]["schema"]["$ref"].endswith("/ApprovalProtocolResponse")
    properties = schema["components"]["schemas"]["ApprovalProtocolResponse"]["properties"]
    assert "rejected" in properties["execution_status"]["enum"]
    assert "rejected" in properties["approval_status"]["enum"]
