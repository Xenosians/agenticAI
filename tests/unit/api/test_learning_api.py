"""Learning's internal HTTP boundary: only Phoenix may call it, always for one owner."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions, learning
from services.extensions.service import extension_store

TOKEN = "test-internal-secret"
HEADERS = {"x-internal-token": TOKEN}
CATALOG, PREFLIGHT = "/internal/v1/learning/catalog", "/internal/v1/learning/preflight"
RUNS, START, CANCEL = ("/internal/v1/learning/runs", "/internal/v1/learning/runs/start",
                       "/internal/v1/learning/runs/cancel")


def examples():
    return "\n".join(json.dumps({"messages": [{"role": "user", "content": f"question {index}"},
                                              {"role": "assistant", "content": f"answer {index}"}]})
                     for index in range(12))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    settings = SimpleNamespace(require_internal_job_token=lambda: TOKEN)
    monkeypatch.setattr(extensions, "get_settings", lambda: settings)
    extension_store.cache_clear()
    app = FastAPI()
    app.include_router(learning.router)
    try:
        yield TestClient(app)
    finally:
        extension_store.cache_clear()


def owned_selection(owner="alice"):
    store = extension_store()
    hub = store.put(owner, "model", {"name": "Hub model", "backend": "local", "model": "fixture", "state": "ready"})
    data = store.dataset(owner, name="Ticket triage", text=examples(), source="Owned tickets", license="Owned",
                         purpose="training", reviewed=True)
    return {"method": "sft", "role": "hub", "tuning": "qlora", "model_id": hub["id"],
            "dataset_id": data["id"], "dataset_sha256": data["sha256"]}


class FakeRegistry:
    """Records what the routes ask of the run registry for one owner."""

    def __init__(self):
        self.calls = []
        self.refusal = None

    def list(self, owner):
        self.calls.append(("list", owner))
        return [{"run_id": "a" * 32}]

    def start(self, owner, run, store):
        self.calls.append(("start", owner, run.model_dump()))
        if self.refusal:
            raise learning.LearningSelectionError(self.refusal)
        return {"run_id": "b" * 32, "state": "running"}

    def cancel(self, owner, run_id):
        self.calls.append(("cancel", owner, run_id))
        if self.refusal:
            raise learning.LearningSelectionError(self.refusal)
        return {"run_id": run_id, "state": "cancelled"}


@pytest.fixture
def runs(monkeypatch):
    registry = FakeRegistry()
    monkeypatch.setattr(learning, "run_registry", lambda: registry)
    return registry


def test_run_routes_require_the_internal_token(client, runs):
    run = {**owned_selection(), "preflight_sha256": "c" * 64}
    for path, body in ((RUNS, {"owner": "alice"}), (START, {"owner": "alice", "run": run}),
                       (CANCEL, {"owner": "alice", "run_id": "a" * 32})):
        assert client.post(path, json=body).status_code == 401
    assert runs.calls == []


def test_runs_are_listed_started_and_cancelled_for_the_named_owner(client, runs):
    run = {**owned_selection(), "preflight_sha256": "c" * 64}

    listed = client.post(RUNS, json={"owner": "alice"}, headers=HEADERS)
    started = client.post(START, json={"owner": "alice", "run": run}, headers=HEADERS)
    cancelled = client.post(CANCEL, json={"owner": "alice", "run_id": "b" * 32}, headers=HEADERS)

    assert listed.json() == {"schema": "learning-runs.v1", "runs": [{"run_id": "a" * 32}]}
    assert started.status_code == 200 and started.json()["state"] == "running"
    assert cancelled.json() == {"run_id": "b" * 32, "state": "cancelled"}
    assert runs.calls == [("list", "alice"), ("start", "alice", run), ("cancel", "alice", "b" * 32)]


@pytest.mark.parametrize("change", [{"preflight_sha256": "short"}, {"start_now": True}, {"owner": "mallory"}])
def test_a_malformed_run_request_never_reaches_the_registry(client, runs, change):
    run = {**owned_selection(), "preflight_sha256": "c" * 64, **change}

    response = client.post(START, json={"owner": "alice", "run": run}, headers=HEADERS)

    assert response.status_code == 422 and runs.calls == []


@pytest.mark.parametrize("code, status", [("learning_host_busy", 409), ("learning_check_changed", 422),
                                          ("learning_run_not_ready", 422)])
def test_run_refusals_keep_their_code(client, runs, code, status):
    runs.refusal = code
    run = {**owned_selection(), "preflight_sha256": "c" * 64}

    response = client.post(START, json={"owner": "alice", "run": run}, headers=HEADERS)

    assert response.status_code == status and response.json() == {"detail": code}


def test_cancelling_an_unavailable_run_is_not_found(client, runs):
    runs.refusal = "learning_run_unavailable"

    response = client.post(CANCEL, json={"owner": "alice", "run_id": "a" * 32}, headers=HEADERS)

    assert response.status_code == 404 and response.json() == {"detail": "learning_run_unavailable"}


def test_both_routes_require_the_internal_token(client):
    selection = owned_selection()
    for path, body in ((CATALOG, {"owner": "alice"}), (PREFLIGHT, {"owner": "alice", "selection": selection})):
        assert client.post(path, json=body).status_code == 401
        assert client.post(path, json=body, headers={"x-internal-token": "wrong"}).status_code == 401


def test_catalog_and_check_answer_only_for_the_named_owner(client):
    selection = owned_selection()

    mine = client.post(CATALOG, json={"owner": "alice"}, headers=HEADERS)
    assert mine.status_code == 200
    assert [row["id"] for row in mine.json()["models"]] == [selection["model_id"]]
    theirs = client.post(CATALOG, json={"owner": "bob"}, headers=HEADERS).json()
    assert (theirs["models"], theirs["datasets"]) == ([], [])

    receipt = client.post(PREFLIGHT, json={"owner": "alice", "selection": selection}, headers=HEADERS)
    # The fixture profile has no weights on this host, so the check is blocked on exactly that.
    assert receipt.status_code == 200 and receipt.json()["status"] == "blocked"
    assert receipt.json()["blockers"] == ["local_weights_unavailable"]
    refused = client.post(PREFLIGHT, json={"owner": "bob", "selection": selection}, headers=HEADERS)
    assert refused.status_code == 422 and refused.json() == {"detail": "learning_resource_unavailable"}


@pytest.mark.parametrize("change", [
    {"owner": "mallory"}, {"method": "rag"}, {"tuning": "everything"}, {"model_id": "../models"},
    {"dataset_sha256": "short"},
])
def test_malformed_selections_are_refused(client, change):
    selection = owned_selection()
    body = {"owner": "alice", "selection": {**selection, **change}}

    response = client.post(PREFLIGHT, json=body, headers=HEADERS)

    assert response.status_code == 422


def test_a_damaged_registry_is_unavailable_rather_than_empty(client):
    extension_store().put("alice", "dataset", {"name": "Damaged", "purpose": "training", "state": "ready",
                                               "sha256": "1" * 64, "source": "Owned", "license": "Owned",
                                               "content": {"train": [], "validation": []}})

    response = client.post(CATALOG, json={"owner": "alice"}, headers=HEADERS)

    assert response.status_code == 503 and response.json() == {"detail": "learning_service_unavailable"}
