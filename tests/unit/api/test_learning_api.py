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
    assert receipt.status_code == 200 and receipt.json()["status"] == "blocked"
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
