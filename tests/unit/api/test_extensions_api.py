from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from services.extensions.service import extension_store


def test_internal_auth_and_private_dataset_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    settings = SimpleNamespace(require_internal_job_token=lambda: "test-internal-secret", model_profiles={})
    monkeypatch.setattr(extensions, "get_settings", lambda: settings)
    extension_store.cache_clear()
    app = FastAPI()
    app.include_router(extensions.router)
    client = TestClient(app)
    url = "/internal/v1/extensions"
    headers = {"x-internal-token": "test-internal-secret"}
    try:
        body = {"owner": "alice", "action": "dataset", "payload": {
            "name": "Operations manual", "source": "Owned manual", "license": "Owned",
            "text": "Router maintenance requires a planned window.", "purpose": "rag", "reviewed": True}}
        assert client.post(url, json=body).status_code == 401
        assert client.post(url, json=body, headers={"x-internal-token": "wrong"}).status_code == 401
        response = client.post(url, json=body, headers=headers)
        assert response.status_code == 200
        row = response.json()
        assert "content" not in row and "api_key" not in row
        body = {"owner": "alice", "action": "search", "payload": {"id": row["id"], "query": "router"}}
        assert client.post(url, json=body, headers=headers).json()["hits"][0]["dataset_id"] == row["id"]
        body["owner"] = "bob"
        assert client.post(url, json=body, headers=headers).status_code == 422
        assert client.post(url, json={"owner": "bob", "action": "list"}, headers=headers).json()["resources"] == []
    finally:
        extension_store.cache_clear()
