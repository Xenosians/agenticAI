"""Tool switches through the internal extensions API: owner-scoped, validated, shown in the catalog."""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from services.extensions.service import extension_store


def client_for(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    settings = SimpleNamespace(require_internal_job_token=lambda: "test-internal-secret", model_profiles={},
                               agents_dir=None)
    monkeypatch.setattr(extensions, "get_settings", lambda: settings)
    extension_store.cache_clear()
    app = FastAPI()
    app.include_router(extensions.router)
    client = TestClient(app)
    headers = {"x-internal-token": "test-internal-secret"}

    def call(owner, action, payload):
        return client.post("/internal/v1/extensions", json={"owner": owner, "action": action, "payload": payload},
                           headers=headers)
    return call


def test_an_owner_turns_a_tool_off_and_on_and_the_catalog_shows_it(tmp_path, monkeypatch):
    call = client_for(tmp_path, monkeypatch)
    try:
        off = call("alice", "tool_enable", {"id": "account_status", "enabled": False})
        assert off.status_code == 200 and off.json() == {"id": "account_status", "enabled": False}
        rows = {row["id"]: row for row in call("alice", "list", {}).json()["capabilities"]}
        assert rows["account_status"]["enabled"] is False
        assert all(row["enabled"] is True for name, row in rows.items() if name != "account_status")
        bob = {row["id"]: row for row in call("bob", "list", {}).json()["capabilities"]}
        assert bob["account_status"]["enabled"] is True
        assert call("alice", "tool_enable", {"id": "account_status", "enabled": True}).json()["enabled"] is True
        rows = {row["id"]: row for row in call("alice", "list", {}).json()["capabilities"]}
        assert rows["account_status"]["enabled"] is True
    finally:
        extension_store.cache_clear()


def test_malformed_switch_requests_are_refused_with_the_reason(tmp_path, monkeypatch):
    call = client_for(tmp_path, monkeypatch)
    try:
        for payload, code in (({"id": "no_such_tool", "enabled": False}, "tool_not_found"),
                              ({"enabled": False}, "tool_not_found"),
                              ({"id": "account_status", "enabled": "false"}, "tool_enabled_invalid"),
                              ({"id": "account_status"}, "tool_enabled_invalid")):
            response = call("alice", "tool_enable", payload)
            assert response.status_code == 422, payload
            assert response.json()["detail"]["code"] == code, payload
        # A non-text identifier is refused by the shared field check before any action runs.
        assert call("alice", "tool_enable", {"id": 7, "enabled": False}).status_code == 422
        rows = {row["id"]: row for row in call("alice", "list", {}).json()["capabilities"]}
        assert all(row["enabled"] is True for row in rows.values())
    finally:
        extension_store.cache_clear()
