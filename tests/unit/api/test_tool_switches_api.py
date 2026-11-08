"""Tool switches through the internal extensions API: owner-scoped, validated, shown in the catalog."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from config.settings import Settings
from services.extensions.service import extension_store


def client_for(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    # Real settings with nothing configured: the catalog reads each package's setup state from them.
    settings = Settings(_env_file=None, itsm_internal_job_token="test-internal-secret")
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


def test_the_catalog_names_each_tools_package_and_a_package_switch_saves_them_all(tmp_path, monkeypatch):
    call = client_for(tmp_path, monkeypatch)
    try:
        catalog = call("alice", "list", {}).json()
        packages = {row["id"]: row for row in catalog["tool_packages"]}
        tools = {row["id"]: row for row in catalog["capabilities"]}
        # Every registered tool belongs to exactly one listed package.
        assert all(row["provider"] in packages for row in tools.values())
        assert packages["openwrt"]["name"] == "OpenWrt" and packages["openwrt"]["kind"] in {"provider", "local"}
        # Each package says whether it is set up, by the same rule as the integration status.
        from services.integration_readiness import integration_readiness
        readiness = {row["id"]: row["state"] for row in integration_readiness(extensions.get_settings())}
        assert {row["id"]: row["state"] for row in catalog["tool_packages"]} == readiness
        off = call("alice", "tool_package_enable", {"package": "openwrt", "enabled": False})
        assert off.status_code == 200 and off.json()["enabled"] is False
        rows = {row["id"]: row for row in call("alice", "list", {}).json()["capabilities"]}
        assert all(row["enabled"] is False for row in rows.values() if row["provider"] == "openwrt")
        assert all(row["enabled"] is True for row in rows.values() if row["provider"] != "openwrt")
        refused = call("alice", "tool_package_enable", {"package": "nowhere", "enabled": False})
        assert refused.status_code == 422 and refused.json()["detail"]["code"] == "tool_package_not_found"
    finally:
        extension_store.cache_clear()
