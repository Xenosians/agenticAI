from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from api import proxmox_actions
from config import Settings


class ProviderSnapshot:
    def overview(self):
        return {
            "ok": True,
            "status": "success",
            "provider": "proxmox",
            "observed_at": "2026-11-06T12:00:00+00:00",
            "data": {"schema": "proxmox-overview.v1", "cluster": {"members": []},
                     "nodes": {"nodes": []}, "guests": {"guests": []}},
        }

    def close(self):
        pass


def client_for(settings, monkeypatch, service_factory):
    monkeypatch.setattr(proxmox_actions, "require_ready_runtime", lambda request: SimpleNamespace(settings=settings))
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: "test-internal-token"))
    monkeypatch.setattr("services.proxmox.build_proxmox_service", service_factory)
    application = FastAPI()
    application.add_api_route("/internal/v1/proxmox/overview", proxmox_actions.proxmox_overview, methods=["GET"])
    return TestClient(application)


def test_overview_requires_internal_authentication(monkeypatch):
    client = client_for(Settings(_env_file=None), monkeypatch, lambda settings: ProviderSnapshot())
    assert client.get("/internal/v1/proxmox/overview").status_code == 401


def test_overview_reports_unconfigured_provider_without_fabricating_data(monkeypatch):
    client = client_for(Settings(_env_file=None), monkeypatch, lambda settings: None)
    response = client.get("/internal/v1/proxmox/overview", headers={"x-internal-token": "test-internal-token"})
    assert response.status_code == 503
    assert response.json() == {"error": "proxmox_not_configured"}


def test_overview_returns_the_provider_snapshot_and_closes_its_client(monkeypatch):
    provider = ProviderSnapshot()
    client = client_for(Settings(_env_file=None), monkeypatch, lambda settings: provider)
    response = client.get("/internal/v1/proxmox/overview", headers={"x-internal-token": "test-internal-token"})
    assert response.status_code == 200
    assert response.json()["data"]["schema"] == "proxmox-overview.v1"


class FailingSnapshot:
    def __init__(self, error):
        self.error = error

    def overview(self):
        raise self.error

    def close(self):
        pass


def test_a_failed_overview_says_why_without_echoing_the_provider(monkeypatch):
    import httpx

    request = httpx.Request("GET", "https://pve.example.test/api2/json/cluster/status")
    cases = [
        (httpx.ConnectError("dns failure for pve.example.test", request=request), "provider_unreachable"),
        (httpx.ReadTimeout("slow", request=request), "provider_timeout"),
        (RuntimeError("Proxmox refused the read (HTTP 401)."), "provider_error"),
    ]
    for error, reason in cases:
        client = client_for(Settings(_env_file=None), monkeypatch, lambda settings, error=error: FailingSnapshot(error))
        response = client.get("/internal/v1/proxmox/overview", headers={"x-internal-token": "test-internal-token"})
        assert response.status_code == 502
        body = response.json()
        assert body["error"] == "proxmox_read_failed"
        assert body["reason"] == reason
        assert isinstance(body["message"], str) and body["message"]
        assert "pve.example.test" not in body["message"]


def test_a_lua_refusal_is_not_reported_as_an_invalid_configuration(monkeypatch):
    from services.lua_adapters.runtime import LuaAdapterError

    client = client_for(Settings(_env_file=None), monkeypatch,
                        lambda settings: FailingSnapshot(LuaAdapterError("lua_adapter_time_limit")))
    response = client.get("/internal/v1/proxmox/overview", headers={"x-internal-token": "test-internal-token"})
    assert response.status_code == 502
    assert response.json()["reason"] == "lua_adapter_time_limit"
