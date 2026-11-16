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
    application.add_api_route("/internal/v1/proxmox/health", proxmox_actions.proxmox_health, methods=["GET"])
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


class ProviderHealth:
    """Backups (one cluster-wide read set) and each node's disks."""

    def __init__(self, fail_disks_on=()):
        self.fail_disks_on = fail_disks_on
        self.closed = False

    def nodes(self):
        return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T08:00:00+00:00",
                "data": {"count": 2, "truncated": False, "nodes": [{"node": "pve-a"}, {"node": "pve-b"}]}}

    def backups(self):
        return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T08:00:01+00:00",
                "data": {"schema": "proxmox-backups.v1", "guests": [], "count": 0}}

    def node_disks(self, node):
        if node in self.fail_disks_on:
            raise RuntimeError("Proxmox refused the read (HTTP 403).")
        return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T08:00:02+00:00",
                "data": {"node": node, "count": 0, "disks": []}}

    def close(self):
        self.closed = True


def test_health_returns_backups_and_every_nodes_disks_and_closes_its_client(monkeypatch):
    provider = ProviderHealth()
    client = client_for(Settings(_env_file=None), monkeypatch, lambda settings: provider)
    assert client.get("/internal/v1/proxmox/health").status_code == 401
    response = client.get("/internal/v1/proxmox/health", headers={"x-internal-token": "test-internal-token"})
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "proxmox-health.v1"
    assert body["backups"]["data"]["schema"] == "proxmox-backups.v1"
    assert [entry["node"] for entry in body["disks"]] == ["pve-a", "pve-b"]
    assert all(entry["result"]["ok"] for entry in body["disks"])
    assert provider.closed


def test_one_nodes_unreadable_disks_are_named_while_the_rest_still_show(monkeypatch):
    client = client_for(Settings(_env_file=None), monkeypatch, lambda settings: ProviderHealth(fail_disks_on=("pve-b",)))
    body = client.get("/internal/v1/proxmox/health", headers={"x-internal-token": "test-internal-token"}).json()
    disks = {entry["node"]: entry["result"] for entry in body["disks"]}
    assert disks["pve-a"]["ok"] is True
    assert disks["pve-b"] == {"ok": False, "status": "error", "provider": "proxmox", "error_code": "provider_error",
                              "error": "Proxmox refused the read (HTTP 403)."}


def test_health_reports_unconfigured_provider_without_fabricating_data(monkeypatch):
    client = client_for(Settings(_env_file=None), monkeypatch, lambda settings: None)
    response = client.get("/internal/v1/proxmox/health", headers={"x-internal-token": "test-internal-token"})
    assert response.status_code == 503 and response.json() == {"error": "proxmox_not_configured"}
