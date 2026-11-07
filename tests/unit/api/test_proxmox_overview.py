from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import app as api
from api import extensions
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
    monkeypatch.setattr(api, "require_ready_runtime", lambda request: SimpleNamespace(settings=settings))
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: "test-internal-token"))
    monkeypatch.setattr("services.proxmox.build_proxmox_service", service_factory)
    application = FastAPI()
    application.add_api_route("/internal/v1/proxmox/overview", api.proxmox_overview, methods=["GET"])
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
