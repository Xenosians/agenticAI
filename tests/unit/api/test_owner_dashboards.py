"""The Proxmox and Tailscale dashboards read the signed-in account's own link (owner decision, 17 November).

Phoenix authenticates the account and passes it as ``owner``; the AI reads
that account's Proxmox or Tailscale (its own link, else the machine default).
An account without either is told to connect the provider, not that the
server lacks a setting.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions, proxmox_actions, tailnet_overview
from config import Settings
from services.extensions.links import OwnerLinks
from services.extensions.store import ExtensionStore

TOKEN = {"x-internal-token": "test-internal-token"}
PROXMOX = {"url": "https://alice-pve.example.test:8006", "token_id": "janus@pve!dashboard",
           "token_secret": "synthetic-proxmox-secret", "verify_tls": True}


@pytest.fixture
def client(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr("services.extensions.service.extension_store", lambda: store)
    OwnerLinks(store).save("alice", "proxmox", PROXMOX)
    runtime = SimpleNamespace(settings=Settings(_env_file=None))
    monkeypatch.setattr(proxmox_actions, "require_ready_runtime", lambda request: runtime)
    monkeypatch.setattr(tailnet_overview, "require_ready_runtime", lambda request: runtime)
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: "test-internal-token"))
    seen = []

    def overview(self):
        seen.append(self.origin)
        return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "now",
                "data": {"schema": "proxmox-overview.v1"}}

    monkeypatch.setattr("services.proxmox.ProxmoxService.overview", overview)
    application = FastAPI()
    application.include_router(proxmox_actions.router)
    application.include_router(tailnet_overview.router)
    test_client = TestClient(application)
    test_client.seen = seen
    return test_client


def test_the_overview_reads_the_signed_in_accounts_own_proxmox(client):
    response = client.get("/internal/v1/proxmox/overview", params={"owner": "alice"}, headers=TOKEN)
    assert response.status_code == 200
    assert client.seen == ["https://alice-pve.example.test:8006"]


def test_an_account_without_a_link_or_machine_default_is_told_to_connect_proxmox(client):
    response = client.get("/internal/v1/proxmox/overview", params={"owner": "bob"}, headers=TOKEN)
    assert response.status_code == 503
    assert response.json() == {"error": "proxmox_not_configured"}
    assert client.seen == []


def test_the_health_read_uses_the_same_accounts_link(client, monkeypatch):
    origins = []

    def nodes(self):
        origins.append(self.origin)
        return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "now",
                "data": {"count": 0, "truncated": False, "nodes": []}}

    monkeypatch.setattr("services.proxmox.ProxmoxService.nodes", nodes)
    monkeypatch.setattr("services.proxmox.ProxmoxService.backups", lambda self: {
        "ok": True, "status": "success", "provider": "proxmox", "observed_at": "now", "data": {"guests": []}})
    response = client.get("/internal/v1/proxmox/health", params={"owner": "alice"}, headers=TOKEN)
    assert response.status_code == 200
    assert origins == ["https://alice-pve.example.test:8006"]


def test_an_account_without_a_tailnet_is_told_to_connect_it(client):
    response = client.get("/internal/v1/tailnet/overview", params={"owner": "bob"}, headers=TOKEN)
    assert response.status_code == 503
    assert response.json() == {"error": "tailnet_not_configured"}


@pytest.mark.parametrize("owner", ["", "x" * 201])
def test_an_invalid_owner_is_refused(client, owner):
    response = client.get("/internal/v1/proxmox/overview", params={"owner": owner}, headers=TOKEN)
    assert response.status_code == 422
