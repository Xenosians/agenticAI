"""The integration status endpoint reports setup readiness for every provider family (FR-081)."""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import app as api
from config import Settings
from services.integration_readiness import PROVIDER_FAMILIES


def client_for(settings, monkeypatch):
    monkeypatch.setattr(api, "require_ready_runtime", lambda request: SimpleNamespace(settings=settings))
    application = FastAPI()
    application.add_api_route("/v1/integrations", api.integration_status, methods=["GET"])
    return TestClient(application)


def test_every_provider_family_is_listed_with_its_setup_state(monkeypatch):
    settings = Settings(_env_file=None, ticketing_backend="mock",
                        openwrt_base_url="https://router.example.test", openwrt_username="reader",
                        openwrt_password="fixture-password-value",
                        proxmox_base_url="https://pve.example.test:8006",
                        proxmox_api_token_id="reader@pam!janus",
                        proxmox_api_token_secret="fixture-proxmox-token")
    body = client_for(settings, monkeypatch).get("/v1/integrations").json()
    rows = {row["id"]: row for row in body["integrations"]}
    assert set(rows) == {family.id for family in PROVIDER_FAMILIES}
    assert rows["ticketing"]["state"] == "simulated"
    assert rows["ticketing"]["configured"] is False
    assert rows["directory"]["missing"] == ["DIRECTORY_BACKEND"]
    # The existing OpenWrt status fields stay for current readers.
    assert rows["openwrt"]["configured"] is True
    assert rows["openwrt"]["host"] == "router.example.test"
    assert rows["openwrt"]["verify_tls"] is True
    assert rows["proxmox"]["state"] == "configured"
    assert rows["proxmox"]["mode"] == "read_only"
    assert "fixture-password-value" not in repr(body)
    assert "fixture-proxmox-token" not in repr(body)
