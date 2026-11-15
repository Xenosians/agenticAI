"""Phoenix reads the tailnet overview: device health plus DNS settings, both read-only."""
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from api import tailnet_overview
from config import Settings

TOKEN = {"x-internal-token": "test-internal-token"}


class Tailnet:
    closed = False

    def devices(self):
        return {"ok": True, "status": "success", "provider": "tailscale", "observed_at": "2026-11-16T05:00:00+00:00",
                "data": {"count": 1, "truncated": False, "summary": {"online": 1, "offline": 0, "with_issues": 0},
                         "devices": [{"name": "hummingbird", "issues": []}]}}

    def dns_settings(self):
        return {"ok": True, "status": "success", "provider": "tailscale", "observed_at": "2026-11-16T05:00:01+00:00",
                "data": {"magic_dns": True, "override_local_dns": False, "nameservers": [], "split_dns": [],
                         "search_domains": []}}

    def close(self):
        Tailnet.closed = True


def client_for(monkeypatch, factory):
    monkeypatch.setattr(tailnet_overview, "require_ready_runtime",
                        lambda request: SimpleNamespace(settings=Settings(_env_file=None)))
    monkeypatch.setattr(extensions, "get_settings", lambda: SimpleNamespace(
        require_internal_job_token=lambda: "test-internal-token"))
    monkeypatch.setattr("services.tailnet.build_tailnet_service", factory)
    application = FastAPI()
    application.include_router(tailnet_overview.router)
    return TestClient(application)


def test_the_overview_requires_internal_authentication(monkeypatch):
    client = client_for(monkeypatch, lambda settings: Tailnet())
    assert client.get("/internal/v1/tailnet/overview").status_code == 401


def test_an_unconfigured_tailnet_is_reported_without_fabricated_devices(monkeypatch):
    client = client_for(monkeypatch, lambda settings: None)
    response = client.get("/internal/v1/tailnet/overview", headers=TOKEN)
    assert response.status_code == 503
    assert response.json() == {"error": "tailnet_not_configured"}


def test_the_overview_returns_devices_and_dns_in_one_answer_and_closes_the_client(monkeypatch):
    Tailnet.closed = False
    client = client_for(monkeypatch, lambda settings: Tailnet())
    response = client.get("/internal/v1/tailnet/overview", headers=TOKEN)
    assert response.status_code == 200
    body = response.json()
    assert (body["ok"], body["provider"], body["observed_at"]) == (True, "tailscale", "2026-11-16T05:00:00+00:00")
    assert body["data"]["schema"] == "tailnet-overview.v1"
    assert body["data"]["devices"]["summary"] == {"online": 1, "offline": 0, "with_issues": 0}
    assert body["data"]["dns"] == {"magic_dns": True, "override_local_dns": False, "nameservers": [],
                                   "split_dns": [], "search_domains": []}
    assert body["data"]["dns_error"] is None
    assert Tailnet.closed is True


def test_devices_still_show_when_only_the_dns_read_fails(monkeypatch):
    class NoDns(Tailnet):
        def dns_settings(self):
            raise RuntimeError("Tailscale refused the read (HTTP 403).")

    client = client_for(monkeypatch, lambda settings: NoDns())
    body = client.get("/internal/v1/tailnet/overview", headers=TOKEN).json()
    assert body["ok"] is True and body["data"]["devices"]["count"] == 1
    assert body["data"]["dns"] is None
    assert body["data"]["dns_error"] == {"reason": "provider_error",
                                         "message": "Tailscale refused the read (HTTP 403)."}


def test_a_failed_device_read_says_why_without_echoing_the_provider(monkeypatch):
    request = httpx.Request("GET", "https://api.tailscale.com/api/v2/tailnet/-/devices")
    cases = [
        (httpx.ConnectError("dns failure for api.tailscale.com", request=request), "provider_unreachable"),
        (httpx.ReadTimeout("slow", request=request), "provider_timeout"),
        (RuntimeError("Tailscale refused the read (HTTP 401)."), "provider_error"),
    ]
    for error, reason in cases:
        class Failing(Tailnet):
            def devices(self, error=error):
                raise error

        client = client_for(monkeypatch, lambda settings: Failing())
        response = client.get("/internal/v1/tailnet/overview", headers=TOKEN)
        assert response.status_code == 502
        body = response.json()
        assert (body["error"], body["reason"]) == ("tailnet_read_failed", reason)
        assert isinstance(body["message"], str) and body["message"]
        assert "api.tailscale.com" not in body["message"]
