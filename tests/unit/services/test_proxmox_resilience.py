"""Proxmox reachability: the tailnet name first, a configured LAN origin next, bounded retries.

A read has no effect to repeat, so a connection that never opened is tried
again, and so is a connection the server dropped before answering. A change is
sent once: it reaches a second origin only while the connection never opened,
so a request that may have been processed is never repeated.
"""
import httpx
import pytest

from services.proxmox import (CONNECT_TIMEOUT_SECONDS, ProxmoxConfig, ProxmoxOrigin, ProxmoxService,
                              build_proxmox_service)

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"
TAILNET = "https://hummingbird.example.ts.net"
LAN = "https://192.0.2.56:8006"


def config(*, alt=(), attempts=3, wait=0.0, server_name=None):
    return ProxmoxConfig(TAILNET, TOKEN_ID, TOKEN_SECRET, True, None, server_name, 5,
                         tuple(ProxmoxOrigin(url, name) for url, name in alt), attempts, wait)


def service(handler, **options):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(config(**options), client), client


def answer(request):
    return httpx.Response(200, json={"data": [{"id": "cluster", "type": "cluster", "name": "lab"}]},
                          request=request)


def test_a_read_that_cannot_reach_the_tailnet_name_is_answered_by_the_lan_origin():
    seen = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "hummingbird.example.ts.net":
            raise httpx.ConnectError("tailnet is down", request=request)
        return answer(request)

    proxmox, client = service(handler, alt=[(LAN, "hummingbird.local.pve")])
    try:
        assert proxmox.cluster_status()["data"]["members"][0]["name"] == "lab"
    finally:
        client.close()
    assert seen == ["hummingbird.example.ts.net", "192.0.2.56"]
    # The LAN origin is verified against the host's certificate name, never against the bare address.
    assert proxmox.origins[1].server_name == "hummingbird.local.pve"


def test_a_read_uses_the_only_configured_origin_when_no_fallback_is_set():
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return answer(request)

    proxmox, client = service(handler)
    try:
        proxmox.cluster_status()
    finally:
        client.close()
    assert seen == ["hummingbird.example.ts.net"]


def test_a_read_timeout_is_not_retried_because_the_server_may_still_be_working():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        raise httpx.ReadTimeout("slow", request=request)

    proxmox, client = service(handler, alt=[(LAN, None)])
    try:
        with pytest.raises(httpx.ReadTimeout):
            proxmox.cluster_status()
    finally:
        client.close()
    assert calls == ["hummingbird.example.ts.net"]


def test_a_read_stops_after_its_bounded_attempts_and_raises_the_last_failure():
    calls = []

    def handler(request):
        calls.append(request.url.host)
        raise httpx.ConnectError("unreachable", request=request)

    proxmox, client = service(handler, alt=[(LAN, None)], attempts=3)
    try:
        with pytest.raises(httpx.ConnectError):
            proxmox.cluster_status()
    finally:
        client.close()
    # The attempts rotate through both origins, so neither is left unread.
    assert calls == ["hummingbird.example.ts.net", "192.0.2.56", "hummingbird.example.ts.net"]


def test_a_change_moves_to_the_lan_origin_only_when_the_connection_never_opened():
    sent = []

    def handler(request):
        sent.append(request.url.host)
        if request.url.host == "hummingbird.example.ts.net":
            raise httpx.ConnectError("tailnet is down", request=request)
        return httpx.Response(200, json={"data": "UPID:pve-a:1:2:3:qmstart:100:root@pam:"}, request=request)

    proxmox, client = service(handler, alt=[(LAN, None)])
    try:
        response = proxmox._post("/nodes/pve-a/qemu/100/status/start", {})
    finally:
        client.close()
    assert response.status_code == 200
    assert sent == ["hummingbird.example.ts.net", "192.0.2.56"]


def test_a_change_is_never_repeated_after_the_request_may_have_been_sent():
    sent = []

    def handler(request):
        sent.append(request.url.host)
        raise httpx.ReadTimeout("no answer", request=request)

    proxmox, client = service(handler, alt=[(LAN, None)])
    try:
        with pytest.raises(httpx.ReadTimeout):
            proxmox._post("/nodes/pve-a/qemu/100/status/start", {})
    finally:
        client.close()
    assert sent == ["hummingbird.example.ts.net"]


def test_a_dropped_keepalive_connection_is_read_again_on_a_fresh_one():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("server closed the idle connection", request=request)
        return answer(request)

    proxmox, client = service(handler)
    try:
        assert proxmox.cluster_status()["data"]["members"][0]["name"] == "lab"
    finally:
        client.close()
    assert calls == ["/api2/json/cluster/status", "/api2/json/cluster/status"]


def test_fallback_origins_are_normalized_deduplicated_and_refused_when_unsafe():
    proxmox, client = service(lambda request: answer(request), alt=[(LAN + "/", "pve.example.test"),
                                                                    (LAN, None)])
    try:
        assert [origin.url for origin in proxmox.origins] == [TAILNET, LAN]
        assert proxmox.origins[1].server_name == "pve.example.test"
    finally:
        client.close()
    for unsafe in ("http://192.0.2.56:8006", "https://127.0.0.1:8006",
                   "https://192.0.2.56:8006/path"):
        with pytest.raises(ValueError):
            service(answer, alt=[(unsafe, None)])
    with pytest.raises(ValueError):
        service(answer, alt=[(f"https://192.0.2.{index}:8006", None) for index in range(1, 6)])


def test_the_connection_attempt_is_bounded_so_a_read_does_not_wait_for_a_long_tcp_timeout():
    assert CONNECT_TIMEOUT_SECONDS <= 5


def test_settings_build_the_fallback_list_and_the_read_budget():
    from config.settings import Settings

    settings = Settings(_env_file=None, proxmox_base_url=TAILNET, proxmox_api_token_id=TOKEN_ID,
                        proxmox_api_token_secret=TOKEN_SECRET, proxmox_server_name="pve.example.test",
                        proxmox_alt_urls=f"{LAN}|hummingbird.local.pve, https://192.0.2.57:8006",
                        proxmox_read_attempts=2, proxmox_retry_wait_seconds=0)
    proxmox = build_proxmox_service(settings)
    assert proxmox is not None
    try:
        assert [(origin.url, origin.server_name) for origin in proxmox.origins] == [
            (TAILNET, "pve.example.test"), (LAN, "hummingbird.local.pve"),
            ("https://192.0.2.57:8006", None)]
        assert proxmox.config.read_attempts == 2
        assert proxmox.config.retry_wait_seconds == 0
    finally:
        proxmox.close()
