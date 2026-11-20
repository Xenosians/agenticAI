"""Tailnet reads survive a reconnect: the control API is read again, boundedly.

The OAuth grant and every device/DNS read have no effect to repeat, so a
connection that fails while Tailscale reconnects is tried again. A refused
answer (a real status) is not retried.
"""
from datetime import datetime, timezone

import httpx
import pytest

from services.tailnet import TailnetConfig, TailnetService

NOW = datetime(2026, 11, 20, 3, 0, tzinfo=timezone.utc)


def service(handler, *, attempts=3, wait=0.0):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return TailnetService(TailnetConfig("client-id", "client-secret", 5, attempts, wait),
                          client, clock=lambda: NOW), client


def grant(request):
    return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})


def devices(request):
    return httpx.Response(200, json={"devices": []})


def test_a_device_read_is_retried_after_the_connection_drops_then_succeeds():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/v2/oauth/token":
            return grant(request)
        if len(calls) == 2:
            raise httpx.ConnectError("tailnet reconnecting", request=request)
        return devices(request)

    tailnet, client = service(handler)
    try:
        assert tailnet.devices()["data"]["count"] == 0
    finally:
        client.close()
    assert calls == ["/api/v2/oauth/token", "/api/v2/tailnet/-/devices", "/api/v2/tailnet/-/devices"]


def test_a_read_stops_after_its_bounded_attempts():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        raise httpx.ConnectError("offline", request=request)

    tailnet, client = service(handler, attempts=2)
    try:
        with pytest.raises(httpx.ConnectError):
            tailnet.devices()
    finally:
        client.close()
    assert calls == ["/api/v2/oauth/token", "/api/v2/oauth/token"]


def test_a_refused_answer_is_reported_once_and_never_retried():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/api/v2/oauth/token":
            return grant(request)
        return httpx.Response(403, json={"message": "denied"}, request=request)

    tailnet, client = service(handler)
    try:
        with pytest.raises(Exception) as refused:
            tailnet.devices()
    finally:
        client.close()
    assert "HTTP 403" in str(refused.value)
    assert calls == ["/api/v2/oauth/token", "/api/v2/tailnet/-/devices"]
