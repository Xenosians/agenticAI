"""A provider read ends with its observed data or a typed failure that says why, never an unconfirmed outcome."""
import json

import httpx
import pytest

from services.lua_adapters.runtime import LuaAdapterError
from tools.provider_reads import not_configured, provider_read

REQUEST = httpx.Request("GET", "https://router.example.test/api")


def failure_of(error: Exception) -> dict:
    def read():
        raise error

    return provider_read("openwrt", "The OpenWrt router", read)


def test_observed_data_passes_through_unchanged():
    observed = {"ok": True, "status": "success", "provider": "openwrt", "data": {"count": 1}}
    assert provider_read("openwrt", "The OpenWrt router", lambda: observed) is observed


def test_a_missing_setup_names_the_settings_and_is_not_an_unknown_outcome():
    result = not_configured("proxmox", "Proxmox VE", ("PROXMOX_BASE_URL", "PROXMOX_API_TOKEN_ID"))
    assert result == {
        "ok": False, "status": "error", "provider": "proxmox", "error_code": "not_configured",
        "error": "Proxmox VE is not set up on this Janus host. Missing settings: "
                 "PROXMOX_BASE_URL, PROXMOX_API_TOKEN_ID.",
    }
    assert not_configured("proxmox", "Proxmox VE")["error"] == "Proxmox VE is not set up on this Janus host."


@pytest.mark.parametrize(("error", "code", "message"), [
    (httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
     "provider_unreachable", "The OpenWrt router could not be reached, or it closed the connection before answering."),
    (httpx.ConnectError("[Errno 111] Connection refused"),
     "provider_unreachable", "The OpenWrt router could not be reached, or it closed the connection before answering."),
    (httpx.ReadTimeout("timed out"), "provider_timeout", "The OpenWrt router did not answer in time."),
    (httpx.HTTPStatusError("Client error '403 Forbidden' for url 'https://router.example.test/api'",
                           request=REQUEST, response=httpx.Response(403, request=REQUEST)),
     "provider_refused", "The OpenWrt router refused the read (HTTP 403)."),
    (json.JSONDecodeError("Expecting value", "<html>", 0), "provider_error", "The OpenWrt router returned an invalid answer."),
    (RuntimeError("OpenWrt rejected the read or returned an invalid RPC response."),
     "provider_error", "OpenWrt rejected the read or returned an invalid RPC response."),
    (ValueError("Invalid logical OpenWrt interface name."), "invalid_argument", "Invalid logical OpenWrt interface name."),
    (LuaAdapterError("lua_adapter_time_limit"), "lua_adapter_time_limit", "The Lua adapter exceeded its time limit."),
])
def test_each_known_failure_is_a_typed_read_failure(error, code, message):
    assert failure_of(error) == {"ok": False, "status": "error", "provider": "openwrt",
                                 "error_code": code, "error": message}


def test_transport_failures_never_echo_the_provider_url():
    result = failure_of(httpx.HTTPStatusError("Client error for url 'https://secret-host.example.test/x'",
                                              request=REQUEST, response=httpx.Response(401, request=REQUEST)))
    assert "example.test" not in result["error"]


def test_a_connection_the_provider_closed_is_retried_once():
    calls = []

    def read():
        calls.append(1)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
        return {"ok": True, "status": "success", "provider": "openwrt", "data": {}}

    assert provider_read("openwrt", "The OpenWrt router", read)["ok"] is True
    assert len(calls) == 2


@pytest.mark.parametrize("error", [httpx.ReadTimeout("timed out"), RuntimeError("Proxmox refused the read (HTTP 403).")])
def test_a_timeout_or_a_refusal_is_not_retried(error):
    calls = []

    def read():
        calls.append(1)
        raise error

    assert provider_read("proxmox", "Proxmox VE", read)["ok"] is False
    assert len(calls) == 1


def test_an_unexpected_error_still_surfaces():
    with pytest.raises(KeyError):
        failure_of(KeyError("bug"))


def test_an_unexpected_error_on_the_second_read_still_surfaces():
    calls = []

    def read():
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("[Errno 111] Connection refused")
        raise KeyError("bug")

    with pytest.raises(KeyError):
        provider_read("openwrt", "The OpenWrt router", read)
