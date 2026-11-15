"""Tailnet reads: which devices are on the Tailscale network and how it resolves names.

The service signs in with the deployment's read-only OAuth client (devices and
DNS scopes only), keeps the one-hour token until shortly before it expires, and
normalizes every answer. Device addresses, keys and user details never leave it.
"""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from services.tailnet import (
    MAX_DEVICES,
    TailnetConfig,
    TailnetService,
    build_tailnet_service,
    tailnet_integration_status,
)

NOW = datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc)
DEVICE = {
    "id": "123", "nodeId": "nABC", "hostname": "janus-router-lab", "name": "janus-router-lab.tail1234.ts.net",
    "os": "linux", "clientVersion": "1.104.4-t9382131983", "updateAvailable": False,
    "connectedToControl": True, "lastSeen": "2026-11-02T02:59:00Z", "expires": "2027-05-01T00:00:00Z",
    "keyExpiryDisabled": False, "authorized": True, "isExternal": False, "tags": ["tag:janus-lab"],
    "addresses": ["100.64.0.7", "fd7a:115c:a1e0::7"], "nodeKey": "nodekey:secret", "machineKey": "mkey:secret",
    "user": "tagged-devices", "advertisedRoutes": [], "enabledRoutes": [],
    # Only with ?fields=all:
    "blocksIncomingConnections": False, "tailnetLockError": "", "sshEnabled": False,
    "clientConnectivity": {
        "endpoints": ["203.0.113.9:41641", "192.168.100.20:41641"], "mappingVariesByDestIP": False,
        "latency": {"Singapore": {"preferred": True, "latencyMs": 27.24}, "Tokyo": {"latencyMs": 90.1}},
        "clientSupports": {"hairPinning": None, "ipv6": True, "pcp": False, "pmp": False, "udp": True, "upnp": False},
    },
    "postureStatus": {"lastSeen": "2026-11-02T02:59:00Z"}, "distro": {"name": "debian", "version": "13"},
}
HEALTHY = {"blocks_incoming": False, "routes_advertised": 0, "routes_waiting_for_approval": 0,
           "exit_node_offered": False, "tailnet_lock_error": False,
           "connectivity": {"udp": True, "hard_nat": False, "port_mapping": False,
                            "nearest_relay": "Singapore", "relay_latency_ms": 27.2},
           "issues": []}
DNS = {
    "nameservers": [{"address": "1.1.1.1"}, {"address": "1.0.0.1"}],
    "splitDNS": {"corp.example": [{"address": "10.0.0.53"}]},
    "searchPaths": ["corp.example"],
    "preferences": {"overrideLocalDNS": True, "magicDNS": True},
}


def service(routes, *, now=NOW, scopes="devices:core:read dns:read"):
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, request.headers.get("authorization"),
                      dict(request.url.params)))
        if request.url.path == "/api/v2/oauth/token":
            form = dict(item.split("=", 1) for item in request.content.decode().split("&"))
            assert form["grant_type"] == "client_credentials"
            return httpx.Response(200, json={"access_token": f"token-{len(calls)}", "expires_in": 3600,
                                             "scope": scopes, "token_type": "Bearer"})
        status, body = routes[request.url.path]
        return httpx.Response(status, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    clock = {"now": now}
    tailnet = TailnetService(TailnetConfig("client-id", "client-secret"), client, clock=lambda: clock["now"])
    return tailnet, calls, clock


def test_devices_report_identity_reachability_and_key_state_without_addresses_or_keys():
    tailnet, calls, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": [DEVICE]})})

    result = tailnet.devices()

    assert result["ok"] is True and result["status"] == "success" and result["provider"] == "tailscale"
    assert result["data"] == {"count": 1, "truncated": False,
                              "summary": {"online": 1, "offline": 0, "with_issues": 0}, "devices": [{
        "name": "janus-router-lab", "dns_name": "janus-router-lab.tail1234.ts.net", "os": "linux",
        "client_version": "1.104.4", "update_available": False, "online": True,
        "last_seen": "2026-11-02T02:59:00Z", "key_expires": "2027-05-01T00:00:00Z",
        "key_expiry_disabled": False, "key_expires_in_days": 179, "authorized": True,
        "shared_from_another_tailnet": False, "tags": ["tag:janus-lab"], **HEALTHY}]}
    text = json.dumps(result)
    for private in ("100.64.0.7", "fd7a:", "nodekey", "mkey", "client-secret", "token-", "203.0.113.9",
                    "192.168.100.20", "41641", "debian", "postureStatus"):
        assert private not in text
    assert [call[1] for call in calls] == ["/api/v2/oauth/token", "/api/v2/tailnet/-/devices"]
    assert calls[1][2] == "Bearer token-1"
    # The connectivity facts come only with every field; the token's read scopes allow it.
    assert calls[1][3] == {"fields": "all"}


def test_each_condition_an_operator_should_look_at_is_named():
    soon = {**DEVICE, "hostname": "laptop", "expires": (NOW + timedelta(days=13)).isoformat().replace("+00:00", "Z")}
    expired = {**DEVICE, "hostname": "old-phone", "expires": "2026-10-01T00:00:00Z", "connectedToControl": False}
    pending = {**DEVICE, "hostname": "new-server", "authorized": False, "updateAvailable": True,
               "advertisedRoutes": ["192.168.100.0/24", "0.0.0.0/0", "::/0"], "enabledRoutes": ["192.168.100.0/24"]}
    shielded = {**DEVICE, "hostname": "kiosk", "blocksIncomingConnections": True,
                "tailnetLockError": "this node is locked out by tailnet lock",
                "clientConnectivity": {**DEVICE["clientConnectivity"], "mappingVariesByDestIP": True,
                                       "clientSupports": {**DEVICE["clientConnectivity"]["clientSupports"],
                                                          "udp": False}}}
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": [soon, expired, pending, shielded]})})

    data = tailnet.devices()["data"]
    rows = {row["name"]: row for row in data["devices"]}

    assert rows["laptop"]["issues"] == ["key_expiring"]
    assert rows["old-phone"]["issues"] == ["key_expired"] and rows["old-phone"]["online"] is False
    assert rows["new-server"]["issues"] == ["waiting_for_approval", "routes_waiting_for_approval", "update_available"]
    assert (rows["new-server"]["routes_advertised"], rows["new-server"]["routes_waiting_for_approval"],
            rows["new-server"]["exit_node_offered"]) == (3, 2, True)
    assert rows["kiosk"]["issues"] == ["tailnet_lock_error", "blocks_incoming", "udp_blocked"]
    assert rows["kiosk"]["connectivity"]["hard_nat"] is True and rows["kiosk"]["tailnet_lock_error"] is True
    assert "locked out" not in json.dumps(data)
    assert data["summary"] == {"online": 3, "offline": 1, "with_issues": 4}


def test_a_device_without_connectivity_facts_reports_them_as_unknown():
    asleep = {key: value for key, value in DEVICE.items()
              if key not in ("clientConnectivity", "blocksIncomingConnections", "tailnetLockError")}
    quiet = {**DEVICE, "hostname": "phone", "connectedToControl": False,
             "clientConnectivity": {"endpoints": [], "latency": {}, "mappingVariesByDestIP": False,
                                    "clientSupports": {"udp": True}}}
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": [asleep, quiet]})})

    rows = {row["name"]: row for row in tailnet.devices()["data"]["devices"]}

    assert rows["janus-router-lab"]["connectivity"] is None
    assert rows["janus-router-lab"]["blocks_incoming"] is None and rows["janus-router-lab"]["issues"] == []
    assert rows["phone"]["connectivity"] == {"udp": True, "hard_nat": False, "port_mapping": None,
                                             "nearest_relay": None, "relay_latency_ms": None}


def test_a_key_that_expired_hours_ago_is_expired_not_expiring():
    lapsed = {**DEVICE, "hostname": "tablet", "expires": (NOW - timedelta(hours=2)).isoformat().replace("+00:00", "Z")}
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": [lapsed]})})

    row = tailnet.devices()["data"]["devices"][0]

    assert row["key_expires_in_days"] == 0 and row["issues"] == ["key_expired"]


@pytest.mark.parametrize("change", [
    {"clientConnectivity": "nope"}, {"clientConnectivity": {"latency": []}},
    {"clientConnectivity": {"latency": {"Singapore": {"preferred": True, "latencyMs": "fast"}}}},
    {"clientConnectivity": {"clientSupports": {"udp": "yes"}}}, {"advertisedRoutes": "all"},
    {"enabledRoutes": [7]}, {"blocksIncomingConnections": "no"}, {"tailnetLockError": 3},
    {"clientConnectivity": {"clientSupports": []}}, {"tags": {}},
])
def test_malformed_health_facts_are_refused(change):
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": [{**DEVICE, **change}]})})
    with pytest.raises(RuntimeError, match="Tailscale"):
        tailnet.devices()


@pytest.mark.parametrize("change", [
    {"preferences": []}, {"splitDNS": []}, {"nameservers": {}}, {"searchPaths": ""},
])
def test_malformed_dns_settings_are_refused_not_read_as_empty(change):
    tailnet, _, _ = service({"/api/v2/tailnet/-/dns/configuration": (200, {**DNS, **change})})
    with pytest.raises(RuntimeError, match="Tailscale"):
        tailnet.dns_settings()


def test_a_key_that_has_expired_or_is_close_to_expiring_is_stated():
    soon = {**DEVICE, "hostname": "laptop", "expires": (NOW + timedelta(days=3)).isoformat().replace("+00:00", "Z")}
    gone = {**DEVICE, "hostname": "old-phone", "expires": "2026-10-01T00:00:00Z", "connectedToControl": False}
    never = {**DEVICE, "hostname": "server", "keyExpiryDisabled": True, "expires": "0001-01-01T00:00:00Z"}
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": [soon, gone, never]})})

    rows = {row["name"]: row for row in tailnet.devices()["data"]["devices"]}

    assert rows["laptop"]["key_expires_in_days"] == 3
    assert rows["old-phone"]["key_expires_in_days"] == -32 and rows["old-phone"]["online"] is False
    assert rows["server"]["key_expires"] is None and rows["server"]["key_expires_in_days"] is None


def test_the_token_is_reused_until_shortly_before_it_expires():
    routes = {"/api/v2/tailnet/-/devices": (200, {"devices": []}), "/api/v2/tailnet/-/dns/configuration": (200, DNS)}
    tailnet, calls, clock = service(routes)

    tailnet.devices()
    tailnet.dns_settings()
    clock["now"] = NOW + timedelta(minutes=58)
    tailnet.devices()

    assert [call[1] for call in calls].count("/api/v2/oauth/token") == 2
    assert calls[-1][2] == "Bearer token-4"


def test_dns_settings_say_whether_names_resolve_through_encrypted_dns():
    tailnet, _, _ = service({"/api/v2/tailnet/-/dns/configuration": (200, DNS)})

    result = tailnet.dns_settings()

    assert result["data"] == {
        "magic_dns": True, "override_local_dns": True,
        "nameservers": [{"address": "1.1.1.1", "encrypted": True}, {"address": "1.0.0.1", "encrypted": True}],
        "split_dns": [{"domain": "corp.example", "nameservers": ["10.0.0.53"]}],
        "search_domains": ["corp.example"],
    }


def test_a_tailnet_on_local_resolvers_is_reported_as_such():
    tailnet, _, _ = service({"/api/v2/tailnet/-/dns/configuration": (200, {
        "nameservers": [{"address": "192.168.1.1"}], "preferences": {"magicDNS": True}})})

    data = tailnet.dns_settings()["data"]

    assert data["override_local_dns"] is False
    assert data["nameservers"] == [{"address": "192.168.1.1", "encrypted": False}]
    assert data["split_dns"] == [] and data["search_domains"] == []


@pytest.mark.parametrize("body", [
    {"devices": "nope"}, {"devices": [{"hostname": 7}]}, [], {"devices": [{"name": "x"}]},
])
def test_malformed_device_answers_are_refused(body):
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, body)})
    with pytest.raises(RuntimeError, match="Tailscale"):
        tailnet.devices()


def test_long_device_lists_are_cut_and_marked():
    many = [{**DEVICE, "hostname": f"host-{index}", "name": f"host-{index}.tail1234.ts.net"}
            for index in range(MAX_DEVICES + 5)]
    tailnet, _, _ = service({"/api/v2/tailnet/-/devices": (200, {"devices": many})})

    data = tailnet.devices()["data"]

    assert data["count"] == MAX_DEVICES + 5 and data["truncated"] is True and len(data["devices"]) == MAX_DEVICES


def test_a_refused_credential_or_read_raises_without_echoing_the_answer():
    def refuse(request):
        return httpx.Response(401, json={"message": "invalid client secret client-secret"})

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    tailnet = TailnetService(TailnetConfig("client-id", "client-secret"), client, clock=lambda: NOW)
    with pytest.raises(RuntimeError) as caught:
        tailnet.devices()
    assert "client-secret" not in str(caught.value)


def test_the_service_exists_only_when_the_read_credential_is_configured():
    class Settings:
        tailscale_client_id = None
        tailscale_client_secret = None
        tailscale_timeout_seconds = 10.0

    assert build_tailnet_service(Settings()) is None
    assert tailnet_integration_status(Settings()) == {"configured": False}
    Settings.tailscale_client_id, Settings.tailscale_client_secret = "id", "secret"
    assert isinstance(build_tailnet_service(Settings()), TailnetService)
    assert tailnet_integration_status(Settings()) == {"configured": True}
