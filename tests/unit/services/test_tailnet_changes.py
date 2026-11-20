"""Governed tailnet device, route, DNS and key changes (owner decision 20 Nov 2026, ~14:10 WIB, item 1).

Every test drives the real service and MCP tools against a fake Tailscale control API that behaves
like the live one (read 20 Nov 2026 with Janus's new client): devices carry a 17-character
``nodeId`` and a 16-digit legacy ``id``; ``POST /device/{id}/routes`` replaces the enabled routes
and answers with both lists; ``DELETE /device/{id}`` answers 200 even for an unknown id on some
paths, so removal is confirmed by listing the devices again; DNS writes answer with the new value.

The pattern is the grant change's: the trusted policy reads the device before approval and freezes
its identity (id, name, tags) plus the value being replaced; the approved tool re-reads, refuses a
device that moved, sends exactly one request, never retries a lost answer and reads the result back.
"""
from __future__ import annotations

import json

import httpx
import pytest

from services.tailnet import TailnetConfig, TailnetService
from services.tailnet_changes import (
    DeviceChange,
    TailnetChangeRefused,
    apply_device_change,
    apply_dns_change,
    apply_key_revoke,
    prepare_device_change,
    prepare_dns_change,
    prepare_key_revoke,
)

HUMMINGBIRD = {"id": "1234567890123456", "nodeId": "nHBXXXXXXXXXCNTRL", "hostname": "hummingbird",
               "name": "hummingbird.tail1234.ts.net", "tags": None, "authorized": True,
               "keyExpiryDisabled": True, "advertisedRoutes": ["10.77.0.0/24"], "enabledRoutes": []}
LAB = {"id": "2234567890123456", "nodeId": "nLABXXXXXXXCNTRL", "hostname": "janus-router-lab",
       "name": "janus-router-lab.tail1234.ts.net", "tags": ["tag:janus-lab"], "authorized": True,
       "keyExpiryDisabled": True, "advertisedRoutes": [], "enabledRoutes": []}
PHONE = {"id": "3234567890123456", "nodeId": "nPHNXXXXXXXCNTRL", "hostname": "Redmi Note 11",
         "name": "redmi-note-11.tail1234.ts.net", "tags": None, "authorized": True,
         "keyExpiryDisabled": False, "advertisedRoutes": [], "enabledRoutes": []}


class Tailnet:
    """The control API as it answered live; every write is counted."""

    def __init__(self):
        self.devices = {row["nodeId"]: json.loads(json.dumps(row)) for row in (HUMMINGBIRD, LAB, PHONE)}
        self.nameservers = ["1.1.1.1"]
        self.magic_dns = True
        self.search_paths: list[str] = []
        self.split: dict[str, list[str]] = {}
        self.keys = {"kAAAA1111CNTRL": {"id": "kAAAA1111CNTRL", "keyType": "auth", "description": "lab enrol",
                                        "capabilities": {"devices": {"create": {"tags": ["tag:janus-lab"]}}},
                                        "created": "2026-11-20T07:00:00Z", "expires": "2026-11-21T07:00:00Z"},
                     "kCLIENT22CNTRL": {"id": "kCLIENT22CNTRL", "keyType": "client", "description": "Janus ITSM",
                                        "created": "2026-11-20T07:48:00Z"}}
        self.writes: list[tuple[str, str]] = []
        self.drop_answer_of: str | None = None   # a write whose answer is lost after it applied

    def _device(self, ident: str):
        for row in self.devices.values():
            if ident in (row["nodeId"], row["id"]):
                return row
        return None

    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if path == "/api/v2/oauth/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        if method != "GET":
            self.writes.append((method, path))
            if self.drop_answer_of and path.endswith(self.drop_answer_of):
                self._apply(request)
                raise httpx.ReadTimeout("answer lost", request=request)
        return self._apply(request)

    def _apply(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        body = json.loads(request.content or b"{}") if request.content else {}
        if path == "/api/v2/tailnet/-/devices":
            return httpx.Response(200, json={"devices": list(self.devices.values())})
        if path.startswith("/api/v2/device/"):
            parts = path.split("/")
            device = self._device(parts[4])
            if device is None:
                return httpx.Response(404, json={"message": "not found"})
            action = parts[5] if len(parts) > 5 else ""
            if action == "" and method == "GET":
                return httpx.Response(200, json=device)
            if action == "" and method == "DELETE":
                del self.devices[device["nodeId"]]
                return httpx.Response(200)
            if action == "routes" and method == "GET":
                return httpx.Response(200, json={"advertisedRoutes": device["advertisedRoutes"],
                                                 "enabledRoutes": device["enabledRoutes"]})
            if action == "routes" and method == "POST":
                device["enabledRoutes"] = [r for r in body["routes"] if r in device["advertisedRoutes"]]
                return httpx.Response(200, json={"advertisedRoutes": device["advertisedRoutes"],
                                                 "enabledRoutes": device["enabledRoutes"]})
            if action == "authorized" and method == "POST":
                device["authorized"] = body["authorized"]
                return httpx.Response(200)
            if action == "tags" and method == "POST":
                device["tags"] = body["tags"] or None
                return httpx.Response(200)
            if action == "key" and method == "POST":
                device["keyExpiryDisabled"] = body["keyExpiryDisabled"]
                return httpx.Response(200)
            if action == "name" and method == "POST":
                device["name"] = body["name"] + ".tail1234.ts.net"
                return httpx.Response(200)
        if path == "/api/v2/tailnet/-/dns/nameservers":
            if method == "POST":
                self.nameservers = list(body["dns"])
            return httpx.Response(200, json={"dns": self.nameservers})
        if path == "/api/v2/tailnet/-/dns/preferences":
            if method == "POST":
                self.magic_dns = body["magicDNS"]
            return httpx.Response(200, json={"magicDNS": self.magic_dns})
        if path == "/api/v2/tailnet/-/dns/searchpaths":
            if method == "POST":
                self.search_paths = list(body["searchPaths"])
            return httpx.Response(200, json={"searchPaths": self.search_paths})
        if path == "/api/v2/tailnet/-/dns/split-dns":
            if method == "PATCH":
                for domain, servers in body.items():
                    if servers is None:
                        self.split.pop(domain, None)
                    else:
                        self.split[domain] = servers
            return httpx.Response(200, json=self.split)
        if path == "/api/v2/tailnet/-/keys":
            return httpx.Response(200, json={"keys": [{"id": key} for key in self.keys]})
        if path.startswith("/api/v2/tailnet/-/keys/"):
            key_id = path.rsplit("/", 1)[1]
            if key_id not in self.keys:
                return httpx.Response(404, json={"message": "not found"})
            if method == "DELETE":
                del self.keys[key_id]
                return httpx.Response(200)
            return httpx.Response(200, json=self.keys[key_id])
        raise AssertionError((method, path))

    def service(self) -> TailnetService:
        return TailnetService(TailnetConfig("id", "secret", read_attempts=1, retry_wait_seconds=0),
                              httpx.Client(transport=httpx.MockTransport(self.handler)))


@pytest.fixture
def tailnet():
    return Tailnet()


# ---- devices: approve routes, authorize, tags, key expiry, name, remove ----

def test_approving_an_advertised_route_freezes_the_device_and_reads_the_routes_back(tailnet):
    # Owner decision item 2: hummingbird advertises the guest NAT subnet and Janus approves it.
    service = tailnet.service()
    change = DeviceChange("hummingbird", "routes", {"routes": ["10.77.0.0/24"]})
    prepared = prepare_device_change(service, change)
    assert prepared["ok"] is True and tailnet.writes == []
    frozen = prepared["frozen"]
    assert frozen["device_id"] == HUMMINGBIRD["nodeId"] and frozen["device_name"] == "hummingbird"
    assert frozen["previous"] == {"enabled_routes": []} and frozen["device_tags"] == []
    result = apply_device_change(service, change, **frozen)
    assert (result["ok"], result["status"], result["verification_ok"]) == (True, "executed", True)
    assert result["observed"] == {"enabled_routes": ["10.77.0.0/24"]}
    assert result["retry_safe"] is False and tailnet.writes == [("POST", f"/api/v2/device/{HUMMINGBIRD['nodeId']}/routes")]


def test_a_route_the_device_does_not_advertise_is_refused_before_approval(tailnet):
    # Approving a route nobody advertises does nothing on Tailscale; it would only look like success.
    prepared = prepare_device_change(tailnet.service(), DeviceChange("hummingbird", "routes",
                                                                     {"routes": ["192.168.0.0/24"]}))
    assert (prepared["ok"], prepared["status"]) == (False, "denied")
    assert "does not advertise" in prepared["error"] and tailnet.writes == []


def test_a_device_named_twice_or_not_at_all_is_refused(tailnet):
    tailnet.devices["dup"] = dict(LAB, nodeId="nDUPXXXXXXXCNTRL", id="9234567890123456")
    twice = prepare_device_change(tailnet.service(), DeviceChange("janus-router-lab", "authorized",
                                                                  {"authorized": False}))
    assert (twice["ok"], twice["status"]) == (False, "denied") and "2 devices" in twice["error"]
    missing = prepare_device_change(tailnet.service(), DeviceChange("nope", "authorized", {"authorized": False}))
    assert (missing["ok"], missing["status"]) == (False, "denied") and "no device" in missing["error"]


def test_a_device_that_changed_since_approval_is_not_touched(tailnet):
    service = tailnet.service()
    change = DeviceChange("janus-router-lab", "tags", {"tags": ["tag:janus-lab", "tag:janus-trainer"]})
    frozen = prepare_device_change(service, change)["frozen"]
    tailnet.devices[LAB["nodeId"]]["tags"] = ["tag:other"]          # someone else retagged it
    result = apply_device_change(service, change, **frozen)
    assert (result["ok"], result["status"], result["changed"]) == (False, "conflict", False)
    assert tailnet.writes == []


def test_a_lost_answer_is_an_unknown_outcome_that_is_never_resent(tailnet):
    service = tailnet.service()
    change = DeviceChange("janus-router-lab", "key_expiry", {"key_expiry_disabled": False})
    frozen = prepare_device_change(service, change)["frozen"]
    tailnet.drop_answer_of = "/key"
    result = apply_device_change(service, change, **frozen)
    assert (result["ok"], result["status"], result["retry_safe"]) == (False, "outcome_unknown", False)
    assert len(tailnet.writes) == 1                                  # exactly one request, no retry


def test_tags_only_accept_tags_and_every_tag_needs_an_owner_prefix(tailnet):
    bad = prepare_device_change(tailnet.service(), DeviceChange("janus-router-lab", "tags", {"tags": ["admin"]}))
    assert (bad["ok"], bad["status"]) == (False, "denied") and "tag:" in bad["error"]


def test_removing_a_device_is_confirmed_by_listing_the_devices_again(tailnet):
    service = tailnet.service()
    change = DeviceChange("Redmi Note 11", "remove", {})
    frozen = prepare_device_change(service, change)["frozen"]
    result = apply_device_change(service, change, **frozen)
    assert (result["ok"], result["verification_ok"]) == (True, True)
    assert PHONE["nodeId"] not in tailnet.devices


def test_rename_and_authorize_read_back_their_value(tailnet):
    service = tailnet.service()
    rename = DeviceChange("janus-router-lab", "name", {"name": "janus-lab-router"})
    result = apply_device_change(service, rename, **prepare_device_change(service, rename)["frozen"])
    assert result["ok"] and result["observed"] == {"name": "janus-lab-router"}
    off = DeviceChange("Redmi Note 11", "authorized", {"authorized": False})
    result = apply_device_change(service, off, **prepare_device_change(service, off)["frozen"])
    assert result["ok"] and result["observed"] == {"authorized": False}


def test_an_unknown_setting_is_refused():
    with pytest.raises(TailnetChangeRefused):
        DeviceChange("x", "expire_everything", {}).validated()


# ---- DNS: nameservers, MagicDNS, search domains, split DNS ----

def test_dns_nameservers_are_replaced_with_the_previous_value_frozen(tailnet):
    service = tailnet.service()
    prepared = prepare_dns_change(service, "nameservers", {"nameservers": ["1.1.1.1", "9.9.9.9"]})
    assert prepared["ok"] and prepared["frozen"]["previous"] == {"nameservers": ["1.1.1.1"]}
    result = apply_dns_change(service, "nameservers", {"nameservers": ["1.1.1.1", "9.9.9.9"]},
                              **prepared["frozen"])
    assert (result["ok"], result["verification_ok"]) == (True, True)
    assert tailnet.nameservers == ["1.1.1.1", "9.9.9.9"]


def test_dns_that_moved_since_approval_is_not_overwritten(tailnet):
    service = tailnet.service()
    prepared = prepare_dns_change(service, "magic_dns", {"magic_dns": False})
    tailnet.magic_dns = False
    tailnet.nameservers = ["8.8.8.8"]
    result = apply_dns_change(service, "magic_dns", {"magic_dns": False}, **prepared["frozen"])
    assert (result["ok"], result["status"]) == (False, "conflict") and tailnet.writes == []


def test_split_dns_sets_one_domain_and_reads_it_back(tailnet):
    service = tailnet.service()
    value = {"domain": "lab.internal", "nameservers": ["10.77.0.1"]}
    prepared = prepare_dns_change(service, "split_dns", value)
    result = apply_dns_change(service, "split_dns", value, **prepared["frozen"])
    assert result["ok"] and tailnet.split == {"lab.internal": ["10.77.0.1"]}


def test_a_nameserver_must_be_an_address(tailnet):
    prepared = prepare_dns_change(tailnet.service(), "nameservers", {"nameservers": ["dns.example"]})
    assert (prepared["ok"], prepared["status"]) == (False, "denied") and "address" in prepared["error"]


# ---- keys: revoke one auth key, never Janus's own client ----

def test_an_auth_key_is_revoked_and_confirmed_gone(tailnet):
    service = tailnet.service()
    prepared = prepare_key_revoke(service, "kAAAA1111CNTRL")
    assert prepared["ok"] and prepared["frozen"]["key_description"] == "lab enrol"
    result = apply_key_revoke(service, "kAAAA1111CNTRL", **prepared["frozen"])
    assert (result["ok"], result["verification_ok"]) == (True, True) and "kAAAA1111CNTRL" not in tailnet.keys


def test_janus_never_revokes_the_client_it_runs_on(tailnet):
    service = TailnetService(TailnetConfig("kCLIENT22CNTRL", "secret", read_attempts=1, retry_wait_seconds=0),
                             httpx.Client(transport=httpx.MockTransport(tailnet.handler)))
    prepared = prepare_key_revoke(service, "kCLIENT22CNTRL")
    assert (prepared["ok"], prepared["status"]) == (False, "denied") and "Janus" in prepared["error"]
