"""Guest snapshots on Proxmox, list only.

Snapshots come from /nodes/{node}/{type}/{vmid}/snapshot (VM.Audit), after the
guest's type is confirmed from the inventory. Proxmox adds a synthetic
"current" entry ("You are here!") for the running state: it is the position
in the tree, never a snapshot. A snapshot's description is free text the
operator wrote and never leaves the service. Nothing here creates, deletes or
rolls back a snapshot.
"""
import json
import time

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"
NOW = 1_794_800_000
DAY = 86_400

RESOURCES = [
    {"id": "node/pve-a", "type": "node", "node": "pve-a", "status": "online"},
    {"id": "qemu/101", "type": "qemu", "vmid": 101, "node": "pve-a", "name": "orders-db", "status": "running"},
    {"id": "lxc/990", "type": "lxc", "vmid": 990, "node": "pve-a", "name": "janus-lab", "status": "running"},
]

QEMU_SNAPSHOTS = [
    {"name": "before-upgrade", "snaptime": NOW - 40 * DAY, "vmstate": 1,
     "description": "root password is hunter2, do not share"},
    {"name": "after-upgrade", "snaptime": NOW - 3 * DAY, "vmstate": 0, "parent": "before-upgrade",
     "description": ""},
    {"name": "current", "digest": "abc123", "running": 1, "description": "You are here!",
     "parent": "after-upgrade"},
]


def service(routes, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/api2/json/cluster/resources":
            wanted = request.url.params.get("type")
            rows = [row for row in routes["resources"]
                    if wanted is None or (wanted == "vm" and row["type"] in {"qemu", "lxc"})]
            return httpx.Response(200, json={"data": rows}, request=request)
        if request.url.path in routes:
            status, data = routes[request.url.path]
            return httpx.Response(status, json={"data": data}, request=request)
        return httpx.Response(404, json={"data": None}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client), client


def test_snapshots_list_name_age_parent_and_ram_without_the_current_marker_or_descriptions(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    calls = []
    client_service, client = service({"resources": RESOURCES,
                                      "/api2/json/nodes/pve-a/qemu/101/snapshot": (200, QEMU_SNAPSHOTS)}, calls)
    try:
        result = client_service.guest_snapshots("pve-a", 101)
    finally:
        client.close()
    data = result["data"]
    assert data["schema"] == "proxmox-guest-snapshots.v1"
    assert (data["node"], data["vmid"], data["type"], data["name"]) == ("pve-a", 101, "qemu", "orders-db")
    # Newest first; "current" is where the guest stands, not a snapshot.
    assert data["snapshots"] == [
        {"name": "after-upgrade", "time": NOW - 3 * DAY, "age_days": 3.0, "parent": "before-upgrade",
         "includes_ram": False},
        {"name": "before-upgrade", "time": NOW - 40 * DAY, "age_days": 40.0, "parent": None,
         "includes_ram": True},
    ]
    assert data["count"] == 2 and data["current_parent"] == "after-upgrade"
    assert data["oldest_age_days"] == 40.0
    text = json.dumps(result)
    assert "hunter2" not in text and "You are here" not in text and "digest" not in text
    assert [path for _, path, _ in calls] == ["/api2/json/cluster/resources",
                                              "/api2/json/nodes/pve-a/qemu/101/snapshot"]
    assert all(method == "GET" for method, _, _ in calls)


def test_a_container_snapshot_has_no_ram_state_and_a_guest_without_snapshots_says_so(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    lxc = [{"name": "clean", "snaptime": NOW - DAY, "description": ""},
           {"name": "current", "digest": "d", "running": 1, "description": "You are here!", "parent": "clean"}]
    client_service, client = service({"resources": RESOURCES,
                                      "/api2/json/nodes/pve-a/lxc/990/snapshot": (200, lxc)})
    try:
        data = client_service.guest_snapshots("pve-a", 990)["data"]
        assert data["type"] == "lxc"
        assert data["snapshots"] == [{"name": "clean", "time": NOW - DAY, "age_days": 1.0, "parent": None,
                                      "includes_ram": None}]
    finally:
        client.close()
    only_current = [{"name": "current", "digest": "d", "running": 0, "description": "You are here!"}]
    client_service, client = service({"resources": RESOURCES,
                                      "/api2/json/nodes/pve-a/lxc/990/snapshot": (200, only_current)})
    try:
        data = client_service.guest_snapshots("pve-a", 990)["data"]
        assert data["snapshots"] == [] and data["count"] == 0
        assert data["current_parent"] is None and data["oldest_age_days"] is None
    finally:
        client.close()


def test_a_snapshot_without_a_recorded_time_has_an_unknown_age(monkeypatch):
    # Proxmox sends snaptime 0 when the config has none; that is not 1970.
    monkeypatch.setattr(time, "time", lambda: NOW)
    rows = [{"name": "legacy", "snaptime": 0, "vmstate": 0, "description": ""},
            {"name": "current", "digest": "d", "running": 1, "description": "You are here!", "parent": "legacy"}]
    client_service, client = service({"resources": RESOURCES,
                                      "/api2/json/nodes/pve-a/qemu/101/snapshot": (200, rows)})
    try:
        data = client_service.guest_snapshots("pve-a", 101)["data"]
    finally:
        client.close()
    assert data["snapshots"] == [{"name": "legacy", "time": None, "age_days": None, "parent": None,
                                  "includes_ram": False}]
    assert data["oldest_age_days"] is None


def test_a_guest_missing_from_the_inventory_is_not_found_and_its_snapshots_are_not_read():
    from services.provider_errors import ProviderResourceNotFound

    calls = []
    client_service, client = service({"resources": RESOURCES}, calls)
    try:
        with pytest.raises(ProviderResourceNotFound, match="989"):
            client_service.guest_snapshots("pve-a", 989)
        with pytest.raises(ValueError, match="node"):
            client_service.guest_snapshots("../etc", 101)
        with pytest.raises(ValueError, match="VMID"):
            client_service.guest_snapshots("pve-a", "101")
    finally:
        client.close()
    assert {path for _, path, _ in calls} == {"/api2/json/cluster/resources"}


@pytest.mark.parametrize("row", [
    {"name": "", "snaptime": 1, "description": ""},
    {"name": 5, "snaptime": 1, "description": ""},
    {"name": "x", "snaptime": -1, "description": ""},
    {"name": "x", "snaptime": "yesterday", "description": ""},
    {"name": "x", "snaptime": 1, "vmstate": 2, "description": ""},
    {"name": "x", "snaptime": 1, "parent": 7, "description": ""},
])
def test_a_malformed_snapshot_is_refused(monkeypatch, row):
    monkeypatch.setattr(time, "time", lambda: NOW)
    client_service, client = service({"resources": RESOURCES,
                                      "/api2/json/nodes/pve-a/qemu/101/snapshot": (200, [row])})
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.guest_snapshots("pve-a", 101)
    finally:
        client.close()
