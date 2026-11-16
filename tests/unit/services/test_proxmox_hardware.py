"""Physical disks and backups on Proxmox nodes, read-only.

Disks come from /nodes/{node}/disks/list: device, type, size, the SMART health
Proxmox reports and the SSD wear (Proxmox's `wearout` is the life *remaining*
in percent; the web UI shows 100 - wearout as used). Serial numbers, WWNs and
by-id links identify hardware and never leave the service.

Backups come from the node's backup-capable storages
(/nodes/{node}/storage?content=backup, then .../content?content=backup): the
newest backup per guest with its age, size, protection, encryption and
verification. Volume ids, notes and paths stay inside the service.
"""
import json
import time

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"
NOW = 1_794_800_000

DISKS = [
    {"devpath": "/dev/nvme0n1", "type": "nvme", "size": 512110190592, "health": "PASSED", "wearout": 88,
     "used": "BIOS boot", "rpm": 0, "gpt": 1, "model": "INTEL SSDPEKNW512G8", "serial": "BTNH0000000000",
     "wwn": "eui.0000000000000000", "by_id_link": "/dev/disk/by-id/nvme-INTEL_BTNH0000000000",
     "vendor": "unknown", "osdid": -1, "osdid-list": None},
    {"devpath": "/dev/sda", "type": "hdd", "size": 4000787030016, "health": "FAILED", "wearout": "N/A",
     "used": "ZFS", "rpm": 7200, "gpt": 1, "model": "WDC WD40EFRX", "serial": "WD-0000", "wwn": "0x50014ee000000000"},
    {"devpath": "/dev/sdb", "type": "ssd", "size": 1000204886016, "health": "OK", "wearout": 8, "used": None,
     "rpm": 0, "gpt": 0, "model": "Samsung SSD 870"},
    {"devpath": "/dev/sdc", "type": "usb", "size": 64023257088, "health": "UNKNOWN", "wearout": "N/A",
     "used": "partitions", "rpm": 0, "gpt": 0, "model": "Flash Disk"},
]


def service(routes, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        status, data = routes[request.url.path]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client), client


def test_node_disks_report_health_and_wear_without_hardware_identifiers():
    calls = []
    client_service, client = service({"/api2/json/nodes/pve-a/disks/list": (200, DISKS)}, calls)
    try:
        result = client_service.node_disks("pve-a")
    finally:
        client.close()
    assert calls == [("GET", "/api2/json/nodes/pve-a/disks/list", {})]
    data = result["data"]
    assert data["node"] == "pve-a" and data["count"] == 4 and data["truncated"] is False
    rows = {row["device"]: row for row in data["disks"]}
    assert rows["/dev/nvme0n1"] == {
        "device": "/dev/nvme0n1", "type": "nvme", "model": "INTEL SSDPEKNW512G8", "size_bytes": 512110190592,
        "health": "passed", "wear_used_percent": 12, "used_by": "BIOS boot", "issues": []}
    # SMART said FAILED: the disk is failing whatever else it reports.
    assert rows["/dev/sda"]["health"] == "failed" and rows["/dev/sda"]["issues"] == ["smart_failed"]
    assert rows["/dev/sda"]["wear_used_percent"] is None
    # 92 % of rated life used is named; "OK" is the SAS/NVMe word for a pass.
    assert rows["/dev/sdb"]["health"] == "passed" and rows["/dev/sdb"]["wear_used_percent"] == 92
    assert rows["/dev/sdb"]["issues"] == ["worn_out"] and rows["/dev/sdb"]["used_by"] is None
    # No SMART answer is unknown, never healthy.
    assert rows["/dev/sdc"]["health"] == "unknown" and rows["/dev/sdc"]["issues"] == []
    # Failing disks first, then other issues, then the rest by device.
    assert [row["device"] for row in data["disks"]] == ["/dev/sda", "/dev/sdb", "/dev/nvme0n1", "/dev/sdc"]
    assert data["summary"] == {"disks": 4, "failing": 1, "worn": 1, "health_unknown": 1}
    text = json.dumps(result)
    for identifier in ("BTNH0000000000", "eui.", "WD-0000", "0x50014ee", "by-id"):
        assert identifier not in text


@pytest.mark.parametrize("health,expected", [
    ("PASSED", "passed"), ("OK", "passed"), ("FAILED", "failed"), ("FAILED!", "failed"),
    ("UNKNOWN", "unknown"), ("SMART Disabled", "smart_disabled"), (None, "unknown"), ("", "unknown"),
])
def test_smart_health_words_map_to_one_fixed_set(health, expected):
    row = dict(DISKS[2], health=health)
    client_service, client = service({"/api2/json/nodes/pve-a/disks/list": (200, [row])})
    try:
        assert client_service.node_disks("pve-a")["data"]["disks"][0]["health"] == expected
    finally:
        client.close()


@pytest.mark.parametrize("change", [{"size": "big"}, {"wearout": -5}, {"wearout": 130}, {"devpath": 7},
                                    {"devpath": "sda"}, {"type": None}])
def test_a_malformed_disk_row_is_refused(change):
    client_service, client = service({"/api2/json/nodes/pve-a/disks/list": (200, [dict(DISKS[2], **change)])})
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.node_disks("pve-a")
    finally:
        client.close()


def test_a_node_argument_cannot_change_the_disk_read_path():
    client_service, client = service({})
    try:
        for node in ("../cluster", "pve-a/disks", "", " pve-a"):
            with pytest.raises(ValueError, match="exact Proxmox node"):
                client_service.node_disks(node)
    finally:
        client.close()


DAY = 86_400
RESOURCES = [
    {"id": "node/pve-a", "type": "node", "node": "pve-a", "status": "online"},
    {"id": "qemu/101", "type": "qemu", "vmid": 101, "node": "pve-a", "name": "orders-db", "status": "running",
     "tags": "db-postgres"},
    {"id": "lxc/102", "type": "lxc", "vmid": 102, "node": "pve-a", "name": "cache", "status": "running"},
    {"id": "lxc/103", "type": "lxc", "vmid": 103, "node": "pve-a", "name": "old", "status": "stopped"},
    {"id": "lxc/104", "type": "lxc", "vmid": 104, "node": "pve-a", "name": "fresh", "status": "running",
     "template": 1},
]
BACKUP_STORAGES = [{"storage": "local", "type": "dir", "content": "iso,vztmpl,backup", "active": 1, "enabled": 1,
                    "shared": 0},
                   {"storage": "pbs", "type": "pbs", "content": "backup", "active": 1, "enabled": 1, "shared": 1}]


def volume(storage, vmid, age_days, **changes):
    row = {"volid": f"{storage}:backup/vzdump-qemu-{vmid}-x.vma.zst", "content": "backup", "vmid": vmid,
           "ctime": NOW - int(age_days * DAY), "size": 2147483648, "format": "vma.zst",
           "notes": "orders-db nightly, contains secret-looking text", "subtype": "qemu"}
    row.update(changes)
    return row


def backup_routes(local=(), pbs=(), jobs=(200, [{"id": "backup-1", "enabled": 1, "schedule": "02:00"}]),
                  uncovered=(200, [{"vmid": 103, "type": "lxc"}])):
    return {"/api2/json/cluster/resources": (200, RESOURCES),
            "/api2/json/cluster/backup": jobs,
            "/api2/json/cluster/backup-info/not-backed-up": uncovered,
            "/api2/json/nodes/pve-a/storage": (200, BACKUP_STORAGES),
            "/api2/json/nodes/pve-a/storage/local/content": (200, list(local)),
            "/api2/json/nodes/pve-a/storage/pbs/content": (200, list(pbs))}


def test_backups_show_the_newest_backup_of_each_guest_and_what_is_overdue(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    calls = []
    client_service, client = service(backup_routes(
        local=[volume("local", 101, 3), volume("local", 102, 12, subtype="lxc", format="tar.zst")],
        pbs=[volume("pbs", 101, 0.5, encrypted="fingerprint", verification={"state": "ok", "upid": "UPID:x"},
                    protected=1)]), calls)
    try:
        result = client_service.backups()
    finally:
        client.close()
    data = result["data"]
    assert data["schema"] == "proxmox-backups.v1"
    assert data["jobs"] == {"count": 1, "enabled": 1}
    assert data["coverage_known"] is True and data["listing_complete"] is True
    # The threshold travels with the answer, so a reader never assumes its own.
    assert data["stale_after_days"] == 8
    rows = {row["vmid"]: row for row in data["guests"]}
    assert set(rows) == {101, 102, 103}
    # The newest copy wins, from whichever storage holds it.
    assert rows[101] == {"vmid": 101, "node": "pve-a", "type": "qemu", "name": "orders-db", "backup_job": True,
                         "backups": 2, "latest": {"storage": "pbs", "time": NOW - DAY // 2, "age_days": 0.5,
                                                  "size_bytes": 2147483648, "protected": True, "encrypted": True,
                                                  "verified": "ok"},
                         "issues": []}
    assert rows[102]["latest"]["age_days"] == 12.0 and rows[102]["latest"]["verified"] is None
    assert rows[102]["issues"] == ["backup_stale"]
    # No job covers 103 and it has never been backed up.
    assert rows[103]["latest"] is None and rows[103]["backups"] == 0
    assert rows[103]["issues"] == ["never_backed_up", "no_backup_job"]
    # Most urgent first.
    assert [row["vmid"] for row in data["guests"]] == [103, 102, 101]
    assert data["summary"] == {"guests": 3, "with_recent_backup": 1, "stale": 1, "never": 1,
                               "without_backup_job": 1}
    assert data["storages"] == [{"node": "pve-a", "storage": "local", "type": "dir", "shared": False, "backups": 2},
                                {"node": "pve-a", "storage": "pbs", "type": "pbs", "shared": True, "backups": 1}]
    text = json.dumps(result)
    assert "vzdump-" not in text and "secret-looking" not in text and "volid" not in text
    assert all(method == "GET" for method, _, _ in calls)
    assert ("GET", "/api2/json/nodes/pve-a/storage", {"content": "backup", "enabled": "1"}) in calls
    assert ("GET", "/api2/json/nodes/pve-a/storage/pbs/content", {"content": "backup"}) in calls


def test_guests_are_ordered_by_how_urgent_their_backup_gap_is(monkeypatch):
    # No backup at all comes first, then an overdue one; a missing job beside a
    # recent manual backup is a risk for later, so it comes after both.
    monkeypatch.setattr(time, "time", lambda: NOW)
    resources = RESOURCES + [{"id": "lxc/105", "type": "lxc", "vmid": 105, "node": "pve-a", "name": "manual",
                              "status": "running"}]
    routes = backup_routes(local=[volume("local", 101, 1), volume("local", 102, 12), volume("local", 105, 1)],
                           uncovered=(200, [{"vmid": 103, "type": "lxc"}, {"vmid": 105, "type": "lxc"}]))
    routes["/api2/json/cluster/resources"] = (200, resources)
    client_service, client = service(routes)
    try:
        data = client_service.backups()["data"]
    finally:
        client.close()
    assert [row["vmid"] for row in data["guests"]] == [103, 102, 105, 101]
    rows = {row["vmid"]: row for row in data["guests"]}
    assert rows[105]["issues"] == ["no_backup_job"]


def test_a_shared_backup_storage_is_listed_once_not_once_per_node(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    resources = RESOURCES + [{"id": "node/pve-b", "type": "node", "node": "pve-b", "status": "online"}]
    routes = backup_routes(pbs=[volume("pbs", 101, 1)])
    routes["/api2/json/cluster/resources"] = (200, resources)
    routes["/api2/json/nodes/pve-b/storage"] = (200, [BACKUP_STORAGES[1]])
    calls = []
    client_service, client = service(routes, calls)
    try:
        data = client_service.backups()["data"]
    finally:
        client.close()
    content_reads = [path for _, path, _ in calls if path.endswith("/content")]
    assert content_reads.count("/api2/json/nodes/pve-a/storage/pbs/content") == 1
    assert "/api2/json/nodes/pve-b/storage/pbs/content" not in content_reads
    rows = {row["vmid"]: row for row in data["guests"]}
    assert rows[101]["backups"] == 1 and data["storages"][-1]["backups"] == 1


def test_an_unreadable_backup_storage_makes_the_listing_incomplete_not_empty(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    routes = backup_routes(local=[volume("local", 101, 1)])
    routes["/api2/json/nodes/pve-a/storage/pbs/content"] = (403, None)
    client_service, client = service(routes)
    try:
        data = client_service.backups()["data"]
    finally:
        client.close()
    # Guests without a backup in what could be read are not called "never backed up".
    assert data["listing_complete"] is False
    assert data["unreadable_storages"] == [{"node": "pve-a", "storage": "pbs",
                                            "reason": "Proxmox refused the read (HTTP 403)."}]
    rows = {row["vmid"]: row for row in data["guests"]}
    assert "never_backed_up" not in rows[103]["issues"] and "backup_stale" not in rows[102]["issues"]
    assert rows[101]["latest"]["storage"] == "local"


def test_coverage_the_token_cannot_read_leaves_backup_jobs_unknown(monkeypatch):
    monkeypatch.setattr(time, "time", lambda: NOW)
    client_service, client = service(backup_routes(local=[volume("local", 101, 1)], jobs=(403, None),
                                                   uncovered=(403, None)))
    try:
        data = client_service.backups()["data"]
    finally:
        client.close()
    assert data["jobs"] is None and data["coverage_known"] is False
    assert {row["backup_job"] for row in data["guests"]} == {None}
    assert data["summary"]["without_backup_job"] is None


@pytest.mark.parametrize("change", [{"vmid": "101"}, {"ctime": -1}, {"ctime": "yesterday"}, {"size": -2},
                                    {"verification": "ok"}])
def test_a_malformed_backup_volume_is_refused(monkeypatch, change):
    monkeypatch.setattr(time, "time", lambda: NOW)
    row = volume("local", 101, 1)
    row.update(change)
    client_service, client = service(backup_routes(local=[row]))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.backups()
    finally:
        client.close()
