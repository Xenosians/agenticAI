"""Database inventory from Proxmox: guests the operator tagged as database hosts, read-only.

A guest is a database host only when its Proxmox tags say so (``db`` or
``db-<engine>``); names are never guessed from. The inventory reports what
Proxmox observed (state, memory, disk, lock) and whether any backup job covers
the guest. It holds no database credentials and runs no SQL.
"""
import json

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService, database_tags

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"

RESOURCES = [
    {"id": "node/pve-a", "type": "node", "node": "pve-a", "status": "online"},
    {"id": "qemu/101", "type": "qemu", "vmid": 101, "node": "pve-a", "name": "orders-db", "status": "running",
     "tags": "db-postgres;prod", "mem": 3221225472, "maxmem": 4294967296, "disk": 0, "maxdisk": 34359738368,
     "uptime": 86400},
    {"id": "lxc/102", "type": "lxc", "vmid": 102, "node": "pve-a", "name": "cache", "status": "running",
     "tags": "db-redis", "mem": 536870912, "maxmem": 1073741824, "disk": 1020054733, "maxdisk": 1073741824,
     "uptime": 600},
    {"id": "lxc/103", "type": "lxc", "vmid": 103, "node": "pve-a", "name": "reports", "status": "stopped",
     "tags": "db", "mem": 0, "maxmem": 1073741824, "disk": 0, "maxdisk": 8589934592},
    # Not tagged: a name alone never makes a database host.
    {"id": "lxc/104", "type": "lxc", "vmid": 104, "node": "pve-a", "name": "mysql-test", "status": "running",
     "tags": "dbx;lab"},
    {"id": "qemu/105", "type": "qemu", "vmid": 105, "node": "pve-a", "name": "web", "status": "running"},
    {"id": "storage/pve-a/local-lvm", "type": "storage", "node": "pve-a", "storage": "local-lvm",
     "status": "available", "plugintype": "lvmthin", "shared": 0, "content": "images,rootdir",
     "disk": 85899345920, "maxdisk": 107374182400},
    {"id": "storage/pve-a/local", "type": "storage", "node": "pve-a", "storage": "local", "status": "available",
     "plugintype": "dir", "shared": 0, "content": "iso,vztmpl,backup", "disk": 1, "maxdisk": 2},
]
GUEST_RESOURCES = [row for row in RESOURCES if row.get("vmid") is not None]
NOT_BACKED_UP = [{"vmid": 102, "type": "lxc", "name": "cache"}, {"vmid": 104, "type": "lxc", "name": "mysql-test"}]


def service(routes, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        status, data = routes[request.url.path]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client), client


def routes(resources=RESOURCES, not_backed_up=(200, NOT_BACKED_UP)):
    return {"/api2/json/cluster/resources": (200, resources),
            "/api2/json/cluster/backup-info/not-backed-up": not_backed_up}


@pytest.mark.parametrize("tags,expected", [
    ("db", (True, None)), ("db-postgres;prod", (True, "postgres")), ("prod;DB-MariaDB", (True, "mariadb")),
    ("dbx;lab", (False, None)), ("", (False, None)), (None, (False, None)), ("db-", (False, None)),
    ("db-postgres;db-redis", (True, "postgres")), ("database", (False, None)),
])
def test_only_the_db_tag_marks_a_database_host_and_names_its_engine(tags, expected):
    assert database_tags(tags) == expected


def test_the_inventory_lists_tagged_guests_with_their_observed_state_and_backup_coverage():
    calls = []
    client_service, client = service(routes(), calls)
    try:
        result = client_service.database_inventory()
    finally:
        client.close()
    assert result["ok"] is True and result["provider"] == "proxmox"
    data = result["data"]
    assert data["schema"] == "proxmox-database-inventory.v1"
    assert data["count"] == 3 and data["truncated"] is False
    assert data["backup_coverage_known"] is True
    rows = {row["vmid"]: row for row in data["databases"]}
    assert set(rows) == {101, 102, 103}
    assert rows[101] == {
        "vmid": 101, "node": "pve-a", "type": "qemu", "name": "orders-db", "engine": "postgres",
        "status": "running", "lock": None, "memory_bytes": 3221225472, "memory_total_bytes": 4294967296,
        "disk_bytes": None, "disk_total_bytes": 34359738368, "uptime_seconds": 86400, "backup_job": True,
        "issues": []}
    # A container measures its disk; 95 % full is named, and so is a missing backup job.
    assert rows[102]["disk_bytes"] == 1020054733 and rows[102]["backup_job"] is False
    assert rows[102]["issues"] == ["disk_nearly_full", "no_backup_job"]
    # A stopped database host is named first among its issues.
    assert rows[103]["engine"] is None and rows[103]["issues"] == ["not_running"]
    assert [row["vmid"] for row in data["databases"]] == [103, 102, 101]
    # Storage the database disks live on, with how full it is.
    assert data["storage"] == [{"node": "pve-a", "storage": "local-lvm", "type": "lvmthin", "shared": False,
                                "status": "available", "used_bytes": 85899345920, "total_bytes": 107374182400}]
    assert data["summary"] == {"running": 2, "not_running": 1, "without_backup_job": 1, "with_issues": 2}
    assert [path for _, path, _ in calls] == ["/api2/json/cluster/resources",
                                              "/api2/json/cluster/backup-info/not-backed-up"]
    assert all(method == "GET" for method, _, _ in calls)
    assert TOKEN_SECRET not in json.dumps(result)


def test_database_vm_filesystem_read_uses_guest_agent_and_reports_real_mount_usage():
    calls = []
    fsinfo = [
        {"mountpoint": "/var/lib/postgresql", "type": "ext4", "used-bytes": 95, "total-bytes": 100,
         "disk": "/dev/sda1", "disk-uuid": "private-device-id"},
        {"mountpoint": "/", "type": "xfs", "used-bytes": 25, "total-bytes": 100},
    ]
    client_service, client = service({
        "/api2/json/cluster/resources": (200, GUEST_RESOURCES),
        "/api2/json/nodes/pve-a/qemu/101/status/current": (200, {"vmid": 101, "node": "pve-a",
                                                                    "name": "orders-db", "status": "running"}),
        "/api2/json/nodes/pve-a/qemu/101/agent/get-fsinfo": (200, fsinfo),
    }, calls)
    try:
        result = client_service.guest_status("pve-a", 101)
    finally:
        client.close()

    data = result["data"]["database_filesystems"]
    assert result["ok"] is True
    assert data["schema"] == "proxmox-database-filesystems.v1"
    assert (data["state"], data["near_full_count"]) == ("observed", 1)
    assert data["filesystems"] == [
        {"mountpoint": "/", "filesystem": "xfs", "used_bytes": 25, "total_bytes": 100,
         "free_bytes": 75, "usage_percent": 25.0},
        {"mountpoint": "/var/lib/postgresql", "filesystem": "ext4", "used_bytes": 95,
         "total_bytes": 100, "free_bytes": 5, "usage_percent": 95.0},
    ]
    assert "private-device-id" not in json.dumps(data)
    assert [path for _, path, _ in calls] == [
        "/api2/json/cluster/resources",
        "/api2/json/nodes/pve-a/qemu/101/status/current",
        "/api2/json/nodes/pve-a/qemu/101/agent/get-fsinfo",
    ]
    assert all(method == "GET" for method, _, _ in calls)
    assert TOKEN_SECRET not in json.dumps(result)


def test_guest_status_reads_filesystems_only_for_tagged_qemu_database_guests():
    calls = []
    client_service, client = service({
        "/api2/json/cluster/resources": (200, GUEST_RESOURCES),
        "/api2/json/nodes/pve-a/lxc/104/status/current": (200, {"vmid": 104, "node": "pve-a",
                                                                    "hostname": "mysql-test", "status": "running"}),
        "/api2/json/nodes/pve-a/lxc/102/status/current": (200, {"vmid": 102, "node": "pve-a",
                                                                    "hostname": "cache", "status": "running"}),
    }, calls)
    try:
        untagged = client_service.guest_status("pve-a", 104)["data"]
        container = client_service.guest_status("pve-a", 102)["data"]
    finally:
        client.close()
    assert "database_filesystems" not in untagged
    assert "database_filesystems" not in container
    assert all("/agent/get-fsinfo" not in path for _, path, _ in calls)


def test_database_filesystem_read_does_not_call_guest_agent_when_vm_is_stopped():
    stopped = {"id": "qemu/107", "type": "qemu", "vmid": 107, "node": "pve-a", "name": "warehouse-db",
              "status": "stopped", "tags": "db-postgres"}
    calls = []
    # The guest status and QEMU agent endpoints are distinct reads; a stopped
    # VM should never reach the agent endpoint.
    client_service, client = service({
        "/api2/json/cluster/resources": (200, [stopped]),
        "/api2/json/nodes/pve-a/qemu/107/status/current": (200, {"vmid": 107, "node": "pve-a",
                                                                      "name": "warehouse-db", "status": "stopped"}),
    }, calls)
    try:
        result = client_service.guest_status("pve-a", 107)["data"]["database_filesystems"]
    finally:
        client.close()
    assert result["state"] == "guest_not_running"
    assert result["filesystems"] == []
    assert [path for _, path, _ in calls] == [
        "/api2/json/cluster/resources", "/api2/json/nodes/pve-a/qemu/107/status/current"]


@pytest.mark.parametrize("failure,expected_note", [
    (403, "Proxmox refused the guest filesystem read (HTTP 403)."),
    (404, "Proxmox did not find the guest filesystem read endpoint or guest."),
    (httpx.ConnectError("connection refused"), "Proxmox did not answer the guest filesystem read."),
    (httpx.ReadTimeout("read timed out"), "Proxmox did not answer the guest filesystem read."),
])
def test_guest_agent_failures_are_reported_as_unavailable(failure, expected_note):
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path.endswith("/agent/get-fsinfo"):
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, json={"data": None}, request=request)
        if request.url.path.endswith("/cluster/resources"):
            return httpx.Response(200, json={"data": GUEST_RESOURCES}, request=request)
        return httpx.Response(200, json={"data": {"vmid": 101, "node": "pve-a",
                                                   "name": "orders-db", "status": "running"}},
                              request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    client_service = ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client)
    try:
        result = client_service.guest_status("pve-a", 101)
    finally:
        client.close()

    observation = result["data"]["database_filesystems"]
    assert result["ok"] is True
    assert observation["state"] == "unavailable"
    assert observation["filesystems"] == []
    assert observation["note"] == expected_note
    assert calls[-1] == ("GET", "/api2/json/nodes/pve-a/qemu/101/agent/get-fsinfo")
    assert TOKEN_SECRET not in json.dumps(result)


def test_optional_guest_agent_counters_produce_a_truthful_partial_observation():
    fsinfo = [
        {"mountpoint": "/", "type": "ext4", "used-bytes": 40},
        {"mountpoint": "/data", "type": "xfs", "total-bytes": 200},
        {"mountpoint": "/logs", "type": "btrfs"},
    ]
    client_service, client = service({
        "/api2/json/cluster/resources": (200, GUEST_RESOURCES),
        "/api2/json/nodes/pve-a/qemu/101/status/current": (200, {"vmid": 101, "node": "pve-a",
                                                                    "name": "orders-db", "status": "running"}),
        "/api2/json/nodes/pve-a/qemu/101/agent/get-fsinfo": (200, fsinfo),
    })
    try:
        observation = client_service.guest_status("pve-a", 101)["data"]["database_filesystems"]
    finally:
        client.close()

    assert observation["state"] == "partial"
    assert observation["note"] == "QEMU Guest Agent did not report all optional filesystem size counters."
    assert observation["filesystems"] == [
        {"mountpoint": "/", "filesystem": "ext4", "used_bytes": 40, "total_bytes": None,
         "free_bytes": None, "usage_percent": None},
        {"mountpoint": "/data", "filesystem": "xfs", "used_bytes": None, "total_bytes": 200,
         "free_bytes": None, "usage_percent": None},
        {"mountpoint": "/logs", "filesystem": "btrfs", "used_bytes": None, "total_bytes": None,
         "free_bytes": None, "usage_percent": None},
    ]
    assert observation["near_full_count"] == 0


@pytest.mark.parametrize("bad_mount", [
    {"mountpoint": "/", "type": "ext4", "used-bytes": 101, "total-bytes": 100},
    {"mountpoint": "/", "type": "ext4", "used-bytes": True, "total-bytes": 100},
    {"mountpoint": "/", "type": "ext4", "used-bytes": 1, "total-bytes": 2 ** 64},
    {"type": "ext4", "used-bytes": 1, "total-bytes": 2},
])
def test_malformed_guest_filesystem_receipts_are_rejected(bad_mount):
    client_service, client = service({
        "/api2/json/cluster/resources": (200, GUEST_RESOURCES),
        "/api2/json/nodes/pve-a/qemu/101/status/current": (200, {"vmid": 101, "node": "pve-a",
                                                                    "name": "orders-db", "status": "running"}),
        "/api2/json/nodes/pve-a/qemu/101/agent/get-fsinfo": (200, [bad_mount]),
    })
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.guest_status("pve-a", 101)
    finally:
        client.close()


def test_guest_filesystem_list_is_bounded():
    too_many = [{"mountpoint": f"/mount/{index}", "type": "ext4", "used-bytes": 1, "total-bytes": 2}
                for index in range(129)]
    client_service, client = service({
        "/api2/json/cluster/resources": (200, GUEST_RESOURCES),
        "/api2/json/nodes/pve-a/qemu/101/status/current": (200, {"vmid": 101, "node": "pve-a",
                                                                    "name": "orders-db", "status": "running"}),
        "/api2/json/nodes/pve-a/qemu/101/agent/get-fsinfo": (200, too_many),
    })
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.guest_status("pve-a", 101)
    finally:
        client.close()


def test_backup_coverage_the_token_cannot_read_stays_unknown_not_missing():
    # /cluster/backup-info needs Sys.Audit on /; without it the guests still list.
    client_service, client = service(routes(not_backed_up=(403, None)))
    try:
        data = client_service.database_inventory()["data"]
    finally:
        client.close()
    assert data["backup_coverage_known"] is False
    assert data["backup_coverage_note"] == ("Proxmox refused the backup coverage read (HTTP 403); "
                                            "the token needs Sys.Audit on / to see backup jobs.")
    assert {row["backup_job"] for row in data["databases"]} == {None}
    assert all("no_backup_job" not in row["issues"] for row in data["databases"])
    assert data["summary"]["without_backup_job"] is None


@pytest.mark.parametrize("failure,note", [
    (httpx.Response(404, json={"data": None}), "This Proxmox version does not report backup coverage."),
    (httpx.ConnectError("refused"), "Proxmox did not answer the backup coverage read."),
    (httpx.ReadTimeout("slow"), "Proxmox did not answer the backup coverage read."),
])
def test_a_coverage_read_without_an_answer_leaves_coverage_unknown(failure, note):
    def handler(request):
        if request.url.path.endswith("/not-backed-up"):
            if isinstance(failure, Exception):
                raise failure
            return failure
        return httpx.Response(200, json={"data": RESOURCES}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    client_service = ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client)
    try:
        data = client_service.database_inventory()["data"]
    finally:
        client.close()
    assert (data["backup_coverage_known"], data["backup_coverage_note"]) == (False, note)
    assert data["count"] == 3 and {row["backup_job"] for row in data["databases"]} == {None}


def test_no_tagged_guest_is_an_empty_inventory_that_says_how_to_label_one():
    client_service, client = service(routes(resources=[row for row in RESOURCES if row.get("vmid") in (104, 105)]))
    try:
        data = client_service.database_inventory()["data"]
    finally:
        client.close()
    assert data["count"] == 0 and data["databases"] == [] and data["storage"] == []
    assert data["label_hint"] == ('Tag a VM or container "db" or "db-<engine>" (for example "db-postgres") '
                                  'in Proxmox to list it here.')


def test_a_stopped_host_comes_first_and_a_lock_is_the_mildest_issue():
    # Every backup run locks its guest while the guest keeps serving, so a lock
    # ranks below a nearly full disk or a missing backup job.
    locked = {"id": "lxc/106", "type": "lxc", "vmid": 106, "node": "pve-a", "name": "billing", "status": "running",
              "tags": "db-mariadb", "lock": "backup", "maxdisk": 8589934592}
    client_service, client = service(routes(resources=RESOURCES + [locked]))
    try:
        data = client_service.database_inventory()["data"]
    finally:
        client.close()
    assert [(row["vmid"], row["issues"]) for row in data["databases"]] == [
        (103, ["not_running"]), (102, ["disk_nearly_full", "no_backup_job"]), (106, ["locked"]), (101, [])]


@pytest.mark.parametrize("change", [{"tags": 7}, {"tags": "db\u0000"}, {"vmid": "101"}, {"status": None}])
def test_a_malformed_guest_row_is_refused(change):
    rows = [dict(RESOURCES[1], **change)]
    client_service, client = service(routes(resources=rows))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.database_inventory()
    finally:
        client.close()


def test_a_malformed_backup_coverage_answer_is_refused_not_read_as_covered():
    client_service, client = service(routes(not_backed_up=(200, [{"vmid": "102"}])))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.database_inventory()
    finally:
        client.close()
