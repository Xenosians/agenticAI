"""Proxmox read adapter: fixed paths, bounded evidence and truthful failures."""
import httpx
import pytest

from config.settings import Settings
from services.proxmox import ProxmoxConfig, ProxmoxService, build_proxmox_service

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"


def service(handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client), client


def response(request, data, status=200):
    return httpx.Response(status, json={"data": data}, request=request)


def test_cluster_nodes_guests_storage_and_tasks_use_fixed_lua_gated_requests():
    calls = []
    fixtures = {
        "/api2/json/cluster/status": [{"id": "cluster", "type": "cluster", "name": "lab"}],
        "/api2/json/nodes": [{"node": "pve-a", "status": "online", "maxcpu": 8, "cpu": 0.2}],
        "/api2/json/cluster/resources": [{"type": "qemu", "vmid": 101, "node": "pve-a",
                                          "name": "router-lab", "status": "running", "mem": 512}],
        "/api2/json/nodes/pve-a/storage": [{"storage": "local-lvm", "type": "lvmthin",
                                             "active": 1, "used": 128, "total": 1024}],
        "/api2/json/nodes/pve-a/tasks": [{"upid": "UPID:pve-a:1:task", "type": "qmstart",
                                           "status": "OK", "starttime": 100, "endtime": 101}],
    }

    def handler(request):
        calls.append(request)
        return response(request, fixtures[request.url.path])

    client_service, client = service(handler)
    try:
        assert client_service.cluster_status()["data"]["members"][0]["name"] == "lab"
        assert client_service.nodes()["data"]["nodes"][0]["node"] == "pve-a"
        guest = client_service.guests()["data"]["guests"][0]
        assert guest == {"vmid": 101, "node": "pve-a", "type": "qemu", "name": "router-lab",
                         "status": "running", "cpu_fraction": None, "memory_bytes": 512,
                         "memory_total_bytes": None, "disk_bytes": None, "disk_total_bytes": None,
                         "uptime_seconds": None}
        assert client_service.node_storage("pve-a")["data"]["storage"][0]["active"] is True
        assert client_service.recent_tasks("pve-a")["data"]["tasks"][0]["status"] == "OK"
        assert [request.method for request in calls] == ["GET"] * 5
        assert [request.url.path for request in calls] == list(fixtures)
        assert dict(calls[2].url.params) == {"type": "vm"}
        assert dict(calls[4].url.params) == {"limit": "50"}
        assert all(request.headers["authorization"] == f"PVEAPIToken={TOKEN_ID}={TOKEN_SECRET}" for request in calls)
        assert all(request.url.host == "pve.example.test" for request in calls)
    finally:
        client.close()


def test_overview_aggregates_live_provider_reads_under_one_typed_schema():
    paths = []

    def handler(request):
        paths.append((request.method, request.url.path, dict(request.url.params)))
        payloads = {
            "/api2/json/cluster/status": [{"id": "cluster", "type": "cluster"}],
            "/api2/json/cluster/resources": [
                {"id": "node/pve-a", "type": "node", "node": "pve-a", "status": "online"},
                {"id": "lxc/201", "type": "lxc", "vmid": 201, "node": "pve-a", "status": "running",
                 "tags": "db-postgres"}],
            "/api2/json/cluster/backup-info/not-backed-up": [{"vmid": 201, "type": "lxc"}],
        }
        return response(request, payloads[request.url.path])

    client_service, client = service(handler)
    try:
        overview = client_service.overview()
        assert overview["ok"] is True
        assert overview["status"] == "success"
        assert overview["provider"] == "proxmox"
        assert overview["data"]["schema"] == "proxmox-overview.v1"
        assert overview["data"]["nodes"]["nodes"][0]["status"] == "online"
        assert overview["data"]["guests"]["guests"][0]["vmid"] == 201
        # The database section comes from the same inventory read plus the coverage read.
        databases = overview["data"]["databases"]
        assert databases["schema"] == "proxmox-database-inventory.v1"
        assert [(row["vmid"], row["engine"], row["backup_job"]) for row in databases["databases"]] == [
            (201, "postgres", False)]
        assert paths == [
            ("GET", "/api2/json/cluster/status", {}),
            ("GET", "/api2/json/cluster/resources", {}),
            ("GET", "/api2/json/cluster/backup-info/not-backed-up", {}),
        ]
    finally:
        client.close()


def test_node_arguments_cannot_change_the_provider_path():
    calls = []
    client_service, client = service(lambda request: calls.append(request) or response(request, []))
    try:
        for node in ("../other", "pve-a/../../access", "", "pve a", None):
            with pytest.raises(ValueError):
                client_service.node_storage(node)
        assert calls == []
    finally:
        client.close()


def test_api_errors_and_invalid_receipts_are_not_converted_to_success():
    client_service, client = service(lambda request: response(request, [], status=403))
    try:
        with pytest.raises(RuntimeError, match="HTTP 403"):
            client_service.nodes()
    finally:
        client.close()

    client_service, client = service(lambda request: response(request, {"nodes": []}))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.nodes()
    finally:
        client.close()

    client_service, client = service(lambda request: httpx.Response(
        200, content=b'{"data":[{"node":"pve-a","status":"online","mem":NaN}]}', request=request))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.nodes()
    finally:
        client.close()


def test_configuration_requires_https_and_a_valid_api_token_id():
    for config in (ProxmoxConfig("http://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET),
                   ProxmoxConfig("https://user:pass@pve.example.test:8006", TOKEN_ID, TOKEN_SECRET),
                   ProxmoxConfig("https://pve.example.test:8006/api2/json", TOKEN_ID, TOKEN_SECRET),
                   ProxmoxConfig("https://pve.example.test:99999", TOKEN_ID, TOKEN_SECRET),
                   ProxmoxConfig("https://127.0.0.1:8006", TOKEN_ID, TOKEN_SECRET),
                   ProxmoxConfig("https://localhost:8006", TOKEN_ID, TOKEN_SECRET)):
        with pytest.raises(ValueError):
            ProxmoxService(config, httpx.Client(transport=httpx.MockTransport(lambda request: response(request, []))))
    with pytest.raises(ValueError, match="token configuration"):
        ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", "invalid", TOKEN_SECRET),
                       httpx.Client(transport=httpx.MockTransport(lambda request: response(request, []))))


def test_a_lan_origin_verifies_the_certificate_name_without_weakening_tls():
    seen = []

    def handler(request):
        seen.append(request.extensions.get("sni_hostname"))
        return response(request, [])

    config = ProxmoxConfig("https://192.0.2.10:8006", TOKEN_ID, TOKEN_SECRET,
                           ca_file=None, server_name="pve.example.test")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    client_service = ProxmoxService(config, client)
    try:
        client_service.nodes()
        assert seen == ["pve.example.test"]
    finally:
        client.close()


def test_a_malformed_certificate_name_is_refused():
    client = httpx.Client(transport=httpx.MockTransport(lambda request: response(request, [])))
    try:
        for name in ("*", "pve..example.test", "-pve.example.test", "pve example.test"):
            with pytest.raises(ValueError, match="TLS server name"):
                ProxmoxService(ProxmoxConfig("https://192.0.2.10:8006", TOKEN_ID, TOKEN_SECRET,
                                             server_name=name), client)
    finally:
        client.close()


def test_both_tailnet_origin_shapes_are_accepted():
    # Native Proxmox behind its own TLS on 8006; delegated reads via
    # `tailscale serve` use the default HTTPS port (no explicit port).
    for origin in ("https://pve.example.test:8006",
                   "https://spare-pc.tail12345.ts.net",
                   "https://spare-pc.tail12345.ts.net:8006",
                   "https://spare-pc.tail12345.ts.net/"):
        parsed = ProxmoxConfig(origin, TOKEN_ID, TOKEN_SECRET)
        assert parsed.origin() == origin.rstrip("/")


def test_factory_keeps_an_unconfigured_provider_unavailable():
    settings = Settings(_env_file=None)
    assert build_proxmox_service(settings) is None
    assert not settings.proxmox_base_url


def test_proxmox_integer_flags_are_reported_as_booleans():
    # Proxmox VE 9 sends these flags as 0/1 integers (observed live on
    # /cluster/status and /nodes/{node}/storage), not JSON booleans.
    def handler(request):
        payloads = {
            "/api2/json/cluster/status": [{"id": "node/pve-a", "type": "node", "name": "pve-a", "online": 1},
                                          {"id": "node/pve-b", "type": "node", "name": "pve-b", "online": 0}],
            "/api2/json/nodes/pve-a/storage": [{"storage": "local", "type": "dir", "active": 1, "enabled": 0}],
        }
        return response(request, payloads[request.url.path])

    client_service, client = service(handler)
    try:
        members = client_service.cluster_status()["data"]["members"]
        assert [member["online"] for member in members] == [True, False]
        pool = client_service.node_storage("pve-a")["data"]["storage"][0]
        assert (pool["active"], pool["enabled"]) == (True, False)
    finally:
        client.close()


def test_a_flag_that_is_neither_boolean_nor_zero_or_one_is_an_invalid_answer():
    client_service, client = service(lambda request: response(
        request, [{"id": "node/pve-a", "type": "node", "online": 2}]))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.cluster_status()
    finally:
        client.close()


def guest_handler(resources, details, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.url.path == "/api2/json/cluster/resources":
            # Proxmox applies the type filter itself: type=vm keeps QEMU and LXC rows.
            wanted = request.url.params.get("type")
            rows = [row for row in resources
                    if wanted is None or (wanted == "vm" and row["type"] in {"qemu", "lxc"}) or row["type"] == wanted]
            return response(request, rows)
        if request.url.path in details:
            return response(request, details[request.url.path])
        return httpx.Response(404, json={"data": None}, request=request)
    return handler


LAB_RESOURCES = [{"id": "node/pve-a", "type": "node", "node": "pve-a", "status": "online"},
                 {"id": "lxc/990", "type": "lxc", "vmid": 990, "node": "pve-a", "name": "lab", "status": "stopped"},
                 {"id": "qemu/101", "type": "qemu", "vmid": 101, "node": "pve-a", "name": "db", "status": "running"}]


def test_guest_status_reads_the_exact_guest_after_confirming_its_type():
    calls = []
    details = {"/api2/json/nodes/pve-a/lxc/990/status/current": {
        "vmid": 990, "name": "lab", "status": "stopped", "uptime": 0, "cpus": 1, "maxmem": 134217728,
        "ha": {"managed": 0}, "lock": "backup"}}
    client_service, client = service(guest_handler(LAB_RESOURCES, details, calls))
    try:
        status = client_service.guest_status("pve-a", 990)["data"]
        assert status["schema"] == "proxmox-guest-status.v1"
        assert (status["node"], status["vmid"], status["type"], status["name"]) == ("pve-a", 990, "lxc", "lab")
        assert status["status"] == "stopped"
        assert status["lock"] == "backup"
        assert status["memory_total_bytes"] == 134217728
        assert [path for _, path, _ in calls] == ["/api2/json/cluster/resources",
                                                  "/api2/json/nodes/pve-a/lxc/990/status/current"]
        assert all(method == "GET" for method, _, _ in calls)
    finally:
        client.close()


def test_a_guest_missing_from_the_inventory_is_not_found_and_nothing_else_is_read():
    from services.provider_errors import ProviderResourceNotFound

    calls = []
    client_service, client = service(guest_handler(LAB_RESOURCES, {}, calls))
    try:
        with pytest.raises(ProviderResourceNotFound, match="989"):
            client_service.guest_status("pve-a", 989)
        with pytest.raises(ProviderResourceNotFound):
            client_service.guest_status("pve-b", 990)
        assert {path for _, path, _ in calls} == {"/api2/json/cluster/resources"}
    finally:
        client.close()


def test_guest_arguments_cannot_change_the_provider_path():
    calls = []
    client_service, client = service(guest_handler(LAB_RESOURCES, {}, calls))
    try:
        for node, vmid in (("../x", 990), ("pve-a", 0), ("pve-a", "990"), ("pve-a", True), ("pve-a", -5)):
            with pytest.raises(ValueError):
                client_service.guest_status(node, vmid)
        assert calls == []
    finally:
        client.close()


def test_guest_config_reports_qemu_memory_given_as_text():
    # A QEMU guest's `memory` is a property string in Proxmox VE 9
    # ("4096" or "current=4096"); an LXC guest's is an integer.
    details = {"/api2/json/nodes/pve-a/qemu/101/config": {
        "name": "db", "memory": "current=4096", "cores": 2, "sockets": 1, "onboot": 1, "ostype": "l26",
        "cipassword": "fixture-secret-value", "sshkeys": "ssh-ed25519 AAAA fixture", "net0": "virtio=AA:BB"},
        "/api2/json/nodes/pve-a/lxc/990/config": {
        "hostname": "lab", "memory": 128, "swap": 0, "cores": 1, "onboot": 0, "ostype": "alpine"}}
    client_service, client = service(guest_handler(LAB_RESOURCES, details))
    try:
        qemu = client_service.guest_config("pve-a", 101)["data"]
        assert (qemu["memory_mib"], qemu["on_boot"], qemu["name"]) == (4096, True, "db")
        assert "fixture-secret-value" not in repr(qemu) and "ssh-ed25519" not in repr(qemu)
        assert "net0" not in qemu
        lxc = client_service.guest_config("pve-a", 990)["data"]
        assert (lxc["memory_mib"], lxc["swap_mib"], lxc["on_boot"], lxc["name"]) == (128, 0, False, "lab")
    finally:
        client.close()


def test_guest_config_refuses_unreadable_memory_text():
    details = {"/api2/json/nodes/pve-a/qemu/101/config": {"name": "db", "memory": "lots"}}
    client_service, client = service(guest_handler(LAB_RESOURCES, details))
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.guest_config("pve-a", 101)
    finally:
        client.close()


def test_a_vm_disk_usage_proxmox_does_not_track_is_unknown_not_zero():
    # Proxmox reports a QEMU guest's used disk as 0 ("no info available"); only
    # its size is known. A container's used disk is measured only while it runs;
    # a stopped one reports 0 as well.
    resources = [{"id": "qemu/101", "type": "qemu", "vmid": 101, "node": "pve-a", "name": "db", "status": "running",
                  "disk": 0, "maxdisk": 34359738368},
                 {"id": "lxc/990", "type": "lxc", "vmid": 990, "node": "pve-a", "name": "lab", "status": "running",
                  "disk": 11993088, "maxdisk": 1073741824},
                 {"id": "lxc/991", "type": "lxc", "vmid": 991, "node": "pve-a", "name": "off", "status": "stopped",
                  "disk": 0, "maxdisk": 1073741824}]
    details = {"/api2/json/nodes/pve-a/qemu/101/status/current": {"vmid": 101, "status": "running", "disk": 0,
                                                                  "maxdisk": 34359738368},
               "/api2/json/nodes/pve-a/lxc/991/status/current": {"vmid": 991, "status": "stopped", "disk": 0,
                                                                 "maxdisk": 1073741824}}
    client_service, client = service(guest_handler(resources, details))
    try:
        rows = {row["vmid"]: row for row in client_service.guests()["data"]["guests"]}
        assert (rows[101]["disk_bytes"], rows[101]["disk_total_bytes"]) == (None, 34359738368)
        assert rows[990]["disk_bytes"] == 11993088
        assert (rows[991]["disk_bytes"], rows[991]["disk_total_bytes"]) == (None, 1073741824)
        status = client_service.guest_status("pve-a", 101)["data"]
        assert (status["disk_bytes"], status["disk_total_bytes"]) == (None, 34359738368)
        assert client_service.guest_status("pve-a", 991)["data"]["disk_bytes"] is None
    finally:
        client.close()


def test_overview_reads_nodes_and_guests_from_one_cluster_resources_call():
    calls = []
    resources = [
        {"id": "node/pve-a", "type": "node", "node": "pve-a", "status": "online", "maxcpu": 8, "cpu": 0.25,
         "mem": 512, "maxmem": 1024, "disk": 10, "maxdisk": 100, "uptime": 3600},
        {"id": "lxc/990", "type": "lxc", "vmid": 990, "node": "pve-a", "name": "lab", "status": "stopped",
         "template": 0},
        {"id": "storage/pve-a/local", "type": "storage", "node": "pve-a", "storage": "local", "status": "available"},
    ]

    def handler(request):
        calls.append(request.url.path)
        payloads = {"/api2/json/cluster/status": [{"id": "node/pve-a", "type": "node", "name": "pve-a",
                                                   "online": 1}],
                    "/api2/json/cluster/resources": resources,
                    "/api2/json/cluster/backup-info/not-backed-up": [{"vmid": 990, "type": "lxc"}]}
        return response(request, payloads[request.url.path])

    client_service, client = service(handler)
    try:
        data = client_service.overview()["data"]
        # Nodes, guests and database hosts share one resources read; coverage is the only extra read.
        assert sorted(calls) == ["/api2/json/cluster/backup-info/not-backed-up", "/api2/json/cluster/resources",
                                 "/api2/json/cluster/status"]
        assert data["nodes"]["nodes"] == [{"node": "pve-a", "status": "online", "uptime_seconds": 3600,
                                           "cpu_fraction": 0.25, "cpu_count": 8, "memory_bytes": 512,
                                           "memory_total_bytes": 1024, "disk_bytes": 10, "disk_total_bytes": 100}]
        assert [guest["vmid"] for guest in data["guests"]["guests"]] == [990]
        assert data["cluster"]["members"][0]["online"] is True
    finally:
        client.close()
