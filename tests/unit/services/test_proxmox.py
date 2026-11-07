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
        assert client_service.node_storage("pve-a")["data"]["storage"][0]["active"] is None
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
            "/api2/json/nodes": [{"node": "pve-a", "status": "online"}],
            "/api2/json/cluster/resources": [{"type": "lxc", "vmid": 201, "node": "pve-a", "status": "running"}],
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
        assert paths == [
            ("GET", "/api2/json/cluster/status", {}),
            ("GET", "/api2/json/nodes", {}),
            ("GET", "/api2/json/cluster/resources", {"type": "vm"}),
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
