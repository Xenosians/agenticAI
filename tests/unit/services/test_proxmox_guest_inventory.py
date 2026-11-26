"""The guest inventory Janus matches human wording against: names, states and tags from one read."""
import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService


def service_for(rows, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"data": rows}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", "reporter@pam!readonly", "secret"), client)


def test_guest_inventory_lists_real_guests_with_tags_from_one_cluster_read():
    calls = []
    service = service_for([
        {"type": "node", "node": "pve-a", "status": "online", "id": "node/pve-a"},
        {"type": "lxc", "vmid": 990, "node": "pve-a", "name": "janus-lab", "status": "stopped", "tags": "janus"},
        {"type": "qemu", "vmid": 120, "node": "pve-b", "name": "db-postgres", "status": "running",
         "tags": "db-postgresql;janus"},
        {"type": "qemu", "vmid": 9000, "node": "pve-a", "name": "tmpl", "status": "stopped", "template": 1},
        {"type": "lxc", "vmid": 101, "node": "pve-a", "status": "running"},
        {"type": "storage", "storage": "local", "node": "pve-a", "status": "available"},
    ], calls)

    data = service.guest_inventory()["data"]

    assert calls == [("/api2/json/cluster/resources", {})]
    assert data == {"truncated": False, "nodes": ["pve-a"], "guests": [
        {"vmid": 101, "node": "pve-a", "type": "lxc", "name": None, "status": "running", "tags": []},
        {"vmid": 990, "node": "pve-a", "type": "lxc", "name": "janus-lab", "status": "stopped", "tags": ["janus"]},
        {"vmid": 120, "node": "pve-b", "type": "qemu", "name": "db-postgres", "status": "running",
         "tags": ["db-postgresql", "janus"]},
    ]}


@pytest.mark.parametrize("row", [
    {"type": "qemu", "vmid": "101", "node": "pve-a", "status": "running"},
    {"type": "qemu", "vmid": 101, "status": "running"},
    {"type": "lxc", "vmid": 101, "node": "pve-a", "status": "running", "name": "bad\nname"},
])
def test_guest_inventory_refuses_a_malformed_answer(row):
    with pytest.raises(RuntimeError, match="invalid answer"):
        service_for([row]).guest_inventory()
