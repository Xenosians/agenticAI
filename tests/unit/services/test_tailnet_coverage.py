"""Tailnet coverage: which Proxmox nodes and guests have a tailnet device (direction note plan item D1).

The two inventories share no identifier Janus may expose (no addresses), so a
machine and a device match by name: the device's host name or the first label
of its MagicDNS name, normalized the way Tailscale does (lower case; anything
but letters, digits and hyphens becomes a hyphen). A machine without a match
may still be on the tailnet under another name. When either listing was cut,
nothing is called "not on the tailnet".
"""
import httpx
import pytest

from services.proxmox import MAX_ROWS, ProxmoxConfig, ProxmoxService
from services.tailnet_coverage import tailnet_coverage

DEVICES = {"count": 4, "truncated": False, "devices": [
    {"name": "Blackhawk", "dns_name": "blackhawk.tail1234.ts.net", "online": True},
    {"name": "hummingbird", "dns_name": "hummingbird.tail1234.ts.net", "online": True},
    {"name": "janus-router-lab", "dns_name": "janus-router-lab.tail1234.ts.net", "online": True},
    {"name": "Redmi Note 11", "dns_name": "redmi-note-11.tail1234.ts.net", "online": False},
]}
MACHINES = {"truncated": False,
            "nodes": [{"node": "hummingbird", "status": "online"}],
            "guests": [{"vmid": 990, "node": "hummingbird", "type": "lxc", "name": "janus-lab",
                        "status": "running"}]}


def test_the_live_lab_shape_reads_node_on_and_guest_off_the_tailnet():
    data = tailnet_coverage(MACHINES, DEVICES)
    assert data["schema"] == "tailnet-coverage.v1" and data["match"] == "name" and data["complete"] is True
    assert data["nodes"] == [{"node": "hummingbird", "status": "online",
                              "tailnet_devices": [{"name": "hummingbird", "online": True}], "issues": []}]
    assert data["guests"] == [{"vmid": 990, "node": "hummingbird", "type": "lxc", "name": "janus-lab",
                               "status": "running", "tailnet_devices": [], "issues": ["no_tailnet_device"]}]
    assert data["devices_outside_proxmox"] == ["Blackhawk", "janus-router-lab", "Redmi Note 11"]
    assert data["summary"] == {"nodes": 1, "nodes_with_device": 1, "guests": 1, "guests_with_device": 0,
                               "device_offline": 0, "devices_outside_proxmox": 3}


def test_names_match_the_way_tailscale_normalizes_them():
    machines = {"truncated": False, "nodes": [],
                "guests": [{"vmid": 101, "node": "pve-a", "type": "qemu", "name": "redmi-note-11", "status": "running"},
                           {"vmid": 102, "node": "pve-a", "type": "qemu", "name": "Web-1", "status": "running"}]}
    devices = {"count": 2, "truncated": False, "devices": [
        {"name": "Redmi Note 11", "dns_name": None, "online": True},
        # A second device named "web" gets the MagicDNS name web-1.
        {"name": "web", "dns_name": "web-1.tail1234.ts.net", "online": True}]}
    rows = tailnet_coverage(machines, devices)["guests"]
    assert [row["tailnet_devices"] for row in rows] == [[{"name": "Redmi Note 11", "online": True}],
                                                         [{"name": "web", "online": True}]]
    assert all(row["issues"] == [] for row in rows)


def test_a_running_machine_whose_device_is_offline_is_flagged_first():
    machines = {"truncated": False, "nodes": [{"node": "pve-b", "status": "online"}],
                "guests": [{"vmid": 103, "node": "pve-b", "type": "lxc", "name": "cache", "status": "stopped"},
                           {"vmid": 104, "node": "pve-b", "type": "qemu", "name": "db", "status": "running"}]}
    devices = {"count": 3, "truncated": False, "devices": [
        {"name": "pve-b", "dns_name": None, "online": False},
        {"name": "cache", "dns_name": None, "online": False},
        {"name": "db", "dns_name": None, "online": None}]}
    data = tailnet_coverage(machines, devices)
    # A node that is up while its device is offline: tailscaled is down there.
    assert data["nodes"][0]["issues"] == ["device_offline"]
    # A stopped guest's device is offline as expected; unknown online state is not an issue.
    assert [row["issues"] for row in data["guests"]] == [[], []]
    assert data["summary"]["device_offline"] == 1


def test_a_cut_listing_calls_nothing_missing_from_the_tailnet():
    data = tailnet_coverage(MACHINES, dict(DEVICES, truncated=True))
    assert data["complete"] is False
    assert data["guests"][0]["issues"] == [] and data["guests"][0]["tailnet_devices"] == []
    assert tailnet_coverage(dict(MACHINES, truncated=True), DEVICES)["complete"] is False


def test_an_unnamed_guest_cannot_match_and_says_so():
    machines = {"truncated": False, "nodes": [],
                "guests": [{"vmid": 105, "node": "pve-a", "type": "qemu", "name": None, "status": "running"}]}
    assert tailnet_coverage(machines, DEVICES)["guests"][0]["issues"] == ["no_tailnet_device"]


def test_machines_reads_nodes_and_guests_from_one_cluster_read_without_templates():
    calls = []

    def handler(request):
        calls.append((request.url.path, dict(request.url.params)))
        return httpx.Response(200, json={"data": [
            {"type": "node", "node": "pve-a", "status": "online", "id": "node/pve-a"},
            {"type": "qemu", "vmid": 101, "node": "pve-a", "name": "web", "status": "running"},
            {"type": "qemu", "vmid": 9000, "node": "pve-a", "name": "tmpl", "status": "stopped", "template": 1},
            {"type": "lxc", "vmid": 990, "node": "pve-a", "name": "janus-lab", "status": "running"},
            {"type": "storage", "storage": "local", "node": "pve-a", "status": "available"},
        ]}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", "reporter@pam!readonly", "secret"), client)
    data = service.machines()["data"]
    assert calls == [("/api2/json/cluster/resources", {})]
    assert data == {"truncated": False, "nodes": [{"node": "pve-a", "status": "online"}],
                    "guests": [{"vmid": 101, "node": "pve-a", "type": "qemu", "name": "web", "status": "running"},
                               {"vmid": 990, "node": "pve-a", "type": "lxc", "name": "janus-lab",
                                "status": "running"}]}
    assert MAX_ROWS >= 1


@pytest.mark.parametrize("row", [
    {"type": "qemu", "vmid": "101", "node": "pve-a", "status": "running"},
    {"type": "qemu", "vmid": 101, "status": "running"},
    {"type": "node", "status": "online"},
])
def test_machines_refuses_a_malformed_cluster_answer(row):
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [row]}, request=request)))
    service = ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", "reporter@pam!readonly", "secret"), client)
    with pytest.raises(RuntimeError, match="invalid answer"):
        service.machines()
