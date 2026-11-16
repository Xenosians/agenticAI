"""Node network interfaces on Proxmox, read-only.

Interfaces come from /nodes/{node}/network: name, type, whether Proxmox reports
it active, whether it starts at boot, how IPv4/IPv6 are configured, and the
ports of each bridge or bond. Addresses, netmasks, gateways, comments and
alternative names never leave the service: an alternative name can embed the
adapter's MAC address ("wlx94e7...").

Proxmox reads `active` from the interfaces file it last parsed, so it can lag
the real link (Proxmox bug 8148); the reading says "reported active", and an
interface Proxmox reports nothing about is "not reported", never down.
"""
import json

import httpx
import pytest

from services.proxmox import ProxmoxConfig, ProxmoxService

TOKEN_ID = "reporter@pam!readonly"
TOKEN_SECRET = "fixture-token-secret"

# The live hummingbird shape (PVE 9.2), with documentation-range values.
NETWORK = [
    {"iface": "vmbr0", "type": "bridge", "method": "static", "method6": "manual", "families": ["inet"],
     "autostart": 1, "priority": 4, "bridge_ports": "nic0", "bridge_stp": "off", "bridge_fd": "0",
     "address": "192.0.2.14", "netmask": "24", "cidr": "192.0.2.14/24", "gateway": "192.0.2.1",
     "comments": "uplink on the switch port labelled hb-1\n"},
    {"iface": "nic0", "type": "eth", "method": "manual", "method6": "manual", "families": ["inet"], "priority": 3},
    {"iface": "nic1", "type": "eth", "method": "manual", "method6": "manual", "families": ["inet"], "priority": 5},
    {"iface": "wlp1s0", "type": "eth", "method": "manual", "method6": "manual", "families": ["inet"],
     "active": 1, "exists": 1, "altnames": ["wlx94e70bc2f27c"]},
    {"iface": "bond0", "type": "bond", "method": "manual", "method6": "manual", "families": ["inet"],
     "active": 0, "exists": 1, "autostart": 1, "slaves": "nic1 nic2", "bond_mode": "active-backup", "mtu": 9000},
]


def service(routes, calls=None):
    def handler(request):
        if calls is not None:
            calls.append((request.method, request.url.path, dict(request.url.params)))
        status, data = routes[request.url.path]
        return httpx.Response(status, json={"data": data}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ProxmoxService(ProxmoxConfig("https://pve.example.test:8006", TOKEN_ID, TOKEN_SECRET), client), client


def test_node_network_lists_interfaces_and_bridge_ports_without_addresses_comments_or_altnames():
    calls = []
    client_service, client = service({"/api2/json/nodes/pve-a/network": (200, NETWORK)}, calls)
    try:
        result = client_service.node_network("pve-a")
    finally:
        client.close()
    assert calls == [("GET", "/api2/json/nodes/pve-a/network", {})]
    data = result["data"]
    assert data["schema"] == "proxmox-node-network.v1" and data["node"] == "pve-a"
    rows = {row["name"]: row for row in data["interfaces"]}
    assert rows["vmbr0"] == {"name": "vmbr0", "type": "bridge", "active": None, "autostart": True,
                             "ipv4": "static", "ipv6": "manual", "has_ipv4_address": True, "ports": ["nic0"],
                             "bond_mode": None, "mtu": None, "vlan_aware": None, "member_of": []}
    assert rows["bond0"]["ports"] == ["nic1", "nic2"] and rows["bond0"]["bond_mode"] == "active-backup"
    assert rows["bond0"]["active"] is False and rows["bond0"]["mtu"] == 9000
    assert rows["wlp1s0"]["active"] is True and rows["wlp1s0"]["has_ipv4_address"] is False
    # Bridges and bonds first, then physical ports; each group by name.
    assert [row["name"] for row in data["interfaces"]] == ["vmbr0", "bond0", "nic0", "nic1", "wlp1s0"]
    # Which bridge or bond each port belongs to, so "nic0 is the uplink of vmbr0" needs no inference.
    assert rows["nic0"]["member_of"] == ["vmbr0"] and rows["nic1"]["member_of"] == ["bond0"]
    assert rows["wlp1s0"]["member_of"] == []
    assert data["summary"] == {"interfaces": 5, "bridges": 1, "bonds": 1, "physical": 3,
                               "reported_active": 1, "reported_inactive": 1, "not_reported": 3}
    text = json.dumps(result)
    for secret in ("192.0.2", "hb-1", "wlx94e70bc2f27c", "gateway", "netmask", "cidr", "comments", "altnames"):
        assert secret not in text, secret


def test_a_port_named_by_a_bridge_but_not_listed_is_still_named_and_marked_missing():
    rows = [{"iface": "vmbr1", "type": "bridge", "method": "manual", "method6": "manual", "families": ["inet"],
             "autostart": 1, "bridge_ports": "enp9s0"}]
    client_service, client = service({"/api2/json/nodes/pve-a/network": (200, rows)})
    try:
        data = client_service.node_network("pve-a")["data"]
    finally:
        client.close()
    bridge = data["interfaces"][0]
    assert bridge["ports"] == ["enp9s0"] and data["missing_ports"] == ["enp9s0"]


def test_a_bridge_without_ports_and_a_vlan_aware_bridge_read_plainly():
    rows = [{"iface": "vmbr9", "type": "bridge", "method": "manual", "method6": "manual", "families": ["inet"],
             "autostart": 1, "bridge_ports": "none", "bridge_vlan_aware": 1}]
    client_service, client = service({"/api2/json/nodes/pve-a/network": (200, rows)})
    try:
        bridge = client_service.node_network("pve-a")["data"]["interfaces"][0]
    finally:
        client.close()
    assert bridge["ports"] == [] and bridge["vlan_aware"] is True


@pytest.mark.parametrize("node", ["../etc", "", "pve a", 5])
def test_only_an_exact_node_name_is_read(node):
    calls = []
    client_service, client = service({}, calls)
    try:
        with pytest.raises(ValueError, match="node"):
            client_service.node_network(node)
    finally:
        client.close()
    assert calls == []


@pytest.mark.parametrize("change", [{"iface": ""}, {"iface": 5}, {"type": 3}, {"active": 2}, {"autostart": "yes"},
                                    {"bridge_ports": 4}, {"mtu": "9000"}, {"method": ["static"]}])
def test_a_malformed_interface_is_refused(change):
    row = dict(NETWORK[0], **change)
    client_service, client = service({"/api2/json/nodes/pve-a/network": (200, [row])})
    try:
        with pytest.raises(RuntimeError, match="invalid answer"):
            client_service.node_network("pve-a")
    finally:
        client.close()
