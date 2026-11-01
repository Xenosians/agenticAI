"""More OpenWrt reads: router health, every interface with its routes and DNS, known hosts.

Each is one fixed read through the existing scoped ubus session. Nothing can change
the router, and nothing the router returns is passed back without being normalized.
"""
import json

import httpx
import pytest

from services.openwrt import MAX_KNOWN_HOSTS, OpenWrtConfig, OpenWrtService


def service(responses):
    calls = []

    def handler(request):
        params = json.loads(request.content)["params"]
        calls.append(params)
        if params[1] == "session":
            data = {"ubus_rpc_session": "a" * 32}
        else:
            data = responses[(params[1], params[2])]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": [0, data]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenWrtService(OpenWrtConfig("https://router.example", "reader", "test-secret"), client), calls


def test_router_health_reports_uptime_load_and_memory_in_plain_units():
    svc, calls = service({("system", "info"): {
        "localtime": 1793500000, "uptime": 93784, "load": [13107, 6553, 0],
        "memory": {"total": 268435456, "free": 134217728, "available": 201326592, "shared": 1, "buffered": 2, "cached": 3},
        "root": {"total": 102400, "free": 51200, "used": 51200, "avail": 51200}, "swap": {"total": 0, "free": 0},
    }})
    result = svc.router_health()
    assert result["ok"] is True
    assert result["data"] == {
        "uptime_seconds": 93784, "load_average": [0.2, 0.1, 0.0],
        "memory_total_mib": 256.0, "memory_available_mib": 192.0, "memory_used_percent": 25.0,
        "root_filesystem_used_percent": 50.0,
    }
    assert [call[1:3] for call in calls] == [["session", "login"], ["system", "info"]]
    assert "test-secret" not in json.dumps(result)


def test_interfaces_lists_every_logical_interface_with_addresses_routes_and_dns():
    svc, _ = service({("network.interface", "dump"): {"interface": [
        {"interface": "lan", "up": True, "proto": "static", "l3_device": "br-lan", "uptime": 128277,
         "ipv4-address": [{"address": "192.168.1.1", "mask": 24}],
         "ipv6-address": [], "ipv6-prefix-assignment": [{"address": "fd77::", "mask": 60}],
         "route": [{"target": "0.0.0.0", "mask": 0, "nexthop": "192.168.1.254", "source": "0.0.0.0/0"}],
         "dns-server": ["192.168.1.254"], "dns-search": ["lan"], "inactive": {"secret": "ignored"}},
        {"interface": "wan", "up": False, "proto": "dhcp", "pending": True},
    ]}})
    result = svc.interfaces()
    assert result["data"]["count"] == 2
    lan, wan = result["data"]["interfaces"]
    assert lan == {"name": "lan", "up": True, "protocol": "static", "device": "br-lan", "uptime_seconds": 128277,
                   "ipv4": ["192.168.1.1/24"], "ipv6": [], "routes": ["0.0.0.0/0 via 192.168.1.254"],
                   "dns_servers": ["192.168.1.254"], "dns_search": ["lan"]}
    assert wan == {"name": "wan", "up": False, "protocol": "dhcp", "device": None, "uptime_seconds": None,
                   "ipv4": [], "ipv6": [], "routes": [], "dns_servers": [], "dns_search": []}


def test_known_hosts_merges_names_and_addresses_and_is_bounded():
    hints = {f"AA:BB:CC:00:00:{index:02X}": {"name": f"host{index}", "ipaddrs": [f"192.168.1.{index}"],
                                              "ip6addrs": [f"fd00::{index}"]}
             for index in range(MAX_KNOWN_HOSTS + 3)}
    hints["not-a-mac"] = {"name": "ignored"}
    svc, _ = service({("luci-rpc", "getHostHints"): hints})
    result = svc.known_hosts()
    data = result["data"]
    assert data["count"] == MAX_KNOWN_HOSTS + 3 and data["truncated"] is True
    assert len(data["hosts"]) == MAX_KNOWN_HOSTS
    assert data["hosts"][0] == {"mac": "aa:bb:cc:00:00:00", "name": "host0", "ipv4": ["192.168.1.0"],
                                "ipv6": ["fd00::0"]}
    assert result["scope"] == "router_neighbour_table_and_leases"


@pytest.mark.parametrize("method", ["router_health", "interfaces", "known_hosts"])
def test_malformed_router_answers_are_refused_not_reported(method):
    svc, _ = service({("system", "info"): {"uptime": "soon"}, ("network.interface", "dump"): {"interface": "lan"},
                      ("luci-rpc", "getHostHints"): ["not", "a", "map"]})
    with pytest.raises(RuntimeError):
        getattr(svc, method)()


def test_dns_config_reports_resolver_protections_and_upstream_servers():
    svc, calls = service({("uci", "get"): {"values": {"cfg01411c": {
        ".type": "dnsmasq", ".name": "cfg01411c", "domain": "lan", "rebind_protection": "1", "domainneeded": "1",
        "boguspriv": "1", "localservice": "1", "cachesize": "1000", "noresolv": "1",
        "server": ["127.0.0.1#5053", "/corp.example/10.0.0.53"], "dnssec": "1"}}}})
    result = svc.dns_config()
    assert calls[-1][1:] == ["uci", "get", {"config": "dhcp", "type": "dnsmasq"}]
    assert result["data"] == {"count": 1, "resolvers": [{
        "domain": "lan", "uses_wan_resolvers": False, "upstream_servers": ["127.0.0.1#5053", "/corp.example/10.0.0.53"],
        "rebind_protection": True, "dnssec": True, "drops_unqualified_names": True,
        "drops_private_reverse_lookups": True, "answers_local_networks_only": True, "cache_size": 1000}]}
    assert result["scope"] == "configured_dns_settings; not live resolution"


def test_dns_config_leaves_unset_settings_unknown_and_defaults_to_wan_resolvers():
    svc, _ = service({("uci", "get"): {"values": {"a": {".type": "dnsmasq", "cachesize": "lots", "server": "9.9.9.9"}}}})
    resolver = svc.dns_config()["data"]["resolvers"][0]
    assert resolver["uses_wan_resolvers"] is True
    assert resolver["upstream_servers"] == ["9.9.9.9"]
    assert resolver["rebind_protection"] is None and resolver["dnssec"] is None and resolver["cache_size"] is None


def test_dns_config_without_values_is_refused():
    svc, _ = service({("uci", "get"): {"config": "dhcp"}})
    with pytest.raises(RuntimeError):
        svc.dns_config()
