import json

import httpx

from services.openwrt import MAX_DHCP_LEASES, OpenWrtConfig, OpenWrtService


def service(responses):
    calls = []

    def handler(request):
        params = json.loads(request.content)["params"]
        calls.append(params)
        if params[1] == "session":
            data = {"ubus_rpc_session": "a" * 32}
        else:
            data = responses[(params[1], params[2], json.dumps(params[3], sort_keys=True))]
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": [0, data]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenWrtService(OpenWrtConfig("https://router.example", "reader", "test-secret"), client), calls


def test_dhcp_leases_normalizes_both_families_without_credentials():
    svc, calls = service({("luci-rpc", "getDHCPLeases", "{}"): {
        "dhcp_leases": [{"hostname": "laptop", "ipaddr": "192.168.1.20", "macaddr": "aa:bb:cc:dd:ee:ff", "expires": 3600}],
        "dhcp6_leases": [{"hostname": "phone", "ip6addrs": ["fd00::20/128"], "duid": "0001", "expires": 120}],
    }})
    result = svc.dhcp_leases()
    assert result["ok"] is True and result["scope"] == "dhcp_server_leases"
    assert result["data"]["count"] == 2 and result["data"]["truncated"] is False
    assert result["data"]["leases"][0] == {"family": "ipv4", "hostname": "laptop", "address": "192.168.1.20",
                                           "mac": "aa:bb:cc:dd:ee:ff", "duid": None, "expires_in_seconds": 3600}
    assert result["data"]["leases"][1]["address"] == "fd00::20/128"
    assert "test-secret" not in json.dumps(result)
    assert [call[1:3] for call in calls] == [["session", "login"], ["luci-rpc", "getDHCPLeases"]]


def test_dhcp_leases_are_bounded():
    many = [{"hostname": f"h{i}", "ipaddr": f"10.0.{i // 250}.{i % 250}", "macaddr": "aa:aa:aa:aa:aa:aa"}
            for i in range(MAX_DHCP_LEASES + 5)]
    svc, _ = service({("luci-rpc", "getDHCPLeases", "{}"): {"dhcp_leases": many, "dhcp6_leases": []}})
    data = svc.dhcp_leases()["data"]
    assert len(data["leases"]) == MAX_DHCP_LEASES and data["count"] == MAX_DHCP_LEASES + 5 and data["truncated"]


def test_wireless_clients_per_interface_in_one_session():
    svc, calls = service({
        ("iwinfo", "devices", "{}"): {"devices": ["phy0-ap0", "bad;name"]},
        ("iwinfo", "assoclist", json.dumps({"device": "phy0-ap0"})): {"results": [
            {"mac": "11:22:33:44:55:66", "signal": -51, "noise": -95, "inactive": 40,
             "rx": {"rate": 144400}, "tx": {"rate": 72200}}]},
    })
    result = svc.wireless_clients()
    assert result["data"]["count"] == 1
    assert result["data"]["interfaces"] == [{"device": "phy0-ap0", "count": 1, "truncated": False, "clients": [
        {"mac": "11:22:33:44:55:66", "signal_dbm": -51, "noise_dbm": -95, "inactive_ms": 40,
         "rx_rate_kbps": 144400, "tx_rate_kbps": 72200}]}]
    # One login for all reads; the invalid device name never reaches the router.
    assert sum(1 for call in calls if call[1] == "session") == 1
    assert all(call[3].get("device") != "bad;name" for call in calls)


def test_router_without_radios_reports_explicit_empty_result():
    svc, _ = service({("iwinfo", "devices", "{}"): {"devices": []}})
    result = svc.wireless_clients()
    assert result["ok"] is True
    assert result["data"] == {"interfaces": [], "count": 0}
    assert result["note"] == "The router reports no wireless interfaces."
