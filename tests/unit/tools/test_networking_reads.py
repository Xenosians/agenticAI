"""The new OpenWrt reads are registered tools: read-only, on the OpenWrt provider, used by the networking specialist."""
from pathlib import Path

import pytest

from services.integration_readiness import provider_for_capability
from tools.registry import TOOLS

NEW_READS = ("network_router_health", "network_interfaces", "network_known_hosts", "network_dns_config")
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", NEW_READS)
def test_each_new_read_is_a_registered_read_without_approval(name):
    tool = TOOLS[name]
    assert tool["effect"] == "read" and tool["risk"] == "read" and tool["requires_approval"] is False
    assert tool["parameters"] == {} and tool["grounded_arguments"] == []
    assert provider_for_capability(name) == "openwrt"


def test_the_networking_specialist_can_use_them():
    definition = (ROOT / "subagents/agents/networking-specialist.md").read_text(encoding="utf-8")
    for name in NEW_READS:
        assert f"  - {name}\n" in definition


def test_results_read_as_plain_sentences_for_the_user():
    health = TOOLS["network_router_health"]["result_formatter"]({"ok": True, "data": {
        "uptime_seconds": 93784, "load_average": [0.2, 0.1, 0.0], "memory_total_mib": 256.0,
        "memory_available_mib": 192.0, "memory_used_percent": 25.0, "root_filesystem_used_percent": 50.0}})
    assert health.startswith("Router up 1 day 2 h 3 min; load 0.2 / 0.1 / 0.0; memory 25.0% used")
    interfaces = TOOLS["network_interfaces"]["result_formatter"]({"ok": True, "data": {"count": 1, "interfaces": [
        {"name": "lan", "up": True, "protocol": "static", "device": "br-lan", "uptime_seconds": 60,
         "ipv4": ["192.168.1.1/24"], "ipv6": [], "routes": ["0.0.0.0/0 via 192.168.1.254"],
         "dns_servers": ["192.168.1.254"], "dns_search": []}]}})
    assert "lan: up (static, br-lan) 192.168.1.1/24; routes 0.0.0.0/0 via 192.168.1.254; DNS 192.168.1.254" in interfaces
    hosts = TOOLS["network_known_hosts"]["result_formatter"]({"ok": True, "data": {"count": 1, "truncated": False,
        "hosts": [{"mac": "aa:bb:cc:dd:ee:ff", "name": "laptop", "ipv4": ["192.168.1.20"], "ipv6": []}]}})
    assert "laptop (aa:bb:cc:dd:ee:ff) 192.168.1.20" in hosts
    assert TOOLS["network_router_health"]["result_formatter"]({"ok": False}) == "Network provider read did not succeed."


def test_dns_settings_read_in_words():
    text = TOOLS["network_dns_config"]["result_formatter"]({"ok": True, "data": {"count": 1, "resolvers": [{
        "domain": "lan", "uses_wan_resolvers": True, "upstream_servers": ["1.1.1.1"], "rebind_protection": True,
        "dnssec": None, "drops_unqualified_names": True, "drops_private_reverse_lookups": True,
        "answers_local_networks_only": True, "cache_size": 1000}]}})
    assert text == ("DNS resolver (domain lan): upstream: the internet connection's resolvers, 1.1.1.1; "
                    "rebind protection on; DNSSEC not set; cache 1000 names.")
    empty = TOOLS["network_dns_config"]["result_formatter"]({"ok": True, "data": {"count": 0, "resolvers": []}})
    assert empty == "The router has no DNS resolver configured."
