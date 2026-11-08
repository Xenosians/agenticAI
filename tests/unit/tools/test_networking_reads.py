"""The new OpenWrt and tailnet reads are registered tools: read-only, on their provider, used by the networking specialist."""
from pathlib import Path

import pytest

from services.integration_readiness import provider_for_capability
from tools.registry import TOOLS

NEW_READS = ("network_router_health", "network_interfaces", "network_known_hosts", "network_dns_config")
TAILNET_READS = ("network_tailnet_devices", "network_tailnet_dns")
ROOT = Path(__file__).resolve().parents[3]


def test_the_dns_integrity_check_is_a_read_on_its_own_provider():
    tool = TOOLS["network_dns_integrity_check"]
    assert tool["effect"] == "read" and tool["risk"] == "read" and tool["requires_approval"] is False
    assert set(tool["parameters"]) == {"names"} and tool["grounded_arguments"] == ["names"]
    assert provider_for_capability("network_dns_integrity_check") == "dns_check"
    definition = (ROOT / "subagents/agents/networking-specialist.md").read_text(encoding="utf-8")
    assert "  - network_dns_integrity_check\n" in definition


def test_dns_integrity_reads_in_words():
    text = TOOLS["network_dns_integrity_check"]["result_formatter"]({"ok": True, "data": {
        "resolver": "1.1.1.1", "names_checked": 4, "rewritten": ["vimeo.com"], "system_follows_rewrite": [],
        "system_resolver_protected": True, "names": [
            {"name": "example.com", "plain": ["104.20.23.154"], "system": ["104.20.23.154"],
             "encrypted": ["104.20.23.154"], "verdict": "consistent", "system_resolver": "matches_encrypted",
             "errors": []},
            {"name": "github.com", "plain": ["20.205.243.166"], "system": ["140.82.121.3"],
             "encrypted": ["20.205.243.166"], "verdict": "consistent", "system_resolver": "different_answer",
             "errors": []},
            {"name": "vimeo.com", "plain": ["36.86.63.185"], "system": ["162.159.128.61"],
             "encrypted": ["162.159.128.61"], "verdict": "plain_dns_rewritten", "system_resolver": "matches_encrypted",
             "errors": []},
            {"name": "example.org", "plain": None, "system": ["96.7.128.198"], "encrypted": ["96.7.128.198"],
             "verdict": "unknown", "system_resolver": "matches_encrypted", "errors": ["plain DNS: no answer in time"]}]}})
    assert text.splitlines() == [
        "DNS check of 4 name(s) through 1.1.1.1: plain DNS answers were rewritten for 1 (vimeo.com). "
        "This host's own resolver gave the true answers, so the services here are not affected.",
        "example.com: consistent (104.20.23.154)",
        "github.com: consistent (20.205.243.166); this host's resolver gives 140.82.121.3 "
        "(another resolver's view, as for geo-routed names)",
        "vimeo.com: plain DNS rewritten to 36.86.63.185; DNS-over-HTTPS gives 162.159.128.61; "
        "this host's resolver gives 162.159.128.61",
        "example.org: could not be checked (plain DNS: no answer in time)",
    ]
    clean = TOOLS["network_dns_integrity_check"]["result_formatter"]({"ok": True, "data": {
        "resolver": "1.1.1.1", "names_checked": 1, "rewritten": [], "system_follows_rewrite": [],
        "system_resolver_protected": None, "names": [
            {"name": "example.com", "plain": ["104.20.23.154"], "system": ["104.20.23.154"],
             "encrypted": ["104.20.23.154"], "verdict": "consistent", "system_resolver": "matches_encrypted",
             "errors": []}]}})
    assert clean.splitlines()[0] == "DNS check of 1 name(s) through 1.1.1.1: no rewritten answers."
    exposed = TOOLS["network_dns_integrity_check"]["result_formatter"]({"ok": True, "data": {
        "resolver": "1.1.1.1", "names_checked": 1, "rewritten": ["vimeo.com"], "system_follows_rewrite": ["vimeo.com"],
        "system_resolver_protected": False, "names": [
            {"name": "vimeo.com", "plain": [], "system": [], "encrypted": ["162.159.128.61"],
             "verdict": "plain_dns_rewritten", "system_resolver": "follows_rewrite", "errors": []}]}})
    lines = exposed.splitlines()
    assert "This host's own resolver also returned rewritten answers" in lines[0]
    assert lines[1] == ("vimeo.com: plain DNS rewritten to no address (name not found); DNS-over-HTTPS gives "
                        "162.159.128.61; this host's resolver gives no address (name not found)")


@pytest.mark.parametrize("name", NEW_READS + TAILNET_READS)
def test_each_new_read_is_a_registered_read_without_approval(name):
    tool = TOOLS[name]
    assert tool["effect"] == "read" and tool["risk"] == "read" and tool["requires_approval"] is False
    assert tool["parameters"] == {} and tool["grounded_arguments"] == []
    assert provider_for_capability(name) == ("tailscale" if name in TAILNET_READS else "openwrt")


def test_the_networking_specialist_can_use_them():
    definition = (ROOT / "subagents/agents/networking-specialist.md").read_text(encoding="utf-8")
    for name in NEW_READS + TAILNET_READS:
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


def test_tailnet_devices_read_in_words():
    text = TOOLS["network_tailnet_devices"]["result_formatter"]({"ok": True, "data": {
        "count": 2, "truncated": False, "devices": [
            {"name": "janus-router-lab", "dns_name": "janus-router-lab.tail1234.ts.net", "os": "linux",
             "client_version": "1.104.4", "update_available": False, "online": True, "last_seen": None,
             "key_expires": None, "key_expiry_disabled": True, "key_expires_in_days": None, "authorized": True,
             "shared_from_another_tailnet": False, "tags": ["tag:janus-lab"]},
            {"name": "laptop", "dns_name": "laptop.tail1234.ts.net", "os": "windows", "client_version": "1.90.1",
             "update_available": True, "online": False, "last_seen": "2026-10-28T09:00:00Z",
             "key_expires": "2026-11-05T00:00:00Z", "key_expiry_disabled": False, "key_expires_in_days": 3,
             "authorized": True, "shared_from_another_tailnet": False, "tags": []}]}})
    assert text.splitlines() == [
        "2 device(s) on the tailnet:",
        "janus-router-lab (linux, Tailscale 1.104.4): online; key never expires; tags tag:janus-lab",
        "laptop (windows, Tailscale 1.90.1): offline since 2026-10-28; key expires in 3 days; update available",
    ]


def test_tailnet_dns_reads_in_words():
    text = TOOLS["network_tailnet_dns"]["result_formatter"]({"ok": True, "data": {
        "magic_dns": True, "override_local_dns": True,
        "nameservers": [{"address": "1.1.1.1", "encrypted": True}, {"address": "10.0.0.53", "encrypted": False}],
        "split_dns": [{"domain": "corp.example", "nameservers": ["10.0.0.53"]}], "search_domains": []}})
    assert text == ("Tailnet DNS: MagicDNS on; devices use the tailnet's resolvers instead of the local network's: "
                    "1.1.1.1 (encrypted, DNS-over-HTTPS), 10.0.0.53 (plain DNS); corp.example via 10.0.0.53.")
    local = TOOLS["network_tailnet_dns"]["result_formatter"]({"ok": True, "data": {
        "magic_dns": False, "override_local_dns": False, "nameservers": [], "split_dns": [], "search_domains": []}})
    assert local == "Tailnet DNS: MagicDNS off; devices use the local network's resolvers."


class _Server:
    def __init__(self):
        self.tools = {}

    def tool(self):
        return lambda function: self.tools.setdefault(function.__name__, function)


def test_unconfigured_network_providers_answer_with_what_is_missing():
    from tools.networking.mcp import register_networking_tools

    server = _Server()
    register_networking_tools(server, None, None, None, missing_settings={
        "openwrt": ["OPENWRT_BASE_URL"], "tailscale": ["TAILSCALE_CLIENT_ID"]})
    router = server.tools["network_dhcp_leases"]()
    assert (router.ok, router.provider, router.error_code) == (False, "openwrt", "not_configured")
    assert router.error == "The OpenWrt router is not set up on this Janus host. Missing settings: OPENWRT_BASE_URL."
    tailnet = server.tools["network_tailnet_devices"]()
    assert (tailnet.provider, tailnet.error_code) == ("tailscale", "not_configured")
    assert server.tools["network_dns_integrity_check"](["example.com"]).error_code == "not_configured"


def test_a_router_that_drops_the_connection_is_read_again_once():
    import httpx
    from tools.networking.mcp import register_networking_tools

    class Router:
        calls = 0

        def dhcp_leases(self):
            Router.calls += 1
            if Router.calls == 1:
                raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
            return {"ok": True, "status": "success", "provider": "openwrt", "data": {"count": 0, "leases": []}}

    server = _Server()
    register_networking_tools(server, Router())
    assert server.tools["network_dhcp_leases"]().ok is True and Router.calls == 2
