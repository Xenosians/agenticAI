"""The new OpenWrt and tailnet reads are registered tools: read-only, on their provider, used by the networking specialist."""
from pathlib import Path

import pytest

from services.integration_readiness import provider_for_capability
from tools.registry import TOOLS

NEW_READS = ("network_router_health", "network_interfaces", "network_known_hosts", "network_dns_config")
TAILNET_READS = ("network_tailnet_devices", "network_tailnet_dns", "network_tailnet_policy",
                 "network_tailnet_posture", "network_tailnet_coverage")
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


def test_an_empty_listing_says_it_is_empty_instead_of_ending_on_a_colon():
    # A reading is synthesis evidence: "0 known host(s):" invites a model to
    # fill the list. An empty result states that nothing was found.
    def read(tool, data):
        return TOOLS[tool]["result_formatter"]({"ok": True, "data": data})

    assert read("network_known_hosts", {"count": 0, "truncated": False, "hosts": []}) == (
        "The router knows no hosts.")
    assert read("network_interfaces", {"count": 0, "interfaces": []}) == (
        "The router reports no logical interfaces.")
    assert read("network_tailnet_devices", {"count": 0, "truncated": False,
                                            "summary": {"online": 0, "offline": 0, "with_issues": 0},
                                            "devices": []}) == "No device is on the tailnet."


def _health_device(**changes):
    device = {"name": "hummingbird", "dns_name": "hummingbird.tail1234.ts.net", "os": "linux",
              "client_version": "1.104.4", "update_available": False, "online": True, "last_seen": None,
              "key_expires": "2027-05-01T00:00:00Z", "key_expiry_disabled": False, "key_expires_in_days": 179,
              "authorized": True, "shared_from_another_tailnet": False, "tags": [], "blocks_incoming": False,
              "routes_advertised": 0, "routes_waiting_for_approval": 0, "exit_node_offered": False,
              "tailnet_lock_error": False,
              "connectivity": {"udp": True, "hard_nat": False, "port_mapping": False,
                               "nearest_relay": "Singapore", "relay_latency_ms": 27.2},
              "issues": []}
    return {**device, **changes}


def test_tailnet_device_health_reads_in_words():
    text = TOOLS["network_tailnet_devices"]["result_formatter"]({"ok": True, "data": {
        "count": 3, "truncated": False, "summary": {"online": 2, "offline": 1, "with_issues": 2}, "devices": [
            _health_device(),
            _health_device(name="kiosk", blocks_incoming=True, tailnet_lock_error=True,
                           connectivity={"udp": False, "hard_nat": True, "port_mapping": None,
                                         "nearest_relay": "Tokyo", "relay_latency_ms": 90.1},
                           issues=["tailnet_lock_error", "blocks_incoming", "udp_blocked"]),
            _health_device(name="new-server", online=False, last_seen="2026-11-01T00:00:00Z", authorized=False,
                           routes_advertised=3, routes_waiting_for_approval=2, exit_node_offered=True,
                           connectivity=None,
                           issues=["waiting_for_approval", "routes_waiting_for_approval"])]}})
    assert text.splitlines() == [
        "3 device(s) on the tailnet: 2 online, 1 offline; 2 need attention.",
        "hummingbird (linux, Tailscale 1.104.4): online; key expires in 179 days; direct connections possible, "
        "nearest relay Singapore 27.2 ms",
        "kiosk (linux, Tailscale 1.104.4): online; key expires in 179 days; tailnet lock reports a problem; "
        "blocks incoming connections; UDP blocked, so traffic goes through a relay; "
        "hard NAT (more connections relayed); nearest relay Tokyo 90.1 ms",
        "new-server (linux, Tailscale 1.104.4): offline since 2026-11-01; key expires in 179 days; "
        "waiting for approval; 2 of 3 advertised routes waiting for approval; offers itself as an exit node",
    ]


def test_a_key_that_lapsed_today_reads_as_expired_not_as_expiring_in_0_days():
    text = TOOLS["network_tailnet_devices"]["result_formatter"]({"ok": True, "data": {
        "count": 1, "truncated": False, "summary": {"online": 0, "offline": 1, "with_issues": 1}, "devices": [
            _health_device(name="tablet", online=False, last_seen="2026-11-02T01:00:00Z", key_expires_in_days=0,
                           connectivity=None, issues=["key_expired"])]}})
    assert text.splitlines()[1] == "tablet (linux, Tailscale 1.104.4): offline since 2026-11-02; key expired"
    assert text.splitlines()[0] == "1 device(s) on the tailnet: 0 online, 1 offline; 1 needs attention."


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


def test_the_tailnet_policy_reads_in_words_with_users_and_addresses_masked():
    formatter = TOOLS["network_tailnet_policy"]["result_formatter"]
    data = {"schema": "tailnet-policy.v1", "externally_managed": False, "truncated": False,
            "rules": [{"kind": "grant", "sources": ["autogroup:member"], "destinations": ["*"], "ports": ["*"],
                       "app_capabilities": 0},
                      {"kind": "acl", "sources": ["group:ops", "a user"], "destinations": ["tag:db", "an address"],
                       "ports": ["5432", "22"], "app_capabilities": 0}],
            "ssh": [{"action": "check", "sources": ["autogroup:member"], "destinations": ["autogroup:self"],
                     "users": ["autogroup:nonroot", "root"]}],
            "tag_owners": [{"tag": "tag:janus-lab", "owners": ["autogroup:admin"]}],
            "groups": [{"name": "group:ops", "members": 2}], "hosts": 1, "tests": 2, "other_sections": ["nodeAttrs"],
            "summary": {"rules": 2, "ssh_rules": 1, "tags": 1, "groups": 1, "open_to_all_members": True}}
    lines = formatter({"ok": True, "data": data}).splitlines()
    assert lines == [
        "Tailnet policy: 2 access rule(s), 1 SSH rule(s), 1 tag(s), 1 group(s), 1 host alias(es), 2 policy test(s); "
        "edited in the Tailscale admin console.",
        "Every member can reach every device on every port (a rule allows autogroup:member to * on all ports).",
        "Access: grant autogroup:member -> * (all ports); acl group:ops, a user -> tag:db:5432, an address:22.",
        "SSH: check autogroup:member -> autogroup:self as autogroup:nonroot, root.",
        "Tag owners: tag:janus-lab by autogroup:admin.",
        "Groups: group:ops (2 members).",
        "Also set: nodeAttrs.",
        "Users, host aliases and addresses are shown as \"a user\", \"a host alias\" and \"an address\".",
    ]
    git = dict(data, externally_managed=True, other_sections=[], groups=[], hosts=0,
               summary=dict(data["summary"], groups=0, open_to_all_members=False))
    text = formatter({"ok": True, "data": git})
    assert "managed outside the admin console (GitOps)" in text and "Every member can reach" not in text
    assert formatter({"ok": False}) == "Tailscale read did not succeed."


def test_device_posture_reads_in_words_and_names_what_needs_attention():
    formatter = TOOLS["network_tailnet_posture"]["result_formatter"]
    devices = [{"name": "Blackhawk", "os": "windows", "os_version": "10.0.26300.9974", "tailscale_version": "1.104.4",
                "auto_update": True, "release_track": "stable", "state_encrypted": False, "other_attributes": 0},
               {"name": "janus-router-lab", "os": "linux", "os_version": "6.18.58.1-microsoft-standard-WSL2",
                "tailscale_version": "1.104.4", "auto_update": False, "release_track": "stable",
                "state_encrypted": False, "other_attributes": 2},
               {"name": "old-phone", "os": None, "os_version": None, "tailscale_version": None, "auto_update": None,
                "release_track": None, "state_encrypted": None, "other_attributes": None}]
    text = formatter({"ok": True, "data": {"schema": "tailnet-posture.v1", "count": 3, "truncated": False,
                                           "devices": devices,
                                           "summary": {"auto_update_off": 1, "state_not_encrypted": 2}}})
    # node:tsStateEncrypted says whether the device stores its Tailscale state (keys) encrypted at rest.
    assert text.splitlines() == [
        "Posture of 3 tailnet device(s): 1 with Tailscale auto-update off, 2 storing Tailscale state unencrypted.",
        "Blackhawk: windows 10.0.26300.9974, Tailscale 1.104.4 (stable); auto-update on; "
        "Tailscale state stored unencrypted.",
        "janus-router-lab: linux 6.18.58.1-microsoft-standard-WSL2, Tailscale 1.104.4 (stable); auto-update off; "
        "Tailscale state stored unencrypted; 2 other attribute(s).",
        "old-phone: posture could not be read.",
    ]


def test_policy_and_posture_are_grounded_reads_in_the_networking_specialist():
    definition = (ROOT / "subagents/agents/networking-specialist.md").read_text(encoding="utf-8")
    for name in ("network_tailnet_policy", "network_tailnet_posture"):
        tool = TOOLS[name]
        assert (tool["effect"], tool["requires_approval"], tool["parameters"]) == ("read", False, {})
        assert tool["synthesis_evidence"] == "answer"
        assert f"  - {name}\n" in definition
        assert provider_for_capability(name) == "tailscale"


def test_tailnet_coverage_joins_proxmox_and_the_tailnet_and_names_each_sides_failure():
    from tools.networking.mcp import register_networking_tools

    class Tailnet:
        def devices(self):
            return {"ok": True, "status": "success", "provider": "tailscale", "data": {
                "count": 2, "truncated": False, "devices": [
                    {"name": "hummingbird", "dns_name": "hummingbird.tail1234.ts.net", "online": True},
                    {"name": "Blackhawk", "dns_name": "blackhawk.tail1234.ts.net", "online": True}]}}

    class Proxmox:
        def machines(self):
            return {"ok": True, "status": "success", "provider": "proxmox", "data": {
                "truncated": False, "nodes": [{"node": "hummingbird", "status": "online"}],
                "guests": [{"vmid": 990, "node": "hummingbird", "type": "lxc", "name": "janus-lab",
                            "status": "running"}]}}

    class Refusing:
        def machines(self):
            raise RuntimeError("Proxmox refused the read (HTTP 403).")

    server = _Server()
    register_networking_tools(server, None, Tailnet(), proxmox=Proxmox())
    result = server.tools["network_tailnet_coverage"]()
    assert (result.ok, result.provider, result.data["schema"]) == (True, "tailscale", "tailnet-coverage.v1")
    assert result.data["summary"]["guests_with_device"] == 0
    text = TOOLS["network_tailnet_coverage"]["result_formatter"](result.model_dump())
    assert text.splitlines() == [
        "Tailnet coverage, matched by name: 1 of 1 Proxmox node(s) and 0 of 1 guest(s) have a tailnet device "
        "named like them.",
        "hummingbird (node, online): tailnet device hummingbird (online).",
        "janus-lab (LXC 990 on hummingbird, running): no tailnet device named like it.",
        "Tailnet devices named like no Proxmox node or guest: Blackhawk.",
        "A machine may be on the tailnet under another name; names are the only link Janus reads.",
    ]
    refused = _Server()
    register_networking_tools(refused, None, Tailnet(), proxmox=Refusing())
    failed = refused.tools["network_tailnet_coverage"]()
    assert (failed.ok, failed.provider, failed.error_code) == (False, "proxmox", "provider_error")
    assert failed.error == "Proxmox refused the read (HTTP 403)."
    unconfigured = _Server()
    register_networking_tools(unconfigured, None, Tailnet(), proxmox=None,
                              missing_settings={"proxmox": ["PROXMOX_BASE_URL"]})
    missing = unconfigured.tools["network_tailnet_coverage"]()
    assert (missing.provider, missing.error_code) == ("proxmox", "not_configured")
    assert "PROXMOX_BASE_URL" in missing.error


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
