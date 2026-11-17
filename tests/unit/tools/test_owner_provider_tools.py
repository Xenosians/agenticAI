"""Each tool call reads the Proxmox and Tailscale of the account it serves (owner decision, 17 November).

The MCP server registers its tools once, but every call resolves its service
for the requesting owner: that account's own link, else the machine default.
An account with neither is told to connect the provider in Connections.
"""
from contextlib import contextmanager

from services.extensions.context import current_owner, request_owner
from tools.networking.mcp import register_networking_tools
from tools.proxmox.mcp import register_proxmox_tools


class Server:
    def __init__(self):
        self.tools = {}

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function
        return register


def reading(provider, name):
    return {"ok": True, "status": "success", "provider": provider, "observed_at": "now", "data": {"from": name}}


class Proxmox:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def nodes(self):
        return reading("proxmox", self.name)

    def machines(self):
        return {"ok": True, "status": "success", "provider": "proxmox", "data": {
            "truncated": False, "nodes": [{"node": self.name, "status": "online"}], "guests": []}}

    def guest_power(self, node, vmid, action, *, expected_type, expected_name):
        self.calls.append((node, vmid, action))
        return {"ok": True, "status": "executed", "provider": "proxmox", "operation": "guest_power",
                "action": action, "node": node, "vmid": vmid, "type": expected_type, "name": expected_name,
                "expected_status": "running", "observed_status": "running", "upid": "UPID:x",
                "task_exit_status": "OK", "changed": True, "verification_ok": True, "retry_safe": False,
                "error": None, "observed_at": "now"}


class Tailnet:
    def __init__(self, name):
        self.name = name

    def devices(self):
        return {"ok": True, "status": "success", "provider": "tailscale", "observed_at": "now", "data": {
            "count": 1, "truncated": False, "devices": [{"name": self.name, "online": True}]}}


def resolver(services):
    @contextmanager
    def for_request():
        yield services.get(current_owner(), services.get(None))
    return for_request


def as_owner(owner, call):
    token = request_owner.set(owner)
    try:
        return call()
    finally:
        request_owner.reset(token)


def test_each_proxmox_read_uses_the_service_of_the_account_it_serves():
    server = Server()
    register_proxmox_tools(server, None, for_request=resolver({"alice": Proxmox("alice-pve"),
                                                                None: Proxmox("machine-pve")}))
    assert as_owner("alice", server.tools["proxmox_nodes"]).data == {"from": "alice-pve"}
    assert as_owner("bob", server.tools["proxmox_nodes"]).data == {"from": "machine-pve"}
    assert server.tools["proxmox_nodes"]().data == {"from": "machine-pve"}


def test_an_approved_power_action_runs_on_the_proposing_accounts_proxmox_only():
    alice, machine = Proxmox("alice-pve"), Proxmox("machine-pve")
    server = Server()
    register_proxmox_tools(server, None, for_request=resolver({"alice": alice, None: machine}))
    result = as_owner("alice", lambda: server.tools["proxmox_guest_start"](
        "pve-a", 991, expected_type="lxc", expected_name="lab"))
    assert result.ok is True
    assert alice.calls == [("pve-a", 991, "start")] and machine.calls == []


def test_an_account_with_no_proxmox_at_all_is_told_to_connect_it():
    server = Server()
    register_proxmox_tools(server, None, for_request=resolver({}))
    result = as_owner("bob", server.tools["proxmox_nodes"])
    assert (result.ok, result.error_code) == (False, "not_configured")
    assert "Proxmox link in Connections" in result.error
    power = as_owner("bob", lambda: server.tools["proxmox_guest_start"](
        "pve-a", 991, expected_type="lxc", expected_name="lab"))
    assert (power.ok, power.error_code) == (False, "not_configured")


def test_tailnet_reads_and_coverage_use_the_requesting_accounts_links():
    server = Server()
    register_networking_tools(server, None, missing_settings={},
                              tailnet_for_request=resolver({"alice": Tailnet("alice-laptop"),
                                                            None: Tailnet("machine-laptop")}),
                              proxmox_for_request=resolver({"alice": Proxmox("alice-pve"),
                                                            None: Proxmox("machine-pve")}))
    devices = as_owner("alice", server.tools["network_tailnet_devices"])
    assert devices.data["devices"][0]["name"] == "alice-laptop"
    assert server.tools["network_tailnet_devices"]().data["devices"][0]["name"] == "machine-laptop"
    coverage = as_owner("alice", server.tools["network_tailnet_coverage"])
    assert coverage.ok is True
    assert [row["node"] for row in coverage.data["nodes"]] == ["alice-pve"]


def test_an_account_with_no_tailnet_is_told_to_connect_it():
    server = Server()
    register_networking_tools(server, None, missing_settings={}, tailnet_for_request=resolver({}),
                              proxmox_for_request=resolver({}))
    result = as_owner("bob", server.tools["network_tailnet_devices"])
    assert (result.ok, result.error_code) == (False, "not_configured")
    assert "Tailscale link in Connections" in result.error
    coverage = as_owner("bob", server.tools["network_tailnet_coverage"])
    assert "Tailscale link in Connections" in coverage.error
