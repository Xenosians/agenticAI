"""Catalog and setup-state coverage for the Proxmox tool family."""
from config.settings import Settings
from services.integration_readiness import integration_readiness, provider_for_capability
from tools.proxmox.catalog import PROXMOX_TOOLS
from tools.proxmox.mcp import register_proxmox_tools
from tools.registry import TOOLS

POWER_TOOLS = {"proxmox_guest_start", "proxmox_guest_shutdown", "proxmox_guest_reboot", "proxmox_guest_force_stop"}


def test_each_proxmox_read_is_registered_with_read_only_semantics_and_one_provider():
    reads = {
        "proxmox_cluster_status", "proxmox_nodes", "proxmox_guests",
        "proxmox_guest_status", "proxmox_guest_config",
        "proxmox_node_storage", "proxmox_recent_tasks", "proxmox_database_inventory",
    }
    assert set(PROXMOX_TOOLS) == reads | POWER_TOOLS
    assert set(PROXMOX_TOOLS) <= set(TOOLS)
    for name in PROXMOX_TOOLS:
        assert provider_for_capability(name) == "proxmox"
    for name in reads:
        assert TOOLS[name]["effect"] == "read"
        assert TOOLS[name]["requires_approval"] is False
    for name in POWER_TOOLS:
        assert TOOLS[name]["effect"] == "mutation"
        assert TOOLS[name]["requires_approval"] is True


def test_proxmox_setup_reports_only_missing_setting_names():
    row = next(item for item in integration_readiness(Settings(_env_file=None)) if item["id"] == "proxmox")
    assert row["state"] == "not_configured"
    assert set(row["missing"]) == {"PROXMOX_BASE_URL", "PROXMOX_API_TOKEN_ID", "PROXMOX_API_TOKEN_SECRET"}
    assert row["configured"] is False


def test_mcp_registers_only_the_fixed_reads_and_unconfigured_calls_fail_honestly():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    server = Server()
    register_proxmox_tools(server, None, missing_settings=("PROXMOX_BASE_URL",))
    assert set(server.tools) == set(PROXMOX_TOOLS)
    result = server.tools["proxmox_nodes"]()
    assert result.ok is False and result.error_code == "not_configured"
    assert result.error == "Proxmox VE is not set up on this Janus host. Missing settings: PROXMOX_BASE_URL."
    assert server.tools["proxmox_node_storage"]("pve-a").error_code == "not_configured"


def test_a_failed_proxmox_read_says_why_instead_of_leaving_an_unknown_outcome():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    class Refusing:
        def nodes(self):
            raise RuntimeError("Proxmox refused the read (HTTP 403).")

    server = Server()
    register_proxmox_tools(server, Refusing())
    result = server.tools["proxmox_nodes"]()
    assert (result.ok, result.status, result.error_code) == (False, "error", "provider_error")
    assert result.error == "Proxmox refused the read (HTTP 403)."


def test_the_database_inventory_is_a_read_without_arguments_in_the_specialist():
    from pathlib import Path

    tool = TOOLS["proxmox_database_inventory"]
    assert (tool["effect"], tool["risk"], tool["requires_approval"]) == ("read", "read", False)
    assert tool["parameters"] == {} and tool["grounded_arguments"] == []
    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    assert "  - proxmox_database_inventory\n" in definition


def database_result(**changes):
    data = {"schema": "proxmox-database-inventory.v1", "count": 3, "truncated": False,
            "summary": {"running": 2, "not_running": 1, "without_backup_job": 1, "with_issues": 2},
            "backup_coverage_known": True, "backup_coverage_note": None, "label_hint": None,
            "databases": [
                {"vmid": 103, "node": "pve-a", "type": "lxc", "name": "reports", "engine": None, "status": "stopped",
                 "lock": None, "memory_bytes": 0, "memory_total_bytes": 1073741824, "disk_bytes": None,
                 "disk_total_bytes": 8589934592, "uptime_seconds": None, "backup_job": True, "issues": ["not_running"]},
                {"vmid": 102, "node": "pve-a", "type": "lxc", "name": "cache", "engine": "redis", "status": "running",
                 "lock": None, "memory_bytes": 536870912, "memory_total_bytes": 1073741824,
                 "disk_bytes": 1020054733, "disk_total_bytes": 1073741824, "uptime_seconds": 600,
                 "backup_job": False, "issues": ["disk_nearly_full", "no_backup_job"]},
                {"vmid": 101, "node": "pve-a", "type": "qemu", "name": "orders-db", "engine": "postgres",
                 "status": "running", "lock": None, "memory_bytes": 3221225472, "memory_total_bytes": 4294967296,
                 "disk_bytes": None, "disk_total_bytes": 34359738368, "uptime_seconds": 86400, "backup_job": True,
                 "issues": []}],
            "storage": [{"node": "pve-a", "storage": "local-lvm", "type": "lvmthin", "shared": False,
                         "status": "available", "used_bytes": 85899345920, "total_bytes": 107374182400}]}
    data.update(changes)
    return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
            "data": data}


def test_the_database_inventory_reads_in_words():
    formatter = TOOLS["proxmox_database_inventory"]["result_formatter"]
    assert formatter(database_result()).splitlines() == [
        "3 database host(s) tagged in Proxmox: 2 running, 1 not running; 1 without a backup job.",
        "reports (LXC 103 on pve-a): stopped; memory 0 B of 1.0 GiB; disk 8.0 GiB (usage not measured); "
        "backup job: yes. Needs attention: not running.",
        "cache (LXC 102 on pve-a, redis): running; memory 512 MiB of 1.0 GiB; disk 973 MiB of 1.0 GiB (95%); "
        "backup job: no. Needs attention: disk nearly full, no backup job.",
        "orders-db (VM 101 on pve-a, postgres): running; memory 3.0 GiB of 4.0 GiB; disk 32.0 GiB "
        "(usage not measured); backup job: yes.",
        "Guest disk storage: local-lvm on pve-a (lvmthin) 80.0 GiB of 100.0 GiB used (80%).",
    ]
    unknown = formatter(database_result(backup_coverage_known=False,
                                        backup_coverage_note="Proxmox refused the backup coverage read (HTTP 403); "
                                                             "the token needs Sys.Audit on / to see backup jobs.",
                                        summary={"running": 2, "not_running": 1, "without_backup_job": None,
                                                 "with_issues": 1}))
    assert unknown.splitlines()[0] == ("3 database host(s) tagged in Proxmox: 2 running, 1 not running; backup "
                                       "coverage unknown (Proxmox refused the backup coverage read (HTTP 403); "
                                       "the token needs Sys.Audit on / to see backup jobs.)")
    empty = formatter(database_result(count=0, databases=[], storage=[],
                                      summary={"running": 0, "not_running": 0, "without_backup_job": 0,
                                               "with_issues": 0},
                                      label_hint="Tag a guest db or db-<engine> (for example db-postgres) in "
                                                 "Proxmox to list it here."))
    assert empty == ("No Proxmox guest is tagged as a database host. Tag a guest db or db-<engine> (for example "
                     "db-postgres) in Proxmox to list it here.")
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def test_the_mcp_tool_reads_the_inventory_and_fails_honestly_when_unconfigured():
    class Server:
        tools = {}

        def tool(self):
            return lambda function: self.tools.setdefault(function.__name__, function)

    class Inventory:
        def database_inventory(self):
            return database_result()

    server = Server()
    register_proxmox_tools(server, Inventory())
    result = server.tools["proxmox_database_inventory"]()
    assert result.ok is True and result.data["count"] == 3
    unconfigured = Server()
    unconfigured.tools = {}
    register_proxmox_tools(unconfigured, None, missing_settings=("PROXMOX_BASE_URL",))
    assert unconfigured.tools["proxmox_database_inventory"]().error_code == "not_configured"
