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
        "proxmox_backups", "proxmox_node_disks",
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


def test_backups_and_disks_are_reads_in_the_specialist():
    from pathlib import Path

    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    backups, disks = TOOLS["proxmox_backups"], TOOLS["proxmox_node_disks"]
    assert (backups["effect"], backups["requires_approval"], backups["parameters"]) == ("read", False, {})
    assert (disks["effect"], disks["requires_approval"], disks["grounded_arguments"]) == ("read", False, ["node"])
    for name in ("proxmox_backups", "proxmox_node_disks"):
        assert f"  - {name}\n" in definition
        assert TOOLS[name]["synthesis_evidence"] == "answer"


def backups_result(**changes):
    data = {"schema": "proxmox-backups.v1", "jobs": {"count": 1, "enabled": 1}, "coverage_known": True,
            "coverage_note": None, "listing_complete": True, "stale_after_days": 8, "unreadable_storages": [],
            "summary": {"guests": 3, "with_recent_backup": 1, "stale": 1, "never": 1, "without_backup_job": 1},
            "storages": [{"node": "pve-a", "storage": "pbs", "type": "pbs", "shared": True, "backups": 3}],
            "count": 3, "truncated": False,
            "guests": [
                {"vmid": 103, "node": "pve-a", "type": "lxc", "name": "old", "backup_job": False, "backups": 0,
                 "latest": None, "issues": ["never_backed_up", "no_backup_job"]},
                {"vmid": 102, "node": "pve-a", "type": "lxc", "name": "cache", "backup_job": True, "backups": 1,
                 "latest": {"storage": "pbs", "time": 1, "age_days": 12.0, "size_bytes": 536870912,
                            "protected": False, "encrypted": False, "verified": None},
                 "issues": ["backup_stale"]},
                {"vmid": 101, "node": "pve-a", "type": "qemu", "name": "orders-db", "backup_job": True,
                 "backups": 2, "latest": {"storage": "pbs", "time": 2, "age_days": 0.5, "size_bytes": 2147483648,
                                          "protected": True, "encrypted": True, "verified": "ok"},
                 "issues": []}]}
    data.update(changes)
    return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
            "data": data}


def test_the_database_inventory_leaves_backup_questions_to_the_backups_read():
    # Live, "Which of our Proxmox guests are overdue for a backup or have never
    # been backed up?" was routed to the database inventory, whose description
    # claimed "database servers, their backups and capacity"; it lists only
    # tagged hosts, so the answer invented "backup tracking relies on VM tags".
    inventory = TOOLS["proxmox_database_inventory"]["description"]
    backups = TOOLS["proxmox_backups"]["description"]
    assert "their backups" not in inventory
    assert "Use this only for database hosts" in inventory and "proxmox_backups" in inventory
    assert backups.startswith("Use this for any question about Proxmox backups")


def test_backups_read_in_words():
    formatter = TOOLS["proxmox_backups"]["result_formatter"]
    assert formatter(backups_result()).splitlines() == [
        "3 guest(s); 1 backup job (1 enabled). 1 with a recent backup, 1 overdue, 1 never backed up, "
        "1 without a backup job. A backup is overdue when the newest copy is more than 8 days old.",
        "old (LXC 103 on node pve-a): never backed up; no backup job.",
        "cache (LXC 102 on node pve-a): newest backup 12.0 days ago on storage pbs (512 MiB). Overdue.",
        "orders-db (VM 101 on node pve-a): newest backup 0.5 days ago on storage pbs (2.0 GiB, protected, "
        "encrypted, verified ok); 2 backups.",
        "Backup storage: pbs on node pve-a (pbs, shared) holds 3 backup(s).",
    ]
    # Live, with no backup at all, a definition beside the count ("0 overdue (newest older than 8 days)")
    # was restated as a fact ("the oldest backup is older than 8 days"); the rule now stands apart, and
    # an empty listing says no backup exists.
    none = formatter(backups_result(
        jobs={"count": 0, "enabled": 0}, count=1,
        summary={"guests": 1, "with_recent_backup": 0, "stale": 0, "never": 1, "without_backup_job": 1},
        storages=[{"node": "hummingbird", "storage": "local", "type": "dir", "shared": False, "backups": 0}],
        guests=[{"vmid": 990, "node": "hummingbird", "type": "lxc", "name": "janus-lab", "backup_job": False,
                 "backups": 0, "latest": None, "issues": ["never_backed_up", "no_backup_job"]}]))
    assert none.splitlines() == [
        "1 guest(s); no backup jobs. No backup exists on any backup storage that was read.",
        "janus-lab (LXC 990 on node hummingbird): never backed up; no backup job.",
        "Backup storage: local on node hummingbird (dir) holds 0 backup(s).",
    ]
    nowhere = formatter(backups_result(storages=[], guests=[], count=0,
                                       summary={"guests": 0, "with_recent_backup": 0, "stale": 0, "never": 0,
                                                "without_backup_job": 0}))
    assert nowhere.splitlines()[0] == "0 guest(s); 1 backup job (1 enabled). No storage is set up to hold backups."
    partial = formatter(backups_result(listing_complete=False, unreadable_storages=[
        {"node": "pve-a", "storage": "nas", "reason": "Proxmox refused the read (HTTP 403)."}],
        summary={"guests": 3, "with_recent_backup": 1, "stale": 0, "never": 0, "without_backup_job": 1}))
    assert ("Not every backup storage could be read (nas on node pve-a: Proxmox refused the read (HTTP 403).), "
            "so missing or old backups cannot be confirmed.") in partial
    unknown = formatter(backups_result(jobs=None, coverage_known=False,
                                       coverage_note="Proxmox refused the backup coverage read (HTTP 403)."))
    assert "backup jobs unknown (Proxmox refused the backup coverage read (HTTP 403).)" in unknown.splitlines()[0]
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def test_node_disks_read_in_words():
    formatter = TOOLS["proxmox_node_disks"]["result_formatter"]
    result = {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
              "data": {"node": "pve-a", "count": 3, "truncated": False,
                       "summary": {"disks": 3, "failing": 1, "worn": 1, "health_unknown": 0},
                       "disks": [
                           {"device": "/dev/sda", "type": "hdd", "model": "WDC WD40EFRX", "size_bytes": 4000787030016,
                            "health": "failed", "wear_used_percent": None, "used_by": "ZFS",
                            "issues": ["smart_failed"]},
                           {"device": "/dev/sdb", "type": "ssd", "model": None, "size_bytes": 1000204886016,
                            "health": "passed", "wear_used_percent": 92, "used_by": None, "issues": ["worn_out"]},
                           {"device": "/dev/nvme0n1", "type": "nvme", "model": "INTEL SSDPEKNW512G8",
                            "size_bytes": 512110190592, "health": "passed", "wear_used_percent": 12,
                            "used_by": "BIOS boot", "issues": []}]}}
    assert formatter(result).splitlines() == [
        "3 physical disk(s) on pve-a: 1 failing SMART, 1 worn out.",
        "/dev/sda (hdd, WDC WD40EFRX, 3.6 TiB, used by ZFS): SMART failed. Replace it.",
        "/dev/sdb (ssd, 931.5 GiB, unused): SMART passed; 92% of rated life used. Plan a replacement.",
        "/dev/nvme0n1 (nvme, INTEL SSDPEKNW512G8, 476.9 GiB, used by BIOS boot): SMART passed; 12% of rated "
        "life used.",
    ]
