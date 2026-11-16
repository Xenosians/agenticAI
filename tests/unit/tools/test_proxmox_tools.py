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
        "proxmox_backups", "proxmox_node_disks", "proxmox_guest_snapshots", "proxmox_node_network",
        "proxmox_ha_status", "proxmox_disk_smart", "proxmox_node_maintenance",
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


def test_guest_snapshots_is_a_grounded_read_in_the_specialist_that_says_it_changes_nothing():
    from pathlib import Path

    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    tool = TOOLS["proxmox_guest_snapshots"]
    assert (tool["effect"], tool["requires_approval"], tool["grounded_arguments"]) == ("read", False, ["node", "vmid"])
    assert tool["synthesis_evidence"] == "answer"
    assert "  - proxmox_guest_snapshots\n" in definition
    assert "never creates, deletes or rolls back" in tool["description"]


def test_guest_snapshots_read_in_words():
    formatter = TOOLS["proxmox_guest_snapshots"]["result_formatter"]
    data = {"schema": "proxmox-guest-snapshots.v1", "node": "pve-a", "vmid": 101, "type": "qemu",
            "name": "orders-db", "count": 3, "truncated": False, "current_parent": "after-upgrade",
            "oldest_age_days": 40.0,
            "snapshots": [
                {"name": "after-upgrade", "time": 2, "age_days": 3.0, "parent": "before-upgrade",
                 "includes_ram": False},
                {"name": "before-upgrade", "time": 1, "age_days": 40.0, "parent": None, "includes_ram": True},
                {"name": "legacy", "time": None, "age_days": None, "parent": None, "includes_ram": None}]}
    result = {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
              "data": data}
    assert formatter(result).splitlines() == [
        "orders-db (VM 101 on node pve-a) has 3 snapshot(s); the current state follows snapshot after-upgrade.",
        "after-upgrade: taken 3.0 days ago, parent before-upgrade, disk only.",
        "before-upgrade: taken 40.0 days ago, with RAM.",
        "legacy: time not recorded.",
    ]
    empty = dict(data, count=0, current_parent=None, oldest_age_days=None, snapshots=[], type="lxc", vmid=990,
                 name="janus-lab")
    assert formatter(dict(result, data=empty)) == "janus-lab (LXC 990 on node pve-a) has no snapshots."
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def test_guest_snapshots_is_served_over_mcp_and_reports_missing_setup():
    class Server:
        def __init__(self):
            self.tools = {}

        def tool(self):
            def register(function):
                self.tools[function.__name__] = function
                return function
            return register

    class Service:
        def guest_snapshots(self, node, vmid):
            return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "now",
                    "data": {"node": node, "vmid": vmid}}

    server = Server()
    register_proxmox_tools(server, Service(), missing_settings=())
    assert server.tools["proxmox_guest_snapshots"]("pve-a", 101).data == {"node": "pve-a", "vmid": 101}
    unconfigured = Server()
    register_proxmox_tools(unconfigured, None, missing_settings=("PROXMOX_BASE_URL",))
    assert unconfigured.tools["proxmox_guest_snapshots"]("pve-a", 101).error_code == "not_configured"


def test_node_network_is_a_grounded_read_in_the_specialist_that_never_shows_addresses():
    from pathlib import Path

    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    tool = TOOLS["proxmox_node_network"]
    assert (tool["effect"], tool["requires_approval"], tool["grounded_arguments"]) == ("read", False, ["node"])
    assert tool["synthesis_evidence"] == "answer"
    assert "  - proxmox_node_network\n" in definition
    assert "never shows IP addresses" in tool["description"]


def test_node_network_reads_in_words():
    formatter = TOOLS["proxmox_node_network"]["result_formatter"]
    interfaces = [
        {"name": "vmbr0", "type": "bridge", "active": None, "autostart": True, "ipv4": "static", "ipv6": "manual",
         "has_ipv4_address": True, "ports": ["nic0"], "bond_mode": None, "mtu": None, "vlan_aware": None,
         "member_of": []},
        {"name": "bond0", "type": "bond", "active": False, "autostart": True, "ipv4": "manual", "ipv6": "manual",
         "has_ipv4_address": False, "ports": ["nic1", "nic2"], "bond_mode": "active-backup", "mtu": 9000,
         "vlan_aware": None, "member_of": []},
        {"name": "nic0", "type": "eth", "active": None, "autostart": None, "ipv4": "manual", "ipv6": "manual",
         "has_ipv4_address": False, "ports": [], "bond_mode": None, "mtu": None, "vlan_aware": None,
         "member_of": ["vmbr0"]},
        {"name": "wlp1s0", "type": "eth", "active": True, "autostart": None, "ipv4": "manual", "ipv6": "manual",
         "has_ipv4_address": False, "ports": [], "bond_mode": None, "mtu": None, "vlan_aware": None,
         "member_of": []}]
    data = {"schema": "proxmox-node-network.v1", "node": "pve-a", "count": 4, "truncated": False,
            "summary": {"interfaces": 4, "bridges": 1, "bonds": 1, "physical": 2, "reported_active": 1,
                        "reported_inactive": 1, "not_reported": 2},
            "missing_ports": ["nic2"], "interfaces": interfaces}
    result = {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
              "data": data}
    assert formatter(result).splitlines() == [
        "4 network interface(s) on node pve-a: 1 bridge(s), 1 bond(s), 2 physical.",
        "vmbr0 (bridge): ports nic0; IPv4 static with an address; starts at boot.",
        "bond0 (bond, active-backup): ports nic1, nic2; MTU 9000; starts at boot; Proxmox reports it inactive.",
        "nic0 (physical): port of vmbr0.",
        "wlp1s0 (physical): Proxmox reports it active.",
        "nic2 is named as a port but is not configured on this node.",
        "Proxmox reports an interface's state from its last read of the network configuration, so it can lag "
        "the link; an interface without a report is not known to be down.",
    ]
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def test_a_missed_scheduled_run_reads_in_words_with_its_rule_apart():
    formatter = TOOLS["proxmox_backups"]["result_formatter"]
    missed = {"vmid": 104, "node": "pve-a", "type": "qemu", "name": "web", "backup_job": True, "backups": 3,
              "last_scheduled_run": 1_794_728_000,
              "latest": {"storage": "pbs", "time": 1, "age_days": 2.0, "size_bytes": None, "protected": False,
                         "encrypted": False, "verified": None},
              "issues": ["missed_run"]}
    text = formatter(backups_result(
        count=1, guests=[missed],
        summary={"guests": 1, "with_recent_backup": 0, "stale": 0, "missed_run": 1, "never": 0,
                 "without_backup_job": 0}))
    lines = text.splitlines()
    assert lines[0] == ("1 guest(s); 1 backup job (1 enabled). 0 with a recent backup, 1 missed its last scheduled "
                        "run, 0 overdue, 0 never backed up, 0 without a backup job. A guest in a backup job missed a "
                        "run when its newest copy is older than the job's last run due at least 12 hours ago; other "
                        "guests are overdue when the newest copy is more than 8 days old.")
    assert lines[1] == ("web (VM 104 on node pve-a): newest backup 2.0 days ago on storage pbs; 3 backups. "
                        "Missed its last scheduled run.")
    two = formatter(backups_result(
        count=2, guests=[missed, dict(missed, vmid=105, name="api")],
        summary={"guests": 2, "with_recent_backup": 0, "stale": 0, "missed_run": 2, "never": 0,
                 "without_backup_job": 0}))
    assert "0 with a recent backup, 2 missed their last scheduled run, 0 overdue" in two.splitlines()[0]


def test_copies_only_on_storage_that_cannot_verify_get_one_plain_hint():
    # Proxmox verifies backups only on Proxmox Backup Server storage; a copy on a
    # directory, NFS or CIFS storage is never verified, so "not verified" there
    # is not a failure and the reading says why once.
    formatter = TOOLS["proxmox_backups"]["result_formatter"]
    local_only = backups_result(storages=[{"node": "pve-a", "storage": "local", "type": "dir", "shared": False,
                                           "backups": 3}])
    hint = ("Every backup sits on storage that cannot verify backups (only Proxmox Backup Server storage "
            "verifies them), so no copy can show as verified.")
    assert formatter(local_only).splitlines()[-1] == hint
    assert hint not in formatter(backups_result())  # PBS holds the copies there
    empty = backups_result(storages=[{"node": "pve-a", "storage": "local", "type": "dir", "shared": False,
                                      "backups": 0}])
    assert hint not in formatter(empty)  # nothing to verify


def test_disk_smart_is_a_grounded_read_in_the_specialist():
    from pathlib import Path

    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    tool = TOOLS["proxmox_disk_smart"]
    assert (tool["effect"], tool["requires_approval"], tool["grounded_arguments"]) == ("read", False, ["node", "disk"])
    assert tool["synthesis_evidence"] == "answer"
    assert "  - proxmox_disk_smart\n" in definition


def test_disk_smart_reads_in_words():
    formatter = TOOLS["proxmox_disk_smart"]["result_formatter"]
    measures = {"temperature_celsius": 30, "power_on_hours": 7686, "power_cycles": 2712, "unsafe_shutdowns": 95,
                "media_errors": 0, "available_spare_percent": 100, "available_spare_threshold_percent": 10,
                "reallocated_sectors": None, "pending_sectors": None, "offline_uncorrectable": None}
    data = {"schema": "proxmox-disk-smart.v1", "node": "hummingbird", "device": "/dev/nvme0n1", "type": "nvme",
            "model": "INTEL SSDPEKNW512G8", "health": "passed", "wear_used_percent": 12, "measures": measures,
            "issues": []}
    result = {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
              "data": data}
    assert formatter(result).splitlines() == [
        "/dev/nvme0n1 on node hummingbird (nvme, INTEL SSDPEKNW512G8): SMART passed; 12% of rated life used.",
        "Temperature 30 °C; powered on 7686 hours (320 days); 2712 power cycles; 95 unsafe shutdowns; "
        "0 media errors; spare capacity 100% (threshold 10%).",
    ]
    failing = dict(data, device="/dev/sda", type="hdd", model=None, wear_used_percent=None,
                   measures=dict(measures, reallocated_sectors=8, pending_sectors=2, unsafe_shutdowns=None,
                                 media_errors=None, available_spare_percent=None,
                                 available_spare_threshold_percent=None, offline_uncorrectable=0),
                   issues=["reallocated_sectors", "pending_sectors"])
    lines = formatter(dict(result, data=failing)).splitlines()
    assert lines[0] == "/dev/sda on node hummingbird (hdd): SMART passed."
    assert lines[1] == ("Temperature 30 °C; powered on 7686 hours (320 days); 2712 power cycles; "
                        "8 reallocated sectors; 2 pending sectors; 0 offline uncorrectable sectors.")
    assert lines[2] == ("Early signs of failure, even though SMART reports passed: reallocated sectors, "
                        "pending sectors.")
    # Media errors and low spare capacity are named as what they are, not as moved sectors.
    nvme_failing = dict(data, measures=dict(measures, media_errors=3, available_spare_percent=8),
                        issues=["media_errors", "spare_below_threshold"])
    assert formatter(dict(result, data=nvme_failing)).splitlines()[-1] == (
        "Early signs of failure, even though SMART reports passed: media errors, spare capacity below its "
        "threshold.")
    assert formatter(dict(result, data=dict(nvme_failing, health="failed"))).splitlines()[-1] == (
        "Needs attention: media errors, spare capacity below its threshold.")
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def test_node_maintenance_is_a_grounded_read_in_the_specialist():
    from pathlib import Path

    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    tool = TOOLS["proxmox_node_maintenance"]
    assert (tool["effect"], tool["requires_approval"], tool["grounded_arguments"]) == ("read", False, ["node"])
    assert tool["synthesis_evidence"] == "answer"
    assert "  - proxmox_node_maintenance\n" in definition
    assert provider_for_capability("proxmox_node_maintenance") == "proxmox"


MAINTENANCE = {
    "schema": "proxmox-node-maintenance.v1", "node": "hummingbird", "version": "9.2.27",
    "kernel": {"running": "7.0.14-28-pve", "newest_installed": "7.0.14-29-pve", "installed": 3},
    "boot": {"mode": "efi", "secure_boot": True}, "in_cluster": False,
    "services": {"installed": 21, "running": 20, "not_running": [], "expected_stopped": ["corosync"],
                 "turned_off": [], "not_installed": 2},
    "certificates": [
        {"file": "pve-ssl.pem", "role": "node", "served": True, "issuer": "proxmox_ca", "key": "RSA 2048",
         "days_left": 725},
        {"file": "pve-root-ca.pem", "role": "cluster_ca", "served": False, "issuer": "self_signed",
         "key": "RSA 4096", "days_left": 3645}],
    "subscription": "none",
    "repositories": {"enabled": ["no-subscription", "ceph-squid-no-subscription"], "turned_off": ["enterprise"],
                     "errors": 0},
    "issues": ["reboot_for_newer_kernel"], "unreadable": [],
}


def test_node_maintenance_reads_in_words():
    formatter = TOOLS["proxmox_node_maintenance"]["result_formatter"]
    result = {"ok": True, "status": "success", "provider": "proxmox", "data": MAINTENANCE}
    assert formatter(result).splitlines() == [
        "Node hummingbird runs Proxmox VE 9.2.27 on kernel 7.0.14-28-pve; boots EFI with Secure Boot on.",
        "Needs attention: kernel 7.0.14-29-pve is installed but not running; a reboot starts it.",
        "Services: 20 of 21 running; corosync stopped as expected (the node is not in a cluster).",
        "Certificates: pve-ssl.pem (served to browsers, signed by the Proxmox cluster CA, RSA 2048) expires in "
        "725 days; pve-root-ca.pem (cluster CA, self-signed, RSA 4096) expires in 3645 days.",
        "Updates come from: no-subscription, ceph-squid-no-subscription; turned off: enterprise; no subscription.",
        "Pending package updates are not read: that needs a privilege Janus does not hold.",
    ]
    failing = dict(MAINTENANCE, kernel=dict(MAINTENANCE["kernel"], newest_installed="7.0.14-28-pve"),
                   boot={"mode": "legacy", "secure_boot": None}, in_cluster=None,
                   services=dict(MAINTENANCE["services"], running=18, expected_stopped=[], turned_off=["spiceproxy"],
                                 not_running=[{"name": "corosync", "state": "inactive"},
                                              {"name": "pvestatd", "state": "failed"}]),
                   certificates=[dict(MAINTENANCE["certificates"][0], days_left=-1),
                                 dict(MAINTENANCE["certificates"][1], days_left=12)],
                   subscription=None, repositories={"enabled": ["enterprise", "test"], "turned_off": [], "errors": 1},
                   issues=["service_not_running", "certificate_expired", "certificate_expiring",
                           "test_repository_enabled"], unreadable=["subscription"])
    lines = formatter(dict(result, data=failing)).splitlines()
    assert lines[0] == "Node hummingbird runs Proxmox VE 9.2.27 on kernel 7.0.14-28-pve; boots legacy BIOS."
    assert lines[1] == ("Needs attention: corosync inactive, pvestatd failed; a served certificate has expired; "
                        "a certificate expires within 30 days; the test repository is on.")
    assert lines[2] == "Services: 18 of 21 running; not running: corosync (inactive), pvestatd (failed); turned off: spiceproxy."
    assert lines[3] == ("Certificates: pve-ssl.pem (served to browsers, signed by the Proxmox cluster CA, RSA 2048) "
                        "expired 1 day ago; pve-root-ca.pem (cluster CA, self-signed, RSA 4096) expires in 12 days.")
    assert lines[4] == "Updates come from: enterprise, test; 1 repository error."
    assert lines[5] == "Not readable with Janus's access: subscription."
    nothing = dict(MAINTENANCE, issues=[], kernel=dict(MAINTENANCE["kernel"], newest_installed="7.0.14-28-pve"))
    assert formatter(dict(result, data=nothing)).splitlines()[1] == "Nothing needs attention."
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def test_empty_proxmox_readings_say_they_are_empty():
    def read(tool, data):
        return TOOLS[tool]["result_formatter"]({"ok": True, "status": "success", "provider": "proxmox",
                                                "observed_at": "2026-11-16T06:00:00+00:00", "data": data})

    assert read("proxmox_node_network", {
        "schema": "proxmox-node-network.v1", "node": "pve-a", "count": 0, "truncated": False,
        "summary": {"interfaces": 0, "bridges": 0, "bonds": 0, "physical": 0, "reported_active": 0,
                    "reported_inactive": 0, "not_reported": 0},
        "missing_ports": [], "interfaces": []}) == "Proxmox reports no network interface on node pve-a."
    assert read("proxmox_node_disks", {
        "node": "pve-a", "count": 0, "truncated": False,
        "summary": {"disks": 0, "failing": 0, "worn": 0, "health_unknown": 0},
        "disks": []}) == "Proxmox lists no physical disk on node pve-a."


def test_ha_status_is_a_read_in_the_specialist_that_never_changes_ha():
    from pathlib import Path

    definition = Path("subagents/agents/proxmox-specialist.md").read_text(encoding="utf-8")
    tool = TOOLS["proxmox_ha_status"]
    assert (tool["effect"], tool["requires_approval"], tool["grounded_arguments"]) == ("read", False, [])
    assert tool["synthesis_evidence"] == "answer"
    assert "  - proxmox_ha_status\n" in definition
    assert "never changes" in tool["description"]


def ha_result(**changes):
    data = {"schema": "proxmox-ha-status.v1", "quorate": True, "fencing": "armed", "in_use": True,
            "manager": {"node": "pve-a", "state": "active", "problem": False},
            "nodes": [{"node": "pve-a", "state": "active", "problem": False},
                      {"node": "pve-b", "state": "not_responding", "problem": True}],
            "count": 2, "truncated": False,
            "resources": [{"sid": "ct:101", "type": "lxc", "vmid": 101, "node": "pve-b", "state": "error",
                           "requested": "started", "problem": True},
                          {"sid": "vm:100", "type": "qemu", "vmid": 100, "node": "pve-a", "state": "started",
                           "requested": "started", "problem": False}],
            "summary": {"nodes": 2, "nodes_with_problems": 1, "resources": 2, "resources_with_problems": 1}}
    data.update(changes)
    return {"ok": True, "status": "success", "provider": "proxmox", "observed_at": "2026-11-16T06:00:00+00:00",
            "data": data}


def test_ha_status_reads_in_words():
    formatter = TOOLS["proxmox_ha_status"]["result_formatter"]
    assert formatter(ha_result()).splitlines() == [
        "High availability is in use; the cluster has quorum. HA manager on pve-a: active. Fencing: armed.",
        "2 node(s) in HA, 1 with a problem: pve-a active; pve-b not responding.",
        "2 HA resource(s), 1 with a problem:",
        "LXC 101 on pve-b: error (requested started).",
        "VM 100 on pve-a: started.",
    ]


def test_ha_that_is_not_in_use_or_without_quorum_reads_plainly():
    formatter = TOOLS["proxmox_ha_status"]["result_formatter"]
    unused = ha_result(in_use=False, manager=None, nodes=[], resources=[], count=0, fencing="standby",
                       summary={"nodes": 0, "nodes_with_problems": 0, "resources": 0, "resources_with_problems": 0})
    assert formatter(unused) == ("High availability is not in use: no VM or container is an HA resource. "
                                 "The cluster has quorum.")
    assert formatter(ha_result(quorate=False)).splitlines()[0] == (
        "High availability is in use; the cluster has NO quorum. HA manager on pve-a: active. Fencing: armed.")
    assert formatter({"ok": False}) == "The Proxmox read did not succeed."


def backups_result(**changes):
    data = {"schema": "proxmox-backups.v1", "jobs": {"count": 1, "enabled": 1}, "coverage_known": True,
            "coverage_note": None, "listing_complete": True, "stale_after_days": 8, "missed_run_grace_hours": 12,
            "unreadable_storages": [],
            "summary": {"guests": 3, "with_recent_backup": 1, "stale": 1, "missed_run": 0, "never": 1,
                        "without_backup_job": 1},
            "storages": [{"node": "pve-a", "storage": "pbs", "type": "pbs", "shared": True, "backups": 3}],
            "count": 3, "truncated": False,
            "guests": [
                {"vmid": 103, "node": "pve-a", "type": "lxc", "name": "old", "backup_job": False, "backups": 0,
                 "last_scheduled_run": None, "latest": None, "issues": ["never_backed_up", "no_backup_job"]},
                {"vmid": 102, "node": "pve-a", "type": "lxc", "name": "cache", "backup_job": True, "backups": 1,
                 "last_scheduled_run": None,
                 "latest": {"storage": "pbs", "time": 1, "age_days": 12.0, "size_bytes": 536870912,
                            "protected": False, "encrypted": False, "verified": None},
                 "issues": ["backup_stale"]},
                {"vmid": 101, "node": "pve-a", "type": "qemu", "name": "orders-db", "backup_job": True,
                 "backups": 2, "last_scheduled_run": None,
                 "latest": {"storage": "pbs", "time": 2, "age_days": 0.5, "size_bytes": 2147483648,
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
