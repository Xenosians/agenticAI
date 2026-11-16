"""Proxmox VE capabilities: bounded reads and four approval-gated guest power actions."""
import json
from typing import Any


def format_read(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    return "Observed Proxmox data: " + json.dumps(result.get("data"), ensure_ascii=False)


def _bytes(value) -> str:
    for unit, size in (("TiB", 1024 ** 4), ("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if value >= size:
            return f"{value / size:.1f} {unit}" if unit in ("TiB", "GiB") else f"{value / size:.0f} {unit}"
    return f"{value:.0f} B"


def _share(used, total) -> str:
    return f"{_bytes(used)} of {_bytes(total)}" + (f" ({used / total:.0%})" if total else "")


def _used_share(used, total) -> str:
    return f"{_bytes(used)} of {_bytes(total)} used" + (f" ({used / total:.0%})" if total else "")


DATABASE_ISSUE_WORDS = {"not_running": "not running", "disk_nearly_full": "disk nearly full",
                        "no_backup_job": "no backup job", "locked": "locked"}


def format_database_inventory(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    if not data["databases"]:
        return "No Proxmox guest is tagged as a database host. " + (data.get("label_hint") or "")
    summary = data["summary"]
    head = (f"{data['count']} database host(s) tagged in Proxmox"
            + (" (list cut to the first 200)" if data["truncated"] else "")
            + f": {summary['running']} running, {summary['not_running']} not running; ")
    if data["backup_coverage_known"]:
        head += f"{summary['without_backup_job']} without a backup job."
    else:
        head += f"backup coverage unknown ({data['backup_coverage_note']})"
    lines = [head]
    for row in data["databases"]:
        about = f"{'VM' if row['type'] == 'qemu' else 'LXC'} {row['vmid']} on {row['node']}"
        name = row["name"] or "unnamed"
        text = f"{name} ({about}" + (f", {row['engine']}" if row["engine"] else "") + f"): {row['status']}"
        if row["lock"]:
            text += f" (locked: {row['lock']})"
        if row["memory_bytes"] is not None and row["memory_total_bytes"]:
            text += f"; memory {_bytes(row['memory_bytes'])} of {_bytes(row['memory_total_bytes'])}"
        if row["disk_total_bytes"]:
            text += ("; disk " + (_share(row["disk_bytes"], row["disk_total_bytes"]) if row["disk_bytes"] is not None
                                  else f"{_bytes(row['disk_total_bytes'])} (usage not measured)"))
        text += "; backup job: " + {True: "yes", False: "no", None: "unknown"}[row["backup_job"]] + "."
        if row["issues"]:
            text += " Needs attention: " + ", ".join(DATABASE_ISSUE_WORDS[issue] for issue in row["issues"]) + "."
        lines.append(text)
    if data["storage"]:
        lines.append("Guest disk storage: " + "; ".join(
            f"{pool['storage']} on {pool['node']}" + (f" ({pool['type']})" if pool["type"] else "")
            + (f" {_used_share(pool['used_bytes'], pool['total_bytes'])}"
               if pool["used_bytes"] is not None and pool["total_bytes"] else "")
            for pool in data["storage"]) + ".")
    return "\n".join(lines)


def _guest_on_node(row: dict) -> str:
    kind = "VM" if row["type"] == "qemu" else "LXC"
    return f"{row['name'] or 'unnamed'} ({kind} {row['vmid']} on node {row['node']})"


def format_backups(result: dict) -> str:
    # Each sentence states a fact; the overdue rule stands apart from the
    # counts, since a definition beside a count was once restated as a fact.
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    summary, jobs = data["summary"], data["jobs"]
    head = f"{data['count']} guest(s); "
    if jobs is None or not data["coverage_known"]:
        head += f"backup jobs unknown ({data['coverage_note'] or 'Proxmox did not report them'}). "
    elif jobs["count"] == 0:
        head += "no backup jobs. "
    else:
        head += f"{jobs['count']} backup job{'s' if jobs['count'] != 1 else ''} ({jobs['enabled']} enabled). "
    if not data["storages"] and not data["unreadable_storages"]:
        head += "No storage is set up to hold backups."
    elif not sum(row["backups"] for row in data["storages"]):
        head += "No backup exists on any backup storage that was read."
    else:
        missed = summary.get("missed_run", 0)
        head += (f"{summary['with_recent_backup']} with a recent backup, "
                 + (f"{missed} missed {'its' if missed == 1 else 'their'} last scheduled run, " if missed else "")
                 + f"{summary['stale']} overdue, {summary['never']} never backed up")
        head += (f", {summary['without_backup_job']} without a backup job."
                 if summary["without_backup_job"] is not None else ".")
        if missed:
            head += (f" A guest in a backup job missed a run when its newest copy is older than the job's last run "
                     f"due at least {data['missed_run_grace_hours']} hours ago; other guests are overdue when the "
                     f"newest copy is more than {data['stale_after_days']} days old.")
        else:
            head += f" A backup is overdue when the newest copy is more than {data['stale_after_days']} days old."
    lines = [head]
    if not data["listing_complete"]:
        lines.append("Not every backup storage could be read (" + "; ".join(
            f"{row['storage']} on node {row['node']}: {row['reason']}" for row in data["unreadable_storages"])
            + "), so missing or old backups cannot be confirmed.")
    for row in data["guests"]:
        latest = row["latest"]
        if latest is None:
            text = f"{_guest_on_node(row)}: " + ("never backed up" if "never_backed_up" in row["issues"]
                                               else "no backup found in the storages read")
        else:
            facts = [_bytes(latest["size_bytes"])] if latest["size_bytes"] is not None else []
            facts += [word for word, present in (("protected", latest["protected"]),
                                                 ("encrypted", latest["encrypted"])) if present]
            if latest["verified"]:
                facts.append(f"verified {latest['verified']}")
            text = (f"{_guest_on_node(row)}: newest backup {latest['age_days']:.1f} days ago on storage "
                    f"{latest['storage']}"
                    + (f" ({', '.join(facts)})" if facts else ""))
            if row["backups"] > 1:
                text += f"; {row['backups']} backups"
        text += "; no backup job." if "no_backup_job" in row["issues"] else "."
        if "missed_run" in row["issues"]:
            text += " Missed its last scheduled run."
        if "backup_stale" in row["issues"]:
            text += " Overdue."
        lines.append(text)
    if data["storages"]:
        lines.append("Backup storage: " + "; ".join(
            f"{row['storage']} on node {row['node']} ({row['type']}{', shared' if row['shared'] else ''}) holds "
            f"{row['backups']} backup(s)" for row in data["storages"]) + ".")
    holding = [row for row in data["storages"] if row["backups"]]
    if holding and all(row["type"] != "pbs" for row in holding):
        lines.append("Every backup sits on storage that cannot verify backups (only Proxmox Backup Server storage "
                     "verifies them), so no copy can show as verified.")
    return "\n".join(lines)


def format_guest_snapshots(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    if not data["snapshots"]:
        return f"{_guest_on_node(data)} has no snapshots."
    head = f"{_guest_on_node(data)} has {data['count']} snapshot(s)"
    head += (f"; the current state follows snapshot {data['current_parent']}." if data["current_parent"] else ".")
    lines = [head]
    for row in data["snapshots"]:
        facts = ["time not recorded" if row["age_days"] is None else f"taken {row['age_days']:.1f} days ago"]
        if row["parent"]:
            facts.append(f"parent {row['parent']}")
        if row["includes_ram"] is not None:
            facts.append("with RAM" if row["includes_ram"] else "disk only")
        lines.append(f"{row['name']}: " + ", ".join(facts) + ".")
    if data["truncated"]:
        lines.append("More snapshots exist than were listed.")
    return "\n".join(lines)


HA_STATE_WORDS = {"not_responding": "not responding", "clock_drift": "clock drift", "lost_lock": "lost its lock",
                  "waiting_for_lock": "waiting for its lock", "shutting_down": "shutting down"}


def format_ha_status(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    quorum = {True: "the cluster has quorum", False: "the cluster has NO quorum", None: "quorum was not reported"}
    if not data["in_use"]:
        return ("High availability is not in use: no VM or container is an HA resource. "
                + quorum[data["quorate"]].capitalize() + ".")
    head = f"High availability is in use; {quorum[data['quorate']]}."
    if data["manager"]:
        manager = data["manager"]
        head += f" HA manager on {manager['node']}: {HA_STATE_WORDS.get(manager['state'], manager['state'])}."
    if data["fencing"]:
        head += f" Fencing: {data['fencing']}."
    lines = [head]
    summary = data["summary"]
    if data["nodes"]:
        lines.append(f"{summary['nodes']} node(s) in HA, {summary['nodes_with_problems']} with a problem: "
                     + "; ".join(f"{row['node']} {HA_STATE_WORDS.get(row['state'], row['state'])}"
                                 for row in data["nodes"]) + ".")
    if data["resources"]:
        lines.append(f"{summary['resources']} HA resource(s), {summary['resources_with_problems']} with a problem:")
        for row in data["resources"]:
            line = f"{'VM' if row['type'] == 'qemu' else 'LXC'} {row['vmid']} on {row['node'] or 'no node'}: {row['state']}"
            if row["requested"] and row["requested"] != row["state"]:
                line += f" (requested {row['requested']})"
            lines.append(line + ".")
    if data["truncated"]:
        lines.append("More HA resources exist than were listed.")
    return "\n".join(lines)


MAINTENANCE_ISSUE_WORDS = {
    "certificate_expired": "a served certificate has expired",
    "certificate_expiring": "a certificate expires within 30 days",
    "enterprise_repository_without_subscription": "the enterprise repository is on without a subscription, so "
                                                  "updates from it fail",
    "no_update_repository": "no Proxmox update repository is on",
    "test_repository_enabled": "the test repository is on",
}
CERTIFICATE_ROLE_WORDS = {"node": "served to browsers", "cluster_ca": "cluster CA", "custom": "uploaded",
                          "other": "other"}
CERTIFICATE_ISSUER_WORDS = {"proxmox_ca": "signed by the Proxmox cluster CA", "self_signed": "self-signed",
                            "other_ca": "signed by another CA"}
SUBSCRIPTION_WORDS = {"active": "active subscription", "none": "no subscription", "expired": "subscription expired",
                      "invalid": "subscription invalid", "suspended": "subscription suspended",
                      "not_checked": "subscription not checked yet", "unknown": "subscription state unknown"}


def _days(days: int) -> str:
    if days < 0:
        return f"expired {-days} day{'s' if -days != 1 else ''} ago"
    return f"expires in {days} day{'s' if days != 1 else ''}"


def format_node_maintenance(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    kernel, boot = data["kernel"], data["boot"]
    head = f"Node {data['node']} runs Proxmox VE {data['version']}"
    if kernel["running"]:
        head += f" on kernel {kernel['running']}"
    if boot and boot["mode"]:
        head += "; boots " + ("EFI" if boot["mode"] == "efi" else "legacy BIOS" if boot["mode"] == "legacy-bios"
                              or boot["mode"] == "legacy" else boot["mode"])
        if boot["mode"] == "efi" and boot["secure_boot"] is not None:
            head += " with Secure Boot " + ("on" if boot["secure_boot"] else "off")
    lines = [head + "."]
    services = data["services"]
    attention = []
    for issue in data["issues"]:
        if issue == "service_not_running":
            attention.append(", ".join(f"{row['name']} {row['state']}" for row in services["not_running"]))
        elif issue == "reboot_for_newer_kernel":
            attention.append(f"kernel {kernel['newest_installed']} is installed but not running; a reboot starts it")
        else:
            attention.append(MAINTENANCE_ISSUE_WORDS[issue])
    lines.append("Needs attention: " + "; ".join(attention) + "." if attention else "Nothing needs attention.")
    if services:
        line = f"Services: {services['running']} of {services['installed']} running"
        if services["not_running"]:
            line += "; not running: " + ", ".join(f"{row['name']} ({row['state']})" for row in services["not_running"])
        if services["expected_stopped"]:
            line += "; " + ", ".join(services["expected_stopped"]) + " stopped as expected (the node is not in a cluster)"
        if services["turned_off"]:
            line += "; turned off: " + ", ".join(services["turned_off"])
        lines.append(line + ".")
    if data["certificates"]:
        lines.append("Certificates: " + "; ".join(
            f"{row['file']} ({CERTIFICATE_ROLE_WORDS.get(row['role'], row['role'])}, "
            f"{CERTIFICATE_ISSUER_WORDS[row['issuer']]}, {row['key']}) {_days(row['days_left'])}"
            for row in data["certificates"]) + ".")
    repositories = data["repositories"]
    if repositories:
        line = "Updates come from: " + (", ".join(repositories["enabled"]) or "no repository")
        if repositories["turned_off"]:
            line += "; turned off: " + ", ".join(repositories["turned_off"])
        if repositories["errors"]:
            line += f"; {repositories['errors']} repository error" + ("s" if repositories["errors"] != 1 else "")
        if data["subscription"]:
            line += "; " + SUBSCRIPTION_WORDS.get(data["subscription"], data["subscription"])
        lines.append(line + ".")
    if data["unreadable"]:
        lines.append("Not readable with Janus's access: " + ", ".join(data["unreadable"]) + ".")
    else:
        lines.append("Pending package updates are not read: that needs a privilege Janus does not hold.")
    return "\n".join(lines)


LOAD_ISSUE_WORDS = {
    "memory_low": "available memory fell under 10% of the total",
    "memory_stalls": "tasks stalled waiting for memory",
    "io_stalls": "all tasks stalled on disk IO 10% of the time or more",
    "cpu_busy": "CPU use reached 90% or more",
    "swapping": "swap use reached 10% or more",
}
LOAD_WINDOWS = {"hour": "hour", "day": "day", "week": "week", "month": "month"}


def _rate(value) -> str:
    return _bytes(value) + "/s"


def _step(seconds) -> str:
    if not seconds:
        return "samples"
    if seconds % 3600 == 0:
        hours = seconds // 3600
        return "one sample an hour" if hours == 1 else f"one sample every {hours} hours"
    if seconds % 60 == 0:
        minutes = seconds // 60
        return "one sample a minute" if minutes == 1 else f"one sample every {minutes} minutes"
    return f"one sample every {seconds} seconds"


def format_node_load(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    window = LOAD_WINDOWS.get(data["timeframe"], data["timeframe"])
    if not data["samples"]:
        return (f"Proxmox has no load samples for node {data['node']} over the last {window} "
                f"({data['expected_samples']} expected), so its load is unknown.")
    verdict = ("needs attention: " + "; ".join(LOAD_ISSUE_WORDS[issue] for issue in data["issues"])
               if data["issues"] else "nothing needs attention")
    lines = [f"Load of node {data['node']} over the last {window} ({_step(data['step_seconds'])}, "
             f"{data['samples']} of {data['expected_samples']} present): {verdict}."]
    cpu, wait, load = data["cpu"], data["io_wait"], data["load"]
    lines.append(f"CPU ({cpu['cores']:g} cores): {cpu['average_percent']}% on average, peak {cpu['peak_percent']}%; "
                 f"IO wait {wait['average_percent']}% on average, peak {wait['peak_percent']}%; "
                 f"load average {load['average']}, peak {load['peak']}.")
    memory, swap = data["memory"], data["swap"]
    lines.append(f"Memory: {_bytes(memory['average_used_bytes'])} of {_bytes(memory['total_bytes'])} used on "
                 f"average, peak {_bytes(memory['peak_used_bytes'])}; at least "
                 f"{_bytes(memory['lowest_available_bytes'])} stayed available; swap peak "
                 f"{_bytes(swap['peak_used_bytes'] or 0)} of {_bytes(swap['total_bytes'] or 0)}.")
    pressure = data["pressure"]
    lines.append(f"Stalls (share of time tasks waited): CPU {pressure['cpu_some']:g}%, IO {pressure['io_some']:g}% "
                 f"(all tasks {pressure['io_full']:g}%), memory {pressure['memory_some']:g}% "
                 f"(all tasks {pressure['memory_full']:g}%).")
    network = data["network"]
    lines.append(f"Network: in {_rate(network['average_in_bytes_per_second'])} on average, peak "
                 f"{_rate(network['peak_in_bytes_per_second'])}; out {_rate(network['average_out_bytes_per_second'])} "
                 f"on average, peak {_rate(network['peak_out_bytes_per_second'])}.")
    disk = data["root_disk"]
    if disk["used_bytes"] is not None and disk["total_bytes"]:
        lines.append(f"Root disk: {_used_share(disk['used_bytes'], disk['total_bytes'])}.")
    return "\n".join(lines)


def format_node_network(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    if not data["interfaces"]:
        return f"Proxmox reports no network interface on node {data['node']}."
    summary = data["summary"]
    lines = [f"{data['count']} network interface(s) on node {data['node']}: {summary['bridges']} bridge(s), "
             f"{summary['bonds']} bond(s), {summary['physical']} physical."]
    kinds = {"bridge": "bridge", "OVSBridge": "OVS bridge", "bond": "bond", "OVSBond": "OVS bond",
             "eth": "physical", "vlan": "VLAN"}
    for row in data["interfaces"]:
        kind = kinds.get(row["type"], row["type"])
        if row["bond_mode"]:
            kind += f", {row['bond_mode']}"
        facts = []
        if row["ports"]:
            facts.append("ports " + ", ".join(row["ports"]))
        if row["member_of"]:
            facts.append("port of " + ", ".join(row["member_of"]))
        if row["ipv4"] == "static" or row["ipv4"] == "dhcp":
            facts.append(f"IPv4 {row['ipv4']}" + (" with an address" if row["has_ipv4_address"] else ""))
        if row["vlan_aware"]:
            facts.append("VLAN aware")
        if row["mtu"]:
            facts.append(f"MTU {row['mtu']}")
        if row["autostart"]:
            facts.append("starts at boot")
        if row["active"] is not None:
            facts.append("Proxmox reports it " + ("active" if row["active"] else "inactive"))
        lines.append(f"{row['name']} ({kind})" + (": " + "; ".join(facts) if facts else "") + ".")
    for port in data["missing_ports"]:
        lines.append(f"{port} is named as a port but is not configured on this node.")
    if data["truncated"]:
        lines.append("More interfaces exist than were listed.")
    lines.append("Proxmox reports an interface's state from its last read of the network configuration, so it can "
                 "lag the link; an interface without a report is not known to be down.")
    return "\n".join(lines)


def format_node_disks(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    if not data["disks"]:
        return f"Proxmox lists no physical disk on node {data['node']}."
    summary = data["summary"]
    problems = [f"{count} {text}" for count, text in ((summary["failing"], "failing SMART"),
                                                      (summary["worn"], "worn out"),
                                                      (summary["health_unknown"], "with unknown health")) if count]
    lines = [f"{data['count']} physical disk(s) on {data['node']}: "
             + (", ".join(problems) if problems else "none failing or worn out") + "."]
    for disk in data["disks"]:
        about = ", ".join(part for part in (disk["type"], disk["model"],
                                            _bytes(disk["size_bytes"]) if disk["size_bytes"] is not None else None,
                                            f"used by {disk['used_by']}" if disk["used_by"] else "unused") if part)
        health = {"passed": "SMART passed", "failed": "SMART failed", "smart_disabled": "SMART disabled",
                  "unknown": "SMART health unknown"}[disk["health"]]
        text = f"{disk['device']} ({about}): {health}"
        if disk["wear_used_percent"] is not None:
            text += f"; {disk['wear_used_percent']}% of rated life used"
        text += "."
        if "smart_failed" in disk["issues"]:
            text += " Replace it."
        elif "worn_out" in disk["issues"]:
            text += " Plan a replacement."
        lines.append(text)
    return "\n".join(lines)


SMART_ISSUE_WORDS = {"media_errors": "media errors", "reallocated_sectors": "reallocated sectors",
                     "pending_sectors": "pending sectors", "offline_uncorrectable": "offline uncorrectable sectors",
                     "spare_below_threshold": "spare capacity below its threshold"}


def format_disk_smart(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
    about = ", ".join(part for part in (data["type"], data["model"]) if part)
    health = {"passed": "SMART passed", "failed": "SMART failed", "smart_disabled": "SMART disabled",
              "unknown": "SMART health unknown"}[data["health"]]
    head = f"{data['device']} on node {data['node']} ({about}): {health}"
    if data["wear_used_percent"] is not None:
        head += f"; {data['wear_used_percent']}% of rated life used"
    lines = [head + "."]
    measures = data["measures"]
    facts = []
    if measures["temperature_celsius"] is not None:
        facts.append(f"Temperature {measures['temperature_celsius']} °C")
    if measures["power_on_hours"] is not None:
        facts.append(f"powered on {measures['power_on_hours']} hours ({measures['power_on_hours'] // 24} days)")
    for key, words in (("power_cycles", "power cycles"), ("unsafe_shutdowns", "unsafe shutdowns"),
                       ("media_errors", "media errors"), ("reallocated_sectors", "reallocated sectors"),
                       ("pending_sectors", "pending sectors"),
                       ("offline_uncorrectable", "offline uncorrectable sectors")):
        if measures[key] is not None:
            facts.append(f"{measures[key]} {words}")
    if measures["available_spare_percent"] is not None:
        spare = f"spare capacity {measures['available_spare_percent']}%"
        if measures["available_spare_threshold_percent"] is not None:
            spare += f" (threshold {measures['available_spare_threshold_percent']}%)"
        facts.append(spare)
    if facts:
        lines.append("; ".join(facts) + ".")
    else:
        lines.append("Proxmox reported no SMART measurements for this disk.")
    if data["issues"]:
        named = ", ".join(SMART_ISSUE_WORDS[issue] for issue in data["issues"])
        lines.append(f"Early signs of failure, even though SMART reports passed: {named}."
                     if data["health"] == "passed" else f"Needs attention: {named}.")
    return "\n".join(lines)


def _read(description: str, resource_type: str, permission: str, grounded=(), parameters=None) -> dict:
    return {"description": description, "resource_type": resource_type, "operation_kind": "read",
            "effect": "read", "permission": permission, "risk": "read", "requires_approval": False,
            "grounded_arguments": list(grounded), "required_arguments": list(grounded), "parameters": parameters or {},
            "condition_fields": ["ok", "status"], "result_formatter": format_read}


# ----------------------------------------------------------------------
# Guest power actions
#
# The model proposes only the exact node and VMID the user named. Before the
# approval exists, trusted policy reads the guest and freezes its type and
# name; the approved call carries them, and the provider refuses to act if the
# guest no longer matches. Success means a finished Proxmox task and a
# read-back of the expected guest state.
# ----------------------------------------------------------------------

POWER_LABELS = {
    "start": ("Start", "started"),
    "shutdown": ("Shut down", "shut down"),
    "reboot": ("Reboot", "rebooted"),
    "force_stop": ("Force stop", "force stopped"),
}
POWER_EFFECTS = {
    "start": "It boots the guest.",
    "shutdown": "The guest's own operating system is asked to shut down cleanly (up to 60 seconds).",
    "reboot": "The guest's own operating system is asked to restart cleanly (up to 60 seconds).",
    "force_stop": "The guest is cut off immediately like pulling its power; unsaved data inside it is lost.",
}
POWER_RISKS = {"start": "medium", "shutdown": "medium", "reboot": "medium", "force_stop": "high"}
POWER_TRUSTED_ARGUMENTS = ["expected_type", "expected_name"]


def _policy_service():
    """The configured Proxmox service for trusted pre-approval reads, or None."""
    from config import Settings
    from services.proxmox import build_proxmox_service

    return build_proxmox_service(Settings())


def _power_policy(action: str):
    def evaluate(node: str, vmid: int) -> dict[str, Any]:
        """Read-only: freeze the exact guest (type and name) and refuse what cannot apply now."""
        def refused(status: str, error: str) -> dict[str, Any]:
            return {"ok": False, "status": status, "risk": POWER_RISKS[action], "requires_approval": True,
                    "error": error}

        from services.provider_errors import ProviderResourceNotFound
        from tools.provider_reads import provider_read

        try:
            service = _policy_service()
        except ValueError:
            return refused("error", "Proxmox VE has an invalid configuration on this Janus host.")
        if service is None:
            return refused("error", "Proxmox VE is not set up on this Janus host.")
        try:
            try:
                target = service.power_target(node, vmid, action)
            except (ProviderResourceNotFound, ValueError) as error:
                # Not there, or not a valid guest identity: nothing to approve.
                return refused("denied", str(error))
            except Exception:
                # A provider failure gets the same typed message a read gets.
                target = provider_read("proxmox", "Proxmox VE", lambda: service.power_target(node, vmid, action))
        finally:
            service.close()
        if not target.get("ok"):
            return refused("error", target.get("error") or "Proxmox VE could not confirm the guest.")
        data = target["data"]
        if not data["allowed"]:
            return refused("denied", data["reason"])
        return {
            "ok": True,
            "status": "ready",
            "risk": POWER_RISKS[action],
            "requires_approval": True,
            "execution_arguments": {"node": node, "vmid": vmid, "expected_type": data["type"],
                                    "expected_name": data["name"]},
            "error": None,
        }

    return evaluate


def _guest_label(arguments: dict) -> str:
    kind = str(arguments.get("expected_type") or arguments.get("type") or "guest").upper()
    name = arguments.get("expected_name") or arguments.get("name")
    return f"{kind} {arguments.get('vmid')}" + (f" ({name})" if name else "")


def _power_approval(action: str):
    def format_approval(arguments: dict[str, Any]) -> str:
        return (f"{POWER_LABELS[action][0]} {_guest_label(arguments)} on Proxmox node {arguments.get('node')}. "
                f"{POWER_EFFECTS[action]} Janus waits for the Proxmox task and reads the guest back.")

    return format_approval


def _power_result(action: str):
    def format_result(result: dict[str, Any]) -> str:
        target = f"{_guest_label(result)} on {result.get('node')}"
        if result.get("ok") and result.get("verification_ok"):
            return (f"{target} was {POWER_LABELS[action][1]} and is now {result.get('observed_status')} "
                    f"(Proxmox task {result.get('upid')}).")
        if result.get("status") == "outcome_unknown":
            return (f"The {POWER_LABELS[action][0].lower()} of {target} has an unknown outcome: "
                    f"{result.get('error')} Do not repeat it before checking the guest.")
        return f"The {POWER_LABELS[action][0].lower()} of {target} did not succeed: {result.get('error')}"

    return format_result


def _power(action: str, description: str) -> dict:
    return {
        "description": description,
        "resource_type": "proxmox.guest",
        "operation_kind": "action",
        "effect": "mutation",
        "permission": f"proxmox.guest.{action}",
        "risk": POWER_RISKS[action],
        "requires_approval": True,
        "policy_owns_preconditions": True,
        "policy_resolver": _power_policy(action),
        "trusted_policy_arguments": list(POWER_TRUSTED_ARGUMENTS),
        "grounded_arguments": ["node", "vmid"],
        "required_arguments": ["node", "vmid"],
        "parameters": {
            "node": {"type": "str", "description": "Exact Proxmox node name of the guest, as the user named it."},
            "vmid": {"type": "int", "description": "Exact VMID of the guest, as the user named it."},
        },
        "condition_fields": ["ok", "status", "verification_ok"],
        "approval_formatter": _power_approval(action),
        "result_formatter": _power_result(action),
    }


PROXMOX_TOOLS = {
    "proxmox_cluster_status": _read(
        "Read Proxmox VE cluster membership and reported online state. Read only.",
        "proxmox.cluster", "proxmox.cluster.read"),
    "proxmox_nodes": _read(
        "List Proxmox VE nodes with reported status, CPU, memory, disk and uptime counters. Read only.",
        "proxmox.node", "proxmox.node.list"),
    "proxmox_guests": _read(
        "List QEMU virtual machines and LXC containers with node, status and resource counters. Read only.",
        "proxmox.guest", "proxmox.guest.list"),
    "proxmox_guest_status": _read(
        "Read live status and resource counters for one exact Proxmox node and VMID. Ground both values in proxmox_guests. Read only.",
        "proxmox.guest", "proxmox.guest.status", ("node", "vmid"),
        {"node": {"type": "str", "description": "Exact node name returned by proxmox_guests."},
         "vmid": {"type": "int", "description": "Exact VMID returned by proxmox_guests."}}),
    "proxmox_guest_config": _read(
        "Read allow-listed non-secret configuration for one exact Proxmox node and VMID. Ground both values in proxmox_guests. Read only.",
        "proxmox.guest", "proxmox.guest.config", ("node", "vmid"),
        {"node": {"type": "str", "description": "Exact node name returned by proxmox_guests."},
         "vmid": {"type": "int", "description": "Exact VMID returned by proxmox_guests."}}),
    "proxmox_node_storage": _read(
        "Read storage pools for one exact Proxmox node. Requires the node name exactly as listed by Proxmox. Read only.",
        "proxmox.storage", "proxmox.storage.read", ("node",),
        {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
    "proxmox_recent_tasks": _read(
        "Read up to 50 recent Proxmox tasks for one exact node. Requires the node name exactly as listed by Proxmox. Read only.",
        "proxmox.task", "proxmox.task.list", ("node",),
        {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
    "proxmox_database_inventory": {
        **_read(
            "List the Proxmox VMs and containers the operator tagged as database hosts (tag db or db-<engine>, for "
            "example db-postgres): engine, whether each database VM or container is running or down, memory, disk "
            "use and size (a VM's disk use is not measured), whether its disk is running out of space, whether a "
            "backup job covers it, and the storage their disks live on. Use this only for database hosts and their "
            "capacity; for when guests were last backed up, overdue or missing backups, use proxmox_backups. Read "
            "only; it never connects to a database or runs queries.",
            "proxmox.database", "proxmox.database.inventory"),
        "result_formatter": format_database_inventory,
        # The formatter states every fact in words (including unknown coverage
        # and the tagging hint); synthesis works from that, not the raw flags.
        "synthesis_evidence": "answer",
    },
    "proxmox_backups": {
        **_read(
            "Use this for any question about Proxmox backups, for every VM and container, tagged or not: when "
            "each guest was last backed up, on which backup storage, its size, whether it is protected, encrypted "
            "and verified, which guests missed their backup job's last scheduled run, are overdue (newest backup "
            "older than 8 days, for guests without a scheduled job) or were never backed up, and which no backup "
            "job covers. Read only; it never starts, deletes or restores a backup.",
            "proxmox.backup", "proxmox.backup.list"),
        "result_formatter": format_backups,
        "synthesis_evidence": "answer",
    },
    "proxmox_guest_snapshots": {
        **_read(
            "List the snapshots of one exact Proxmox VM or container: each snapshot's name, when it was taken, "
            "its parent and whether it saved the guest's RAM, and which snapshot the current state follows. Ground "
            "the node and VMID in proxmox_guests. Read only; it never creates, deletes or rolls back a snapshot.",
            "proxmox.snapshot", "proxmox.snapshot.list", ("node", "vmid"),
            {"node": {"type": "str", "description": "Exact node name returned by proxmox_guests."},
             "vmid": {"type": "int", "description": "Exact VMID returned by proxmox_guests."}}),
        "result_formatter": format_guest_snapshots,
        "synthesis_evidence": "answer",
    },
    "proxmox_ha_status": {
        **_read(
            "Show Proxmox high availability (HA) state: whether HA is in use, whether the cluster has quorum, "
            "which node runs the HA manager and its state, each node's HA state (active, idle, maintenance, not "
            "responding) and each HA resource (VM or container) with its state, such as error, fence or recovery, "
            "and the state the operator requested. Read only; it never changes HA, migrates or fences anything.",
            "proxmox.ha", "proxmox.ha.status"),
        "result_formatter": format_ha_status,
        "synthesis_evidence": "answer",
    },
    "proxmox_node_network": {
        **_read(
            "Show the network interfaces of one exact Proxmox node: each bridge (such as vmbr0), bond and "
            "physical NIC, which NIC is the uplink port of which bridge or bond, whether Proxmox reports each "
            "interface active, whether it starts at boot, how IPv4 is configured, MTU and bond mode. Requires the "
            "node name exactly as listed by Proxmox. Read only; it never shows IP addresses, gateways or MAC "
            "addresses and never changes the network.",
            "proxmox.network", "proxmox.network.list", ("node",),
            {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
        "result_formatter": format_node_network,
        "synthesis_evidence": "answer",
    },
    "proxmox_node_maintenance": {
        **_read(
            "Show the maintenance state of one exact Proxmox node: whether it needs a reboot (a newer installed "
            "kernel waits), which Proxmox services are down (not running), its Proxmox VE version and kernel, "
            "boot mode and Secure Boot, when its certificates expire, the subscription and which update "
            "repositories are on. "
            "Requires the node name exactly as listed by Proxmox. Read only; it never shows certificate names, "
            "fingerprints or the subscription key, and never installs updates.",
            "proxmox.node.maintenance", "proxmox.node.maintenance", ("node",),
            {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
        "result_formatter": format_node_maintenance,
        "synthesis_evidence": "answer",
    },
    "proxmox_node_load": {
        **_read(
            "Show how loaded one exact Proxmox node was over the last hour, day, week or month from Proxmox's own "
            "history: average and peak CPU, IO wait, load average, memory used and the least memory left "
            "available, swap, stalls (pressure), network rates and root disk use, naming low memory, stalls, a "
            "busy CPU or swapping. Requires the node name exactly as listed by Proxmox. Read only.",
            "proxmox.node.load", "proxmox.node.load", ("node",),
            {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."},
             "timeframe": {"type": "str", "default": "day",
                           "description": "Window: hour, day, week or month (day if the user names none)."}}),
        "result_formatter": format_node_load,
        "synthesis_evidence": "answer",
    },
    "proxmox_disk_smart": {
        **_read(
            "Show the SMART details of one physical disk on one Proxmox node: temperature, power-on hours, power "
            "cycles, unsafe shutdowns, media errors, reallocated and pending sectors, spare capacity and wear. "
            "Requires the node name and the disk device exactly as proxmox_node_disks lists them (for example "
            "/dev/sda). Read only; it never shows serial numbers and never runs a self-test.",
            "proxmox.disk.smart", "proxmox.disk.smart", ("node", "disk"),
            {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."},
             "disk": {"type": "str", "description": "Exact disk device listed by proxmox_node_disks, e.g. /dev/sda."}}),
        "result_formatter": format_disk_smart,
        "synthesis_evidence": "answer",
    },
    "proxmox_node_disks": {
        **_read(
            "Show the physical disks of one exact Proxmox node: type, model, size, SMART health and how much of "
            "an SSD's rated life is used, naming failing or worn-out disks. Requires the node name exactly as "
            "listed by Proxmox. Read only; serial numbers are never shown.",
            "proxmox.disk", "proxmox.disk.list", ("node",),
            {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
        "result_formatter": format_node_disks,
        "synthesis_evidence": "answer",
    },
    "proxmox_guest_start": _power(
        "start",
        "Start (power on) one stopped Proxmox VM or container, identified by the exact node and VMID the user "
        "named. Requires approval; Janus waits for the Proxmox task and reads the guest back."),
    "proxmox_guest_shutdown": _power(
        "shutdown",
        "Cleanly shut down one running Proxmox VM or container, identified by the exact node and VMID the user "
        "named; its operating system is asked to stop. Requires approval and reads the guest back."),
    "proxmox_guest_reboot": _power(
        "reboot",
        "Cleanly reboot one running Proxmox VM or container, identified by the exact node and VMID the user "
        "named. Requires approval and reads the guest back."),
    "proxmox_guest_force_stop": _power(
        "force_stop",
        "Force stop one running Proxmox VM or container immediately, like pulling its power (unsaved data is "
        "lost). Use only when the user explicitly asks to force stop or kill it. Requires approval."),
}
