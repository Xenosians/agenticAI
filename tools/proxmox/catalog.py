"""Proxmox VE capabilities: bounded reads, approval-gated power actions and approval-gated guest changes."""
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
    # Said in full (privilege and why it is not used), so an answer never has to guess at it.
    lines.append("Pending package updates are not read: Proxmox shows them only with Sys.Modify on the node, "
                 "which can also change node settings, so Janus does not use it.")
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
    """The proposing account's Proxmox (its own link, else the machine default) for trusted
    pre-approval reads, or None."""
    from config import Settings
    from services.provider_links import owner_proxmox_service

    return owner_proxmox_service(Settings())


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
            return refused("error", "Proxmox VE is not connected for this account. Add a Proxmox link in "
                                    "Connections.")
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


# ----------------------------------------------------------------------
# Guest changes (owner decisions, 17 November)
#
# Every change names the exact guest the user asked for. Trusted policy
# reads that guest on the proposing account's Proxmox, refuses what cannot
# apply now and freezes the identity the owner will approve (type and name,
# plus the configuration digest for configuration writes). The provider
# re-reads the guest, refuses a different identity, sends exactly one
# request, follows its Proxmox task and reads the expected state back. A
# destructive change is also confirmed by the owner typing the guest's name.
# ----------------------------------------------------------------------

CHANGE_RISKS = {
    "proxmox_guest_set_resources": "medium",
    "proxmox_guest_resize_disk": "medium",
    "proxmox_guest_set_options": "low",
    "proxmox_snapshot_create": "low",
    "proxmox_snapshot_delete": "high",
    "proxmox_snapshot_rollback": "high",
    "proxmox_backup_now": "low",
    "proxmox_backup_restore": "high",
    "proxmox_guest_clone": "medium",
    "proxmox_guest_create": "medium",
    "proxmox_guest_delete": "high",
    "proxmox_database_create": "medium",
}
# The owner must type the target's own name (the approval authority checks
# it against the frozen arguments, not the page) for these.
CONFIRMATION_ARGUMENTS = {
    "proxmox_snapshot_delete": "expected_name",
    "proxmox_snapshot_rollback": "expected_name",
    "proxmox_guest_delete": "expected_name",
    "proxmox_backup_restore": "name",
}
CHANGE_TAIL = ("Janus refuses the change if the guest no longer matches this approval, follows Proxmox's "
               "task and reads the result back.")


def _policy_for(risk: str, read):
    """Run one trusted, read-only change check on the proposing account's Proxmox."""
    def refused(status: str, error: str) -> dict[str, Any]:
        return {"ok": False, "status": status, "risk": risk, "requires_approval": True, "error": error}

    def evaluate(**arguments) -> dict[str, Any]:
        from services.provider_errors import ProviderResourceNotFound
        from services.proxmox import ProxmoxConflict
        from tools.provider_reads import provider_read

        # An optional the request left out arrives as None. It is not part of the request: the
        # approval binds only what was named plus the declared trusted facts (the gateway refuses
        # any other added argument), and the change applies its own default for what was left out.
        arguments = {name: value for name, value in arguments.items() if value is not None}
        try:
            service = _policy_service()
        except ValueError:
            return refused("error", "Proxmox VE has an invalid configuration on this Janus host.")
        if service is None:
            return refused("error", "Proxmox VE is not connected for this account. Add a Proxmox link in "
                                    "Connections.")
        try:
            try:
                frozen = read(service, arguments)
            except (ProviderResourceNotFound, ProxmoxConflict, ValueError) as error:
                # Not there, not what was approved, or not a valid value: nothing to approve.
                return refused("denied", str(error))
            except Exception:
                # A provider failure gets the same typed message a read gets.
                result = provider_read("proxmox", "Proxmox VE",
                                       lambda: {"ok": True, "data": read(service, arguments)})
                if not result.get("ok"):
                    return refused("error", result.get("error") or "Proxmox VE could not confirm the request.")
                frozen = result["data"]
        finally:
            service.close()
        return {"ok": True, "status": "ready", "risk": risk, "requires_approval": True,
                "execution_arguments": {**arguments, **frozen}, "error": None}

    return evaluate


def _freeze_guest(service, arguments: dict, *, digest: bool = False, template_allowed: bool = True,
                  running_allowed: bool = True) -> dict[str, Any]:
    """Freeze the exact guest (and its config digest) or refuse what cannot apply now."""
    from services.proxmox import ProxmoxConflict

    data = service.guest_change_target(arguments["node"], arguments["vmid"])["data"]
    label = f"{data['type'].upper()} {data['vmid']} ({data['name'] or 'unnamed'})"
    if data["template"] and not template_allowed:
        raise ProxmoxConflict(f"{label} is a template; changes apply to a real guest, not a template.")
    if data["lock"]:
        raise ProxmoxConflict(f"{label} is locked ({data['lock']}); wait until that operation finishes.")
    if not running_allowed and data["status"] == "running":
        raise ProxmoxConflict(f"{label} is running; shut it down before deleting it.")
    frozen = {"expected_type": data["type"], "expected_name": data["name"]}
    if digest:
        frozen["expected_digest"] = data["digest"]
    return frozen


def _freeze_new_guest(service, arguments: dict, key: str = "newid") -> dict[str, Any]:
    """Confirm the target node and a free VMID; when none was named, ask Proxmox for the next free one.

    Nothing here assumes a numbering range, node or pool: the owner's Proxmox
    answers with its own next free VMID, and the approval freezes that answer.
    """
    requested = arguments.get(key)
    if requested is None:
        allocated = service.next_vmid()
        service.new_guest_target(arguments["node"], allocated)
        return {"allocated_vmid": allocated}
    service.new_guest_target(arguments["node"], requested)
    return {}


def _effective_vmid(arguments: dict) -> int | None:
    return arguments.get("allocated_vmid") or arguments.get("newid") or arguments.get("vmid")


def _read_set_resources(service, arguments: dict) -> dict[str, Any]:
    from services.proxmox import MAX_CORES, MAX_MEMORY_MIB

    if arguments.get("cores") is None and arguments.get("memory_mib") is None:
        raise ValueError("Enter a core count or memory in MiB; one is required.")
    if arguments.get("cores") is not None and (type(arguments["cores"]) is not int
                                               or not (1 <= arguments["cores"] <= MAX_CORES)):
        raise ValueError(f"Enter a core count (1-{MAX_CORES}).")
    if arguments.get("memory_mib") is not None and (type(arguments["memory_mib"]) is not int
                                                    or not (16 <= arguments["memory_mib"] <= MAX_MEMORY_MIB)):
        raise ValueError(f"Enter memory in MiB (16-{MAX_MEMORY_MIB}).")
    return _freeze_guest(service, arguments, digest=True, template_allowed=False)


def _read_resize_disk(service, arguments: dict) -> dict[str, Any]:
    from services.proxmox import DISK_ID, MAX_DISK_GIB

    if not isinstance(arguments.get("disk"), str) or not DISK_ID.fullmatch(arguments["disk"]):
        raise ValueError("Enter one disk of the guest exactly as its configuration lists it (for example scsi0).")
    if type(arguments.get("grow_gib")) is not int or not (1 <= arguments["grow_gib"] <= MAX_DISK_GIB):
        raise ValueError(f"Enter how many GiB to add (1-{MAX_DISK_GIB}).")
    return _freeze_guest(service, arguments, digest=True, template_allowed=False)


def _read_set_options(service, arguments: dict) -> dict[str, Any]:
    if arguments.get("tags") is None and arguments.get("on_boot") is None:
        raise ValueError("Enter tags or a start-at-boot choice.")
    return _freeze_guest(service, arguments, digest=True, template_allowed=False)


def _read_snapshot(service, arguments: dict) -> dict[str, Any]:
    from services.proxmox import SNAPSHOT_NAME

    snapshot = arguments.get("snapshot")
    if not isinstance(snapshot, str) or not SNAPSHOT_NAME.fullmatch(snapshot) or snapshot == "current":
        raise ValueError("Enter a snapshot name starting with a letter (letters, digits, '_' or '-').")
    return _freeze_guest(service, arguments)


def _read_backup_now(service, arguments: dict) -> dict[str, Any]:
    from services.proxmox import STORAGE_ID

    if not isinstance(arguments.get("storage"), str) or not STORAGE_ID.fullmatch(arguments["storage"]):
        raise ValueError("Enter one exact backup storage name.")
    if arguments.get("mode", "snapshot") not in {"snapshot", "suspend", "stop"}:
        raise ValueError("The backup mode must be snapshot, suspend or stop.")
    return _freeze_guest(service, arguments)


def _read_clone(service, arguments: dict) -> dict[str, Any]:
    from services.proxmox import GUEST_NAME

    newid = arguments.get("newid")
    if newid is not None and (type(newid) is not int or newid == arguments.get("vmid")):
        raise ValueError("Enter a new, different Proxmox VMID.")
    if not isinstance(arguments.get("name"), str) or not GUEST_NAME.fullmatch(arguments["name"]):
        raise ValueError("Enter a guest name of letters, digits and hyphens, starting and ending with one.")
    if type(arguments.get("full")) is not bool:
        raise ValueError("Choose a full or linked clone.")
    frozen = _freeze_guest(service, arguments)
    frozen.update(_freeze_new_guest(service, arguments, "newid"))
    return frozen


def _read_create(service, arguments: dict) -> dict[str, Any]:
    from services.proxmox import GUEST_NAME, MAX_CORES, MAX_DISK_GIB, MAX_MEMORY_MIB

    if not isinstance(arguments.get("hostname"), str) or not GUEST_NAME.fullmatch(arguments["hostname"]):
        raise ValueError("Enter a hostname of letters, digits and hyphens, starting and ending with one.")
    if type(arguments.get("cores")) is not int or not (1 <= arguments["cores"] <= MAX_CORES):
        raise ValueError(f"Enter a core count (1-{MAX_CORES}).")
    if type(arguments.get("memory_mib")) is not int or not (16 <= arguments["memory_mib"] <= MAX_MEMORY_MIB):
        raise ValueError(f"Enter memory in MiB (16-{MAX_MEMORY_MIB}).")
    if type(arguments.get("disk_gib")) is not int or not (1 <= arguments["disk_gib"] <= MAX_DISK_GIB):
        raise ValueError(f"Enter a disk size in GiB (1-{MAX_DISK_GIB}).")
    return _freeze_new_guest(service, arguments, "vmid")


def _read_restore(service, arguments: dict) -> dict[str, Any]:
    return _freeze_new_guest(service, arguments, "newid")


def _guest_label(arguments: dict) -> str:
    kind = str(arguments.get("expected_type") or arguments.get("type") or "guest").upper()
    name = arguments.get("expected_name") or arguments.get("name")
    return f"{kind} {arguments.get('vmid')}" + (f" ({name})" if name else "")


def _change_approval(text) -> Any:
    def format_approval(arguments: dict[str, Any]) -> str:
        return f"{text(arguments)} {CHANGE_TAIL}"
    return format_approval


def _change_result(done: str) -> Any:
    def format_result(result: dict[str, Any]) -> str:
        target = f"{_guest_label(result)} on node {result.get('node')}"
        task = f" (Proxmox task {result['upid']})" if result.get("upid") else ""
        if result.get("ok") and result.get("verification_ok"):
            return f"{done} for {target}{task}. Janus read the guest back and confirmed it."
        if result.get("status") == "outcome_unknown":
            return (f"The change to {target} has an unknown outcome: {result.get('error')} "
                    "Do not repeat it before checking the guest.")
        return f"The change to {target} did not succeed: {result.get('error')}"

    return format_result


def _resources_approval(arguments: dict[str, Any]) -> str:
    parts = []
    if arguments.get("cores") is not None:
        parts.append(f"{arguments['cores']} CPU core(s)")
    if arguments.get("memory_mib") is not None:
        parts.append(f"{arguments['memory_mib']} MiB of memory")
    return f"Set {_guest_label(arguments)} on Proxmox node {arguments.get('node')} to {' and '.join(parts)}."


def _resize_approval(arguments: dict[str, Any]) -> str:
    return (f"Grow disk {arguments.get('disk')} of {_guest_label(arguments)} on Proxmox node "
            f"{arguments.get('node')} by {arguments.get('grow_gib')} GiB.")


def _options_approval(arguments: dict[str, Any]) -> str:
    parts = []
    if arguments.get("tags") is not None:
        parts.append(f"tags '{arguments['tags']}'")
    if arguments.get("on_boot") is not None:
        parts.append("start at boot " + ("on" if arguments["on_boot"] else "off"))
    return (f"Set {_guest_label(arguments)} on Proxmox node {arguments.get('node')} to "
            + " and ".join(parts) + ".")


def _snapshot_create_approval(arguments: dict[str, Any]) -> str:
    description = f" with description '{arguments['description']}'" if arguments.get("description") else ""
    return (f"Take snapshot '{arguments.get('snapshot')}' of {_guest_label(arguments)} on Proxmox node "
            f"{arguments.get('node')}{description}.")


def _snapshot_delete_approval(arguments: dict[str, Any]) -> str:
    return (f"Delete snapshot '{arguments.get('snapshot')}' of {_guest_label(arguments)} on Proxmox node "
            f"{arguments.get('node')}. The snapshot cannot be recovered.")


def _snapshot_rollback_approval(arguments: dict[str, Any]) -> str:
    return (f"Roll {_guest_label(arguments)} on Proxmox node {arguments.get('node')} back to snapshot "
            f"'{arguments.get('snapshot')}'. Changes made since that snapshot are lost.")


def _backup_approval(arguments: dict[str, Any]) -> str:
    return (f"Back up {_guest_label(arguments)} on Proxmox node {arguments.get('node')} now to storage "
            f"'{arguments.get('storage')}' ({arguments.get('mode', 'snapshot')} mode).")


def _restore_approval(arguments: dict[str, Any]) -> str:
    picked = (" Janus asked Proxmox for the next free VMID." if arguments.get("allocated_vmid") else "")
    return (f"Restore backup volume '{arguments.get('volume')}' into a new guest {_effective_vmid(arguments)} "
            f"({arguments.get('name')}) on Proxmox node {arguments.get('node')}.{picked} Janus refuses a VMID "
            f"that is already used, so no existing guest is overwritten.")


def _clone_approval(arguments: dict[str, Any]) -> str:
    kind = "full" if arguments.get("full") else "linked"
    picked = " Janus asked Proxmox for the next free VMID." if arguments.get("allocated_vmid") else ""
    return (f"Clone {_guest_label(arguments)} on Proxmox node {arguments.get('node')} to new guest "
            f"{_effective_vmid(arguments)} ({arguments.get('name')}) as a {kind} clone." + picked)


def _create_approval(arguments: dict[str, Any]) -> str:
    picked = " Janus asked Proxmox for the next free VMID." if arguments.get("allocated_vmid") else ""
    return (f"Create LXC {_effective_vmid(arguments)} ({arguments.get('hostname')}) on Proxmox node "
            f"{arguments.get('node')} from template {arguments.get('template')}: {arguments.get('cores')} "
            f"core(s), {arguments.get('memory_mib')} MiB memory, {arguments.get('disk_gib')} GiB disk on "
            f"storage {arguments.get('storage')}." + picked)


def _delete_approval(arguments: dict[str, Any]) -> str:
    return (f"Delete {_guest_label(arguments)} on Proxmox node {arguments.get('node')} and purge its disks. "
            "The guest and its data cannot be recovered.")


def _change_tool(*, name: str, description: str, permission: str, policy, trusted: list[str],
                 required: list[str], parameters: dict, approval, done: str,
                 grounded: list[str] | None = None, derived: list[str] | None = None,
                 operation: str = "action") -> dict:
    tool = {
        "description": description,
        "resource_type": "proxmox.guest",
        "operation_kind": operation,
        "effect": "mutation",
        "permission": permission,
        "risk": CHANGE_RISKS[name],
        "requires_approval": True,
        "policy_owns_preconditions": True,
        "policy_resolver": policy,
        "trusted_policy_arguments": trusted,
        "grounded_arguments": grounded if grounded is not None else ["node", "vmid"],
        "derived_arguments": derived or [],
        "required_arguments": required,
        "parameters": parameters,
        "condition_fields": ["ok", "status", "verification_ok"],
        "approval_formatter": _change_approval(approval),
        "result_formatter": _change_result(done),
    }
    if name in CONFIRMATION_ARGUMENTS:
        tool["confirmation_argument"] = CONFIRMATION_ARGUMENTS[name]
    return tool


GUEST_CHANGE_TRUSTED = ["expected_type", "expected_name"]
CONFIG_CHANGE_TRUSTED = ["expected_type", "expected_name", "expected_digest"]
GUEST_PAIR = {"node": {"type": "str", "description": "Exact Proxmox node name of the guest, as listed by "
                                                       "proxmox_guests."},
              "vmid": {"type": "int", "description": "Exact VMID of the guest, as listed by proxmox_guests."}}


def _set_resources_policy(node: str, vmid: int, cores: int | None = None, memory_mib: int | None = None):
    return _policy_for(CHANGE_RISKS["proxmox_guest_set_resources"], _read_set_resources)(
        node=node, vmid=vmid, cores=cores, memory_mib=memory_mib)


def _resize_disk_policy(node: str, vmid: int, disk: str, grow_gib: int):
    return _policy_for(CHANGE_RISKS["proxmox_guest_resize_disk"], _read_resize_disk)(
        node=node, vmid=vmid, disk=disk, grow_gib=grow_gib)


def _set_options_policy(node: str, vmid: int, tags: str | None = None, on_boot: bool | None = None):
    return _policy_for(CHANGE_RISKS["proxmox_guest_set_options"], _read_set_options)(
        node=node, vmid=vmid, tags=tags, on_boot=on_boot)


def _snapshot_create_policy(node: str, vmid: int, snapshot: str, description: str | None = None):
    return _policy_for(CHANGE_RISKS["proxmox_snapshot_create"], _read_snapshot)(
        node=node, vmid=vmid, snapshot=snapshot, description=description)


def _snapshot_delete_policy(node: str, vmid: int, snapshot: str):
    return _policy_for(CHANGE_RISKS["proxmox_snapshot_delete"], _read_snapshot)(
        node=node, vmid=vmid, snapshot=snapshot)


def _snapshot_rollback_policy(node: str, vmid: int, snapshot: str):
    return _policy_for(CHANGE_RISKS["proxmox_snapshot_rollback"], _read_snapshot)(
        node=node, vmid=vmid, snapshot=snapshot)


def _backup_now_policy(node: str, vmid: int, storage: str, mode: str | None = None):
    # An unnamed mode stays unnamed: the approval reads "snapshot mode" and the change defaults to it.
    return _policy_for(CHANGE_RISKS["proxmox_backup_now"], _read_backup_now)(
        node=node, vmid=vmid, storage=storage, mode=mode)


def _backup_restore_policy(node: str, volume: str, name: str, newid: int | None = None,
                           storage: str | None = None, pool: str | None = None):
    return _policy_for(CHANGE_RISKS["proxmox_backup_restore"], _read_restore)(
        node=node, volume=volume, name=name, newid=newid, storage=storage, pool=pool)


def _clone_policy(node: str, vmid: int, name: str, full: bool, newid: int | None = None,
                  pool: str | None = None):
    return _policy_for(CHANGE_RISKS["proxmox_guest_clone"], _read_clone)(
        node=node, vmid=vmid, name=name, full=full, newid=newid, pool=pool)


def _create_policy(node: str, hostname: str, template: str, cores: int, memory_mib: int, disk_gib: int,
                   storage: str, vmid: int | None = None, pool: str | None = None,
                   description: str | None = None):
    return _policy_for(CHANGE_RISKS["proxmox_guest_create"], _read_create)(
        node=node, hostname=hostname, template=template, cores=cores, memory_mib=memory_mib,
        disk_gib=disk_gib, storage=storage, vmid=vmid, pool=pool, description=description)


def _delete_policy(node: str, vmid: int):
    def read(service, arguments: dict) -> dict[str, Any]:
        return _freeze_guest(service, arguments, running_allowed=False)

    return _policy_for(CHANGE_RISKS["proxmox_guest_delete"], read)(node=node, vmid=vmid)


# Database guests from official OCI images (owner decision 20 Nov 2026, ~14:10 WIB, item 5). The guest
# lives in the lab pool on the subnet the tailnet bridge routes; the policy picks its VMID and its free
# address from Proxmox itself and freezes the whole plan, so the approval shows exactly what runs.
DATABASE_POOL = "janus"
DATABASE_VNET = "janusnat"
DATABASE_GATEWAY = "10.77.0.1"
DATABASE_TEMPLATE_STORAGE = "local"
# Addresses .20-.99 of the bridged subnet are for database guests (.10 is the trainer, .1 the gateway).
DATABASE_ADDRESSES = range(20, 100)


def _used_bridge_addresses(service, node: str) -> set[str]:
    """Every 10.77.0.x a guest's config already names, read from Proxmox (never guessed)."""
    return service.guest_addresses(node, "10.77.0.0/24")


def _read_database_create(service, arguments: dict) -> dict[str, Any]:
    from services.database_guests import DatabaseGuestRefused, database_guest_plan

    frozen = _freeze_new_guest(service, arguments, "vmid")
    vmid = frozen.get("allocated_vmid") or arguments.get("vmid")
    if type(vmid) is not int:
        raise ValueError("Proxmox did not name a free VMID for the database guest.")
    used = _used_bridge_addresses(service, arguments["node"])
    free = next((f"10.77.0.{host}" for host in DATABASE_ADDRESSES if f"10.77.0.{host}" not in used), None)
    if free is None:
        raise ValueError("Every database address in 10.77.0.20-99 is in use; remove a database guest first.")
    try:
        plan = database_guest_plan(
            engine=str(arguments.get("engine") or ""), version=str(arguments.get("version") or ""),
            hostname=str(arguments.get("hostname") or ""), node=arguments["node"], vmid=vmid,
            storage=arguments.get("storage") or "local-lvm",
            template_storage=DATABASE_TEMPLATE_STORAGE, pool=DATABASE_POOL, vnet=DATABASE_VNET,
            address=f"{free}/24", gateway=DATABASE_GATEWAY, disk_gib=arguments.get("disk_gib") or 4,
            memory_mib=arguments.get("memory_mib") or 1024, cores=arguments.get("cores") or 1,
            data_gib=arguments.get("data_gib"))
    except DatabaseGuestRefused as error:
        raise ValueError(str(error)) from None
    return {**frozen, "plan": plan}


def _database_create_policy(node: str, engine: str, version: str, hostname: str, vmid: int | None = None,
                            storage: str | None = None, cores: int | None = None, memory_mib: int | None = None,
                            disk_gib: int | None = None, data_gib: int | None = None):
    return _policy_for("medium", _read_database_create)(
        node=node, engine=engine, version=version, hostname=hostname, vmid=vmid, storage=storage, cores=cores,
        memory_mib=memory_mib, disk_gib=disk_gib, data_gib=data_gib)


def _database_create_approval(arguments: dict[str, Any]) -> str:
    plan = arguments.get("plan") or {}
    create = plan.get("create") or {}
    return (f"Create database guest {plan.get('hostname')} (LXC {plan.get('vmid')}) on Proxmox node "
            f"{plan.get('node')} from the official image {plan.get('reference')}: {create.get('cores')} core(s), "
            f"{create.get('memory')} MiB memory, root disk {create.get('rootfs')}, data volume "
            f"{(create.get('mp0') or '').split(',', 1)[0]}, in pool {create.get('pool')}, at "
            f"{plan.get('address')}:{plan.get('port')} on the tailnet-bridged subnet. "
            + ("Proxmox pulls the image first. " if plan.get("template") else "")
            + "No password is sent; Janus sets one afterwards and keeps it encrypted for your account.")


PROXMOX_CHANGE_TOOLS = {
    "proxmox_guest_set_resources": _change_tool(
        name="proxmox_guest_set_resources",
        description="Set the CPU cores and/or memory (MiB) of one exact Proxmox VM or container named by the "
                    "user. Requires approval; Janus refuses a concurrently changed configuration and reads the "
                    "guest back.",
        permission="proxmox.guest.set_resources", policy=_set_resources_policy,
        trusted=CONFIG_CHANGE_TRUSTED, required=["node", "vmid"],
        parameters={**GUEST_PAIR,
                    "cores": {"type": "int", "description": "CPU cores to set (1-128); omit to leave unchanged."},
                    "memory_mib": {"type": "int", "description": "Memory in MiB to set (16-1048576); omit to "
                                                                "leave unchanged."}},
        approval=_resources_approval, done="CPU and memory were set", operation="update"),
    "proxmox_guest_resize_disk": _change_tool(
        name="proxmox_guest_resize_disk",
        description="Grow one disk of one exact Proxmox VM or container by a number of GiB (grow only; never "
                    "shrinks). Requires approval and reads the disk size back.",
        permission="proxmox.guest.resize_disk", policy=_resize_disk_policy,
        trusted=CONFIG_CHANGE_TRUSTED, required=["node", "vmid", "disk", "grow_gib"],
        derived=["disk", "grow_gib"],
        parameters={**GUEST_PAIR,
                    "disk": {"type": "str", "description": "Exact disk id from the guest's configuration, for "
                                                          "example scsi0 or rootfs."},
                    "grow_gib": {"type": "int", "description": "GiB to add (1-65536); the disk only grows."}},
        approval=_resize_approval, done="The disk was grown", operation="update"),
    "proxmox_guest_set_options": _change_tool(
        name="proxmox_guest_set_options",
        description="Set the tags and/or the start-at-boot flag of one exact Proxmox VM or container named by "
                    "the user. Requires approval; Janus refuses a concurrently changed configuration and reads "
                    "the guest back.",
        permission="proxmox.guest.set_options", policy=_set_options_policy,
        trusted=CONFIG_CHANGE_TRUSTED, required=["node", "vmid"],
        parameters={**GUEST_PAIR,
                    "tags": {"type": "str", "description": "Comma-separated tags to set; omit to leave unchanged."},
                    "on_boot": {"type": "bool", "description": "Start at boot; omit to leave unchanged."}},
        approval=_options_approval, done="Tags and start-at-boot were set", operation="update"),
    "proxmox_snapshot_create": _change_tool(
        name="proxmox_snapshot_create",
        description="Take one named snapshot of one exact Proxmox VM or container named by the user. Requires "
                    "approval; Janus follows the task and confirms the snapshot is listed.",
        permission="proxmox.snapshot.create", policy=_snapshot_create_policy,
        trusted=GUEST_CHANGE_TRUSTED, required=["node", "vmid", "snapshot"], derived=["snapshot"],
        parameters={**GUEST_PAIR,
                    "snapshot": {"type": "str", "description": "Snapshot name: a letter first, then letters, "
                                                              "digits, '_' or '-'."},
                    "description": {"type": "str", "description": "Optional one-line description (up to 256 "
                                                                 "characters)."}},
        approval=_snapshot_create_approval, done="The snapshot was taken", operation="create"),
    "proxmox_snapshot_delete": _change_tool(
        name="proxmox_snapshot_delete",
        description="Delete one named snapshot of one exact Proxmox VM or container. Destructive: requires "
                    "approval and the exact guest name typed by the owner; the snapshot cannot be recovered.",
        permission="proxmox.snapshot.delete", policy=_snapshot_delete_policy,
        trusted=GUEST_CHANGE_TRUSTED, required=["node", "vmid", "snapshot"], derived=["snapshot"],
        parameters={**GUEST_PAIR,
                    "snapshot": {"type": "str", "description": "Exact existing snapshot name."}},
        approval=_snapshot_delete_approval, done="The snapshot was deleted", operation="delete"),
    "proxmox_snapshot_rollback": _change_tool(
        name="proxmox_snapshot_rollback",
        description="Roll one exact Proxmox VM or container back to a named snapshot. Destructive: changes made "
                    "since the snapshot are lost. Requires approval and the exact guest name typed by the owner.",
        permission="proxmox.snapshot.rollback", policy=_snapshot_rollback_policy,
        trusted=GUEST_CHANGE_TRUSTED, required=["node", "vmid", "snapshot"], derived=["snapshot"],
        parameters={**GUEST_PAIR,
                    "snapshot": {"type": "str", "description": "Exact existing snapshot name to roll back to."}},
        approval=_snapshot_rollback_approval, done="The rollback finished"),
    "proxmox_backup_now": _change_tool(
        name="proxmox_backup_now",
        description="Start one backup of one exact Proxmox VM or container named by the user to a backup "
                    "storage. Requires approval; Janus follows the task and confirms a new backup volume.",
        permission="proxmox.backup.now", policy=_backup_now_policy,
        trusted=GUEST_CHANGE_TRUSTED, required=["node", "vmid", "storage"], derived=["storage"],
        parameters={**GUEST_PAIR,
                    "storage": {"type": "str", "description": "Exact backup storage name, as proxmox_node_storage "
                                                             "lists it."},
                    "mode": {"type": "str", "default": "snapshot",
                             "description": "Backup mode: snapshot (default), suspend or stop."}},
        approval=_backup_approval, done="The backup finished"),
    "proxmox_backup_restore": _change_tool(
        name="proxmox_backup_restore",
        description="Restore one backup volume into a NEW guest. Destructive: requires approval and the new "
                    "guest's name typed by the owner. Omit the VMID and Janus uses Proxmox's next free one; "
                    "Janus refuses a used VMID, so no existing guest is overwritten.",
        permission="proxmox.backup.restore", policy=_backup_restore_policy,
        trusted=["allocated_vmid"], required=["node", "volume", "name"],
        grounded=["node", "volume", "newid"],
        derived=["name"],
        parameters={"node": {"type": "str", "description": "Exact Proxmox node to restore onto, as listed by "
                                                          "proxmox_nodes."},
                    "volume": {"type": "str", "description": "Exact backup volume id from the storage's backup "
                                                            "content, as listed by Proxmox."},
                    "newid": {"type": "int", "description": "Optional free VMID; omit and Janus uses Proxmox's "
                                                            "next free one."},
                    "name": {"type": "str", "description": "Name for the new guest: letters, digits and hyphens."},
                    "storage": {"type": "str", "description": "Optional exact target storage name."},
                    "pool": {"type": "str", "description": "Optional exact pool name for the new guest."}},
        approval=_restore_approval, done="The restore created the new guest"),
    "proxmox_guest_clone": _change_tool(
        name="proxmox_guest_clone",
        description="Clone one exact Proxmox VM or container to a NEW guest with a new name. Requires approval. "
                    "Omit the VMID and Janus uses Proxmox's next free one; a used VMID is refused. Janus follows "
                    "the task and confirms the new guest exists.",
        permission="proxmox.guest.clone", policy=_clone_policy,
        trusted=GUEST_CHANGE_TRUSTED + ["allocated_vmid"], required=["node", "vmid", "name", "full"],
        grounded=["node", "vmid", "newid"],
        derived=["name", "full"],
        parameters={**GUEST_PAIR,
                    "newid": {"type": "int", "description": "Optional free VMID; omit and Janus uses Proxmox's "
                                                            "next free one."},
                    "name": {"type": "str", "description": "Name for the clone: letters, digits and hyphens."},
                    "full": {"type": "bool", "description": "True for a full clone, False for a linked clone."},
                    "pool": {"type": "str", "description": "Optional exact pool name for the clone."}},
        approval=_clone_approval, done="The clone was created", operation="create"),
    "proxmox_guest_create": _change_tool(
        name="proxmox_guest_create",
        description="Create one stopped LXC container from a template on one exact Proxmox node. Requires "
                    "approval. Omit the VMID and Janus uses Proxmox's next free one; a used VMID is refused. "
                    "Janus follows the task and confirms the new container exists.",
        permission="proxmox.guest.create", policy=_create_policy,
        trusted=["allocated_vmid"], required=["node", "hostname", "template", "cores", "memory_mib", "disk_gib",
                                              "storage"], grounded=["node", "template", "vmid"],
        derived=["hostname", "cores", "memory_mib", "disk_gib", "storage"],
        parameters={"node": {"type": "str", "description": "Exact Proxmox node, as listed by proxmox_nodes."},
                    "vmid": {"type": "int", "description": "Optional free VMID; omit and Janus uses Proxmox's "
                                                           "next free one."},
                    "hostname": {"type": "str", "description": "Hostname: letters, digits and hyphens."},
                    "template": {"type": "str", "description": "Exact template volume from the node's storage "
                                                               "content, ending in .tar.xz or similar."},
                    "cores": {"type": "int", "description": "CPU cores (1-128)."},
                    "memory_mib": {"type": "int", "description": "Memory in MiB (16-1048576)."},
                    "disk_gib": {"type": "int", "description": "Root disk size in GiB (1-65536)."},
                    "storage": {"type": "str", "description": "Exact storage name for the root disk."},
                    "pool": {"type": "str", "description": "Optional exact pool name for the new container."},
                    "description": {"type": "str", "description": "Optional one-line description (up to 256 "
                                                                 "characters)."}},
        approval=_create_approval, done="The container was created", operation="create"),
    "proxmox_guest_delete": _change_tool(
        name="proxmox_guest_delete",
        description="Delete one exact stopped Proxmox VM or container and purge its disks. Destructive: requires "
                    "approval and the exact guest name typed by the owner; the guest cannot be recovered.",
        permission="proxmox.guest.delete", policy=_delete_policy,
        trusted=GUEST_CHANGE_TRUSTED, required=["node", "vmid"],
        parameters=GUEST_PAIR,
        approval=_delete_approval, done="The guest was deleted", operation="delete"),
    "proxmox_database_create": {
        **_change_tool(
            name="proxmox_database_create",
            description="Create one PostgreSQL or MariaDB database guest from the engine's official OCI image at "
                        "a pinned version, in the lab pool on the tailnet-bridged subnet, with its data on its own "
                        "volume. Requires approval; Janus picks the VMID and a free address and confirms the "
                        "guest exists.",
            permission="proxmox.database.create", policy=_database_create_policy,
            trusted=["allocated_vmid", "plan"], required=["node", "engine", "version", "hostname"],
            grounded=["node", "engine", "version", "vmid"],
            derived=["hostname", "storage", "cores", "memory_mib", "disk_gib", "data_gib"],
            parameters={"node": {"type": "str", "description": "Exact Proxmox node, as listed by proxmox_nodes."},
                        "engine": {"type": "str", "description": "postgresql or mariadb."},
                        "version": {"type": "str", "description": "Pinned image tag, e.g. 17-alpine or 11.8."},
                        "hostname": {"type": "str", "description": "Hostname: lowercase letters, digits, hyphens."},
                        "vmid": {"type": "int", "description": "Optional free VMID; omit and Janus uses Proxmox's "
                                                               "next free one."},
                        "storage": {"type": "str", "description": "Optional disk storage; default local-lvm."},
                        "cores": {"type": "int", "description": "Optional CPU cores (default 1)."},
                        "memory_mib": {"type": "int", "description": "Optional memory in MiB (default 1024)."},
                        "disk_gib": {"type": "int", "description": "Optional root disk GiB (default 4)."},
                        "data_gib": {"type": "int", "description": "Optional data volume GiB (default = root)."}},
            approval=_database_create_approval, done="The database guest was created", operation="create"),
        "resource_type": "proxmox.database",
    },
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
            "List Proxmox guests (VMs and containers) the operator tagged as database hosts (tag db or db-<engine>, "
            "for example db-postgres): engine, whether each database VM or container is running or down, memory, disk "
            "capacity and allocated disk size, scheduled backup-job coverage, and the storage their disks live on. "
            "Use this only for tagged database hosts and their capacity. Read only; it never connects to a database "
            "or runs queries.",
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

# Guest changes share the provider and the approval machinery; one catalog.
PROXMOX_TOOLS.update(PROXMOX_CHANGE_TOOLS)
