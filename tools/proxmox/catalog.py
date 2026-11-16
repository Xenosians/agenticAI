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
        head += (f"{summary['with_recent_backup']} with a recent backup, {summary['stale']} overdue, "
                 f"{summary['never']} never backed up")
        head += (f", {summary['without_backup_job']} without a backup job."
                 if summary["without_backup_job"] is not None else ".")
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
        if "backup_stale" in row["issues"]:
            text += " Overdue."
        lines.append(text)
    if data["storages"]:
        lines.append("Backup storage: " + "; ".join(
            f"{row['storage']} on node {row['node']} ({row['type']}{', shared' if row['shared'] else ''}) holds "
            f"{row['backups']} backup(s)" for row in data["storages"]) + ".")
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


def format_node_disks(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    data = result["data"]
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
            "and verified, which guests are overdue (newest backup older than 8 days) or never backed up, and "
            "which no backup job covers. Read only; it never starts, deletes or restores a backup.",
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
