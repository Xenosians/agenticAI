"""Proxmox VE capabilities: bounded reads and four approval-gated guest power actions."""
import json
from typing import Any


def format_read(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    return "Observed Proxmox data: " + json.dumps(result.get("data"), ensure_ascii=False)


def _bytes(value) -> str:
    for unit, size in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if value >= size:
            return f"{value / size:.1f} {unit}" if unit == "GiB" else f"{value / size:.0f} {unit}"
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
            "backup job covers it, and the storage their disks live on. Use this for database servers, their "
            "backups and capacity. Read only; it never connects to a database or runs queries.",
            "proxmox.database", "proxmox.database.inventory"),
        "result_formatter": format_database_inventory,
        # The formatter states every fact in words (including unknown coverage
        # and the tagging hint); synthesis works from that, not the raw flags.
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
