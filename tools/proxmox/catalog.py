"""Read-only Proxmox VE capabilities and short, typed result summaries."""
import json


def format_read(result: dict) -> str:
    if not result.get("ok"):
        return "The Proxmox read did not succeed."
    return "Observed Proxmox data: " + json.dumps(result.get("data"), ensure_ascii=False)


def _read(description: str, resource_type: str, permission: str, grounded=(), parameters=None) -> dict:
    return {"description": description, "resource_type": resource_type, "operation_kind": "read",
            "effect": "read", "permission": permission, "risk": "read", "requires_approval": False,
            "grounded_arguments": list(grounded), "required_arguments": list(grounded), "parameters": parameters or {},
            "condition_fields": ["ok", "status"], "result_formatter": format_read}


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
    "proxmox_node_storage": _read(
        "Read storage pools for one exact Proxmox node. Requires the node name exactly as listed by Proxmox. Read only.",
        "proxmox.storage", "proxmox.storage.read", ("node",),
        {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
    "proxmox_recent_tasks": _read(
        "Read up to 50 recent Proxmox tasks for one exact node. Requires the node name exactly as listed by Proxmox. Read only.",
        "proxmox.task", "proxmox.task.list", ("node",),
        {"node": {"type": "str", "description": "Exact node name returned by proxmox_nodes."}}),
}
