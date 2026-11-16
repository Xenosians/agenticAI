"""Which Proxmox nodes and guests have a Tailscale device (tailnet coverage), read-only.

The Proxmox inventory and the tailnet share no identifier Janus may expose
(addresses stay inside their services), so a machine and a device match by
name: the device's host name or the first label of its MagicDNS name, both
normalized the way Tailscale derives MagicDNS names (lower case; any run of
characters other than letters and digits becomes one hyphen). A machine
without a match may still be on the tailnet under another name, so the
reading says "no device named like it", not "not on the tailnet". When
either listing was cut, no machine is reported as missing.
"""
from __future__ import annotations

import re

_NON_NAME = re.compile(r"[^a-z0-9]+")
# Most urgent first.
COVERAGE_ISSUES = ("device_offline", "no_tailnet_device")


def name_key(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    key = _NON_NAME.sub("-", value.lower()).strip("-")
    return key or None


def _device_keys(device: dict) -> set[str]:
    keys = {name_key(device.get("name"))}
    dns_name = device.get("dns_name")
    if isinstance(dns_name, str) and dns_name:
        keys.add(name_key(dns_name.split(".", 1)[0]))
    return {key for key in keys if key}


def _row(machine_name: str | None, running: bool, index: dict, complete: bool) -> tuple[list[dict], list[str]]:
    key = name_key(machine_name)
    matched = index.get(key, []) if key else []
    devices = [{"name": device["name"], "online": device.get("online")} for device in matched]
    issues = []
    if running and devices and all(device["online"] is False for device in devices):
        issues.append("device_offline")
    if complete and not devices:
        issues.append("no_tailnet_device")
    return devices, issues


def tailnet_coverage(machines: dict, devices: dict) -> dict:
    """Join a Proxmox machine listing (ProxmoxService.machines data) with tailnet devices (devices() data)."""
    complete = not machines["truncated"] and not devices["truncated"]
    index: dict[str, list[dict]] = {}
    for device in devices["devices"]:
        for key in _device_keys(device):
            index.setdefault(key, []).append(device)
    used: set[int] = set()
    nodes = []
    for node in machines["nodes"]:
        matched, issues = _row(node["node"], node["status"] == "online", index, complete)
        used.update(id(device) for key in [name_key(node["node"])] if key for device in index.get(key, []))
        nodes.append({"node": node["node"], "status": node["status"], "tailnet_devices": matched, "issues": issues})
    guests = []
    for guest in machines["guests"]:
        matched, issues = _row(guest["name"], guest["status"] == "running", index, complete)
        used.update(id(device) for key in [name_key(guest["name"])] if key for device in index.get(key, []))
        guests.append({**{field: guest[field] for field in ("vmid", "node", "type", "name", "status")},
                       "tailnet_devices": matched, "issues": issues})
    rank = {issue: position for position, issue in enumerate(COVERAGE_ISSUES)}
    guests.sort(key=lambda row: (min((rank[issue] for issue in row["issues"]), default=len(rank)),
                                 row["node"].lower(), row["vmid"]))
    outside = sorted((device["name"] for device in devices["devices"] if id(device) not in used), key=str.lower)
    return {
        "schema": "tailnet-coverage.v1",
        "match": "name",
        "complete": complete,
        "nodes": nodes,
        "guests": guests,
        "devices_outside_proxmox": outside,
        "summary": {
            "nodes": len(nodes), "nodes_with_device": sum(bool(row["tailnet_devices"]) for row in nodes),
            "guests": len(guests), "guests_with_device": sum(bool(row["tailnet_devices"]) for row in guests),
            "device_offline": sum("device_offline" in row["issues"] for row in nodes + guests),
            "devices_outside_proxmox": len(outside),
        },
    }
