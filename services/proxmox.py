"""Bounded Proxmox VE API reads and governed guest power actions.

Every request uses the shared Lua-gated HTTP client. The token and host come
from machine configuration; tool arguments can select only validated node IDs,
VMIDs and one of four fixed power actions. A power action runs only after the
owner approved its exact target, sends one POST, waits for its Proxmox task and
reads the guest back before it reports success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import ipaddress
import json
import math
import re
import time
from pathlib import Path
from typing import Literal, overload
from urllib.parse import quote, urlsplit

import httpx

from services.lua_adapters.http_clients import client as lua_http_client
from services.provider_errors import ProviderResourceNotFound

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ROWS = 200
MAX_TASKS = 50
MAX_VMID = 999_999_999
NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
UPID = re.compile(r"UPID:([A-Za-z0-9][A-Za-z0-9_.-]{0,63}):[0-9A-Fa-f]{1,16}:[0-9A-Fa-f]{1,16}:[0-9A-Fa-f]{1,16}:"
                  r"[a-z0-9_]{1,32}:[^:/\s]{0,64}:[^:/\s]{1,192}:")
# Each governed action: its Proxmox endpoint, the form body sent, the guest
# states it may start from and the state that proves it worked. A reboot is
# proven by its finished task and the guest running again afterwards.
POWER_ACTIONS = {
    "start": {"endpoint": "start", "form": {}, "from": {"stopped"}, "expect": "running"},
    "shutdown": {"endpoint": "shutdown", "form": {"timeout": "60"}, "from": {"running"}, "expect": "stopped"},
    "reboot": {"endpoint": "reboot", "form": {"timeout": "60"}, "from": {"running"}, "expect": "running"},
    "force_stop": {"endpoint": "stop", "form": {}, "from": {"running", "paused"}, "expect": "stopped"},
}
POWER_TASK_DEADLINE_SECONDS = 120.0
POWER_POLL_SECONDS = (0.5, 1.0, 2.0)
TOKEN_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}@[A-Za-z0-9_.-]{1,64}![A-Za-z0-9_.-]{1,64}")
TLS_SERVER_NAME = re.compile(
    r"(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


@dataclass(frozen=True)
class ProxmoxConfig:
    base_url: str
    token_id: str
    token_secret: str = field(repr=False)
    verify_tls: bool = True
    ca_file: Path | None = None
    server_name: str | None = None
    timeout_seconds: float = 10.0

    def origin(self) -> str:
        parsed = urlsplit(self.base_url.strip())
        try:
            port = parsed.port
        except ValueError:
            raise ValueError("Proxmox URL port is invalid.") from None
        # Both tailnet shapes are valid HTTPS origins:
        #   native Proxmox: https://machine.tailXXX.ts.net:8006
        #   tailscale serve (delegated reads): https://machine.tailXXX.ts.net
        # The latter uses the implicit 443 port. Only reject an
        # out-of-range or malformed explicit port.
        if port is not None and not (1 <= port <= 65535):
            raise ValueError("Proxmox URL port is invalid.")
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
            raise ValueError("Proxmox URL must be an HTTPS origin without credentials or path.")
        hostname = parsed.hostname.lower()
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if loopback or hostname == "localhost" or hostname.endswith(".localhost"):
            raise ValueError("Proxmox must use its configured HTTPS host, not a loopback alias.")
        return self.base_url.strip().rstrip("/")


def _invalid() -> RuntimeError:
    return RuntimeError("Proxmox returned an invalid answer.")


class ProxmoxRefused(RuntimeError):
    """Proxmox answered a read with an error status other than 404."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Proxmox refused the read (HTTP {status_code}).")


@overload
def _text(item: dict, field: str, *, required: Literal[True], limit: int = 160) -> str: ...
@overload
def _text(item: dict, field: str, *, required: bool = False, limit: int = 160) -> str | None: ...
def _text(item: dict, field: str, *, required: bool = False, limit: int = 160) -> str | None:
    value = item.get(field)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(char) < 32 for char in value):
        raise _invalid()
    return value


def _number(item: dict, field: str) -> int | float | None:
    value = item.get(field)
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise _invalid()
    return value


def _guest_disk_used(item: dict, guest_type: str) -> int | float | None:
    """Used disk of a guest, or None when Proxmox did not measure it.

    Proxmox does not track it for QEMU VMs and sends 0 ("no info available");
    a container's is measured only while it runs, and a stopped one sends 0 too.
    """
    used = _number(item, "disk")
    if guest_type == "qemu" or item.get("status") != "running":
        return None
    return used


def _boolean(item: dict, field: str) -> bool | None:
    """A Proxmox flag: a JSON boolean, or the 0/1 integer Proxmox VE 9 sends."""
    value = item.get(field)
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise _invalid()


QEMU_MEMORY = re.compile(r"(?:current=)?([0-9]{1,9})")

# Operator metadata: a guest hosts a database when its Proxmox tags include
# `db` or `db-<engine>` (for example db-postgres). Names are never guessed from.
DATABASE_ENGINE = re.compile(r"[a-z0-9][a-z0-9_.+-]{0,31}")
# Fixed order, most urgent first. A lock comes last: every backup run locks
# its guest while the guest keeps serving.
DATABASE_ISSUES = ("not_running", "disk_nearly_full", "no_backup_job", "locked")
DISK_NEARLY_FULL = 0.9
DATABASE_LABEL_HINT = ('Tag a VM or container "db" or "db-<engine>" (for example "db-postgres") '
                       'in Proxmox to list it here.')
GUEST_DISK_CONTENT = frozenset({"images", "rootdir"})
# An SSD that has used this much of its rated life is named (Proxmox shows
# the same figure as 100 - wearout).
DISK_WORN = 90
# A disk device path as Proxmox lists it: /dev/<name>, no traversal or shell text.
DISK_DEVICE = re.compile(r"/dev/[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
# The fixed SMART numbers that may leave the service, from NVMe/SAS text lines
# and from ATA attribute names (the raw value's leading integer).
SMART_TEXT_MEASURES = {"Temperature": "temperature_celsius", "Power On Hours": "power_on_hours",
                       "Power Cycles": "power_cycles", "Unsafe Shutdowns": "unsafe_shutdowns",
                       "Media and Data Integrity Errors": "media_errors",
                       "Available Spare": "available_spare_percent",
                       "Available Spare Threshold": "available_spare_threshold_percent"}
SMART_ATA_MEASURES = {"Temperature_Celsius": "temperature_celsius", "Power_On_Hours": "power_on_hours",
                      "Power_Cycle_Count": "power_cycles", "Reallocated_Sector_Ct": "reallocated_sectors",
                      "Current_Pending_Sector": "pending_sectors", "Offline_Uncorrectable": "offline_uncorrectable"}
SMART_MEASURES = ("temperature_celsius", "power_on_hours", "power_cycles", "unsafe_shutdowns", "media_errors",
                  "available_spare_percent", "available_spare_threshold_percent", "reallocated_sectors",
                  "pending_sectors", "offline_uncorrectable")
SMART_LEADING_INTEGER = re.compile(r"\s*([0-9][0-9,]{0,24})")
# A guest whose newest backup is older than this is named: a daily or weekly
# job has missed at least one run.
BACKUP_STALE_DAYS = 8
# Most urgent first: no backup at all, a scheduled run that left no copy, an
# overdue one (guests without a judged job), then no job to make the next.
BACKUP_ISSUES = ("never_backed_up", "missed_run", "backup_stale", "no_backup_job")
MAX_BACKUP_STORAGES = 20
MAX_BACKUP_VOLUMES = 5000
# A guest in an enabled, scheduled backup job is judged against that job: its
# newest copy must be newer than the job's last run that was due at least this
# long ago (a run can take hours). Proxmox computes the schedule's events itself.
MISSED_RUN_GRACE_HOURS = 12
MAX_BACKUP_JOBS = 50
MAX_SCHEDULE_EVENTS = 100
BACKUP_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
DAY_SECONDS = 86_400
# HA: a resource id is vm:<vmid> or ct:<vmid>; a state word is lower case,
# with the spaces Proxmox uses ("not found", "clearing error flag").
HA_SID = re.compile(r"(vm|ct):([0-9]{1,9})")
HA_WORD = re.compile(r"[a-z][a-z_ ]{0,31}")
# The state word in a manager or node status text ("pve-a (active, ...)"),
# as pve-ha-manager's status handler writes it, mapped to one fixed word.
HA_STATES = {"active": "active", "idle": "idle", "old timestamp - dead?": "not_responding",
             "detected time drift!": "clock_drift", "wait_for_agent_lock": "waiting_for_lock",
             "lost_agent_lock": "lost_lock", "maintenance": "maintenance", "maintenance mode": "maintenance",
             "disarm mode": "disarmed", "restart mode": "restarting", "shutdown mode": "shutting_down"}
# Deliberate or steady: anything else (in transition, unreadable, unknown) is a problem.
HA_STEADY_STATES = frozenset({"active", "idle", "maintenance", "disarmed"})
HA_RESOURCE_PROBLEMS = frozenset({"error", "fence", "recovery", "not found", "freeze", "deleting"})


def _ha_state(status: str, node: str) -> str:
    """The fixed state word in a status text such as "pve-a (active, watchdog active, <local time>)"."""
    prefix = node + " ("
    if status == f"{node} (unable to read lrm status)":
        return "unreadable"
    if not status.startswith(prefix):
        return "unknown"
    return HA_STATES.get(status[len(prefix):].split(",", 1)[0].split(")", 1)[0].strip(), "unknown")


def _disk_health(value) -> str:
    """SMART health as one fixed word: ATA says PASSED, SAS/NVMe say OK."""
    if value in (None, ""):
        return "unknown"
    if not isinstance(value, str) or len(value) > 64:
        raise _invalid()
    word = value.strip().upper()
    if word in ("PASSED", "OK"):
        return "passed"
    if word.startswith("FAILED"):
        return "failed"
    if word == "SMART DISABLED":
        return "smart_disabled"
    return "unknown"


def database_tags(tags: str | None) -> tuple[bool, str | None]:
    """Whether Proxmox tags mark a database host, and its engine when one is named."""
    found, engine = False, None
    for tag in re.split(r"[;,\s]+", (tags or "").lower()):
        if tag == "db":
            found = True
        elif tag.startswith("db-") and DATABASE_ENGINE.fullmatch(tag[3:]):
            found = True
            engine = engine or tag[3:]
    return found, engine


def _tags(item: dict) -> str | None:
    """A guest's Proxmox tags; an empty value means none."""
    if item.get("tags") in (None, ""):
        return None
    return _text(item, "tags", limit=2048)


def _memory_mib(item: dict, guest_type: str) -> int | float | None:
    """Configured memory in MiB. QEMU sends a property string such as "4096"
    or "current=4096" (Proxmox VE 8.1+); LXC sends an integer."""
    value = item.get("memory")
    if guest_type == "qemu" and isinstance(value, str):
        match = QEMU_MEMORY.fullmatch(value.split(",", 1)[0].strip())
        if match is None:
            raise _invalid()
        return int(match.group(1))
    return _number(item, "memory")


class ProxmoxService:
    def __init__(self, config: ProxmoxConfig, client: httpx.Client | None = None):
        self.config = config
        self.origin = config.origin()
        if not TOKEN_ID.fullmatch(config.token_id) or not config.token_secret or len(config.token_secret) > 4096:
            raise ValueError("Proxmox API token configuration is invalid.")
        if config.server_name is not None and not TLS_SERVER_NAME.fullmatch(config.server_name):
            raise ValueError("Proxmox TLS server name is invalid.")
        verify = str(config.ca_file) if config.ca_file is not None else config.verify_tls
        self.client = client or lua_http_client(verify=verify, timeout=config.timeout_seconds)
        self.owns_client = client is None

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        # The cluster certificate names the host (for example
        # hummingbird.local.pve) while the URL may use its LAN address. Keep
        # strict verification and tell TLS which name to verify against.
        extensions = {"sni_hostname": self.config.server_name} if self.config.server_name else {}
        request = httpx.Request("GET", self.origin + "/api2/json" + path, params=params,
                                headers={"Authorization": f"PVEAPIToken={self.config.token_id}={self.config.token_secret}",
                                         "Accept": "application/json"},
                                extensions=extensions)
        with self.client.stream(request.method, request.url, headers=request.headers,
                                extensions=request.extensions) as response:
            if response.status_code == 404:
                raise ProviderResourceNotFound("Proxmox did not find the requested resource.")
            if response.status_code != 200:
                raise ProxmoxRefused(response.status_code)
            payload = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                payload.extend(chunk)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("Proxmox's answer exceeds the bounded response size.")
        try:
            body = json.loads(payload)
        except (ValueError, UnicodeError):
            raise _invalid() from None
        if not isinstance(body, dict) or "data" not in body or type(body["data"]) not in (dict, list):
            raise _invalid()
        return body["data"]

    @staticmethod
    def _rows(value, *, required: tuple[str, ...] = ()) -> list[dict]:
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise _invalid()
        rows = value
        if any(not all(isinstance(row.get(field), str) and row[field] for field in required) for row in rows):
            raise _invalid()
        return rows

    def _result(self, data: dict) -> dict:
        return {"ok": True, "status": "success", "provider": "proxmox",
                "observed_at": datetime.now(timezone.utc).isoformat(), "data": data}

    def cluster_status(self) -> dict:
        rows = self._rows(self._get("/cluster/status"))
        result = [{"id": _text(row, "id", required=True), "type": _text(row, "type", required=True),
                   "name": _text(row, "name"), "online": _boolean(row, "online")}
                  for row in rows]
        return self._result({"count": len(result), "truncated": len(result) > MAX_ROWS, "members": result[:MAX_ROWS]})

    @staticmethod
    def _node_rows(rows: list[dict]) -> dict:
        nodes = [{"node": _text(row, "node", required=True), "status": _text(row, "status", required=True),
                  "uptime_seconds": _number(row, "uptime"), "cpu_fraction": _number(row, "cpu"),
                  "cpu_count": _number(row, "maxcpu"), "memory_bytes": _number(row, "mem"),
                  "memory_total_bytes": _number(row, "maxmem"), "disk_bytes": _number(row, "disk"),
                  "disk_total_bytes": _number(row, "maxdisk")}
                 for row in rows]
        nodes.sort(key=lambda row: row["node"].lower())
        return {"count": len(nodes), "truncated": len(nodes) > MAX_ROWS, "nodes": nodes[:MAX_ROWS]}

    @staticmethod
    def _guest_rows(rows: list[dict]) -> dict:
        guests = []
        for row in rows:
            vmid = row.get("vmid")
            if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                raise _invalid()
            if not all(isinstance(row.get(field), str) and row[field] for field in ("type", "node", "status")):
                raise _invalid()
            guests.append({"vmid": vmid, "node": _text(row, "node", required=True),
                           "type": _text(row, "type", required=True), "name": _text(row, "name"),
                           "status": _text(row, "status", required=True), "cpu_fraction": _number(row, "cpu"),
                           "memory_bytes": _number(row, "mem"), "memory_total_bytes": _number(row, "maxmem"),
                           "disk_bytes": _guest_disk_used(row, row["type"]), "disk_total_bytes": _number(row, "maxdisk"),
                           "uptime_seconds": _number(row, "uptime")})
        guests.sort(key=lambda row: (row["node"].lower(), row["vmid"]))
        return {"count": len(guests), "truncated": len(guests) > MAX_ROWS, "guests": guests[:MAX_ROWS]}

    def nodes(self) -> dict:
        rows = self._rows(self._get("/nodes"), required=("node", "status"))
        return self._result(self._node_rows(rows))

    def guests(self) -> dict:
        rows = self._rows(self._get("/cluster/resources", {"type": "vm"}), required=("type", "node", "status"))
        return self._result(self._guest_rows(rows))

    def machines(self) -> dict:
        """Every node and every guest (templates left out) by name, from one cluster read.

        Names and states only: what the tailnet coverage read matches against.
        """
        nodes, guests = [], []
        for row in self._rows(self._get("/cluster/resources"), required=("type",)):
            if row["type"] == "node":
                nodes.append({"node": _text(row, "node", required=True), "status": _text(row, "status") or "unknown"})
            elif row["type"] in ("qemu", "lxc") and not _boolean(row, "template"):
                vmid = row.get("vmid")
                if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                    raise _invalid()
                guests.append({"vmid": vmid, "node": _text(row, "node", required=True), "type": row["type"],
                               "name": _text(row, "name"), "status": _text(row, "status") or "unknown"})
        nodes.sort(key=lambda row: row["node"].lower())
        guests.sort(key=lambda row: (row["node"].lower(), row["vmid"]))
        return self._result({"truncated": len(nodes) > MAX_ROWS or len(guests) > MAX_ROWS,
                             "nodes": nodes[:MAX_ROWS], "guests": guests[:MAX_ROWS]})

    def _inventory_guest(self, node: str, vmid: int) -> dict:
        """The one inventory row for an exact node and VMID (its type is qemu or lxc)."""
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
            raise ValueError("Enter one valid Proxmox VMID.")
        rows = self._rows(self._get("/cluster/resources", {"type": "vm"}), required=("type", "node", "status"))
        matches = []
        for row in rows:
            row_vmid = row.get("vmid")
            if type(row_vmid) is not int or not (1 <= row_vmid <= MAX_VMID):
                raise _invalid()
            if row["node"] == node and row_vmid == vmid:
                matches.append(row)
        if not matches:
            raise ProviderResourceNotFound(f"Proxmox VMID {vmid} was not found on node {node}.")
        if len(matches) != 1 or matches[0]["type"] not in {"qemu", "lxc"}:
            raise _invalid()
        return matches[0]

    def _guest_type(self, node: str, vmid: int) -> str:
        return self._inventory_guest(node, vmid)["type"]

    @staticmethod
    def _validate_guest_identity(data: dict, node: str, vmid: int) -> None:
        returned_node = data.get("node")
        returned_vmid = data.get("vmid")
        if returned_node is not None and returned_node != node:
            raise _invalid()
        if returned_vmid is not None and (type(returned_vmid) is not int or returned_vmid != vmid):
            raise _invalid()

    def guest_status(self, node: str, vmid: int) -> dict:
        guest_type = self._guest_type(node, vmid)
        data = self._get(f"/nodes/{node}/{guest_type}/{vmid}/status/current")
        if not isinstance(data, dict):
            raise _invalid()
        self._validate_guest_identity(data, node, vmid)
        status = {
            "schema": "proxmox-guest-status.v1",
            "node": node,
            "vmid": vmid,
            "type": guest_type,
            "name": _text(data, "name"),
            "status": _text(data, "status", required=True),
            "uptime_seconds": _number(data, "uptime"),
            "cpu_fraction": _number(data, "cpu"),
            "cpu_count": _number(data, "cpus"),
            "memory_bytes": _number(data, "mem"),
            "memory_total_bytes": _number(data, "maxmem"),
            "disk_bytes": _guest_disk_used(data, guest_type),
            "disk_total_bytes": _number(data, "maxdisk"),
            "disk_read_bytes": _number(data, "diskread"),
            "disk_write_bytes": _number(data, "diskwrite"),
            "network_in_bytes": _number(data, "netin"),
            "network_out_bytes": _number(data, "netout"),
            # A lock (backup, migrate, snapshot, ...) means Proxmox refuses most
            # power actions until it clears.
            "lock": _text(data, "lock"),
        }
        return self._result(status)

    def guest_config(self, node: str, vmid: int) -> dict:
        guest_type = self._guest_type(node, vmid)
        data = self._get(f"/nodes/{node}/{guest_type}/{vmid}/config")
        if not isinstance(data, dict):
            raise _invalid()
        self._validate_guest_identity(data, node, vmid)
        # Only expose a reviewed allow-list. Proxmox configs may also contain
        # credentials, SSH keys and private network settings; those stay local.
        config = {
            "schema": "proxmox-guest-config.v1",
            "node": node,
            "vmid": vmid,
            "type": guest_type,
            "name": _text(data, "name" if guest_type == "qemu" else "hostname"),
            "cores": _number(data, "cores"),
            "sockets": _number(data, "sockets") if guest_type == "qemu" else None,
            "memory_mib": _memory_mib(data, guest_type),
            "swap_mib": _number(data, "swap") if guest_type == "lxc" else None,
            "cpu_type": _text(data, "cpu"),
            "machine": _text(data, "machine") if guest_type == "qemu" else None,
            "os_type": _text(data, "ostype"),
            "bios": _text(data, "bios") if guest_type == "qemu" else None,
            "boot_order": _text(data, "boot"),
            "on_boot": _boolean(data, "onboot"),
            "template": _boolean(data, "template"),
            "tags": _text(data, "tags", limit=2048),
        }
        return self._result(config)

    def guest_snapshots(self, node: str, vmid: int) -> dict:
        """One guest's snapshots, newest first: name, time, age, parent and whether RAM was saved.

        Proxmox lists a synthetic "current" entry for where the guest stands;
        it is not a snapshot. Descriptions are operator free text and stay here.
        """
        guest = self._inventory_guest(node, vmid)
        rows = self._rows(self._get(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot"))
        now = time.time()
        snapshots, current_parent = [], None
        for row in rows:
            name = _text(row, "name", required=True)
            parent = _text(row, "parent")
            if name == "current":
                current_parent = parent
                continue
            created = row.get("snaptime")
            if created is not None and (type(created) is not int or created < 0):
                raise _invalid()
            # Proxmox sends 0 when the config records no time; that is unknown, not 1970.
            created = created or None
            snapshots.append({"name": name, "time": created,
                              "age_days": None if created is None else round(max(0.0, now - created) / 86_400, 1),
                              "parent": parent, "includes_ram": _boolean(row, "vmstate")})
        snapshots.sort(key=lambda row: (row["time"] is None, -(row["time"] or 0), row["name"]))
        ages = [row["age_days"] for row in snapshots if row["age_days"] is not None]
        return self._result({
            "schema": "proxmox-guest-snapshots.v1", "node": node, "vmid": vmid, "type": guest["type"],
            "name": _text(guest, "name"), "count": len(snapshots), "truncated": len(snapshots) > MAX_ROWS,
            "current_parent": current_parent, "oldest_age_days": max(ages) if ages else None,
            "snapshots": snapshots[:MAX_ROWS],
        })

    def node_storage(self, node: str) -> dict:
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        rows = self._rows(self._get(f"/nodes/{node}/storage"), required=("storage", "type"))
        storage = [{"storage": _text(row, "storage", required=True), "type": _text(row, "type", required=True),
                    "content": _text(row, "content"), "active": _boolean(row, "active"),
                    "enabled": _boolean(row, "enabled"),
                    "used_bytes": _number(row, "used"), "total_bytes": _number(row, "total")}
                   for row in rows]
        storage.sort(key=lambda row: row["storage"].lower())
        return self._result({"node": node, "count": len(storage), "truncated": len(storage) > MAX_ROWS,
                             "storage": storage[:MAX_ROWS]})

    def recent_tasks(self, node: str) -> dict:
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        rows = self._rows(self._get(f"/nodes/{node}/tasks", {"limit": MAX_TASKS}), required=("upid", "type"))
        tasks = [{"upid": _text(row, "upid", required=True, limit=512), "type": _text(row, "type", required=True),
                  "status": _text(row, "status"), "user": _text(row, "user"),
                  "start_time": _number(row, "starttime"), "end_time": _number(row, "endtime")}
                 for row in rows]
        return self._result({"node": node, "count": len(tasks), "truncated": len(tasks) > MAX_TASKS,
                             "tasks": tasks[:MAX_TASKS]})

    def _ha_rows(self, rows: list) -> dict:
        quorate, fencing, manager, nodes, resources = None, None, None, [], []
        for row in rows:
            if not isinstance(row, dict):
                raise _invalid()
            kind = _text(row, "type", required=True)
            if kind == "quorum":
                quorate = _boolean(row, "quorate")
            elif kind == "fencing":
                fencing = _text(row, "armed-state", limit=32)
                if fencing is not None and not HA_WORD.fullmatch(fencing):
                    raise _invalid()
            elif kind in ("master", "lrm"):
                node = _text(row, "node", required=True)
                if not NODE_ID.fullmatch(node):
                    raise _invalid()
                state = _ha_state(_text(row, "status", limit=512) or "", node)
                entry = {"node": node, "state": state, "problem": state not in HA_STEADY_STATES}
                if kind == "master":
                    manager = entry
                else:
                    nodes.append(entry)
            elif kind == "service":
                match = HA_SID.fullmatch(_text(row, "sid", required=True, limit=32))
                node, state = _text(row, "node"), _text(row, "state", required=True, limit=32)
                requested = _text(row, "request_state", limit=32)
                if (match is None or not 1 <= int(match.group(2)) <= MAX_VMID
                        or (node is not None and not NODE_ID.fullmatch(node))
                        or not HA_WORD.fullmatch(state) or (requested is not None and not HA_WORD.fullmatch(requested))):
                    raise _invalid()
                resources.append({"sid": match.group(0), "type": "qemu" if match.group(1) == "vm" else "lxc",
                                  "vmid": int(match.group(2)), "node": node, "state": state,
                                  "requested": requested, "problem": state in HA_RESOURCE_PROBLEMS})
        nodes.sort(key=lambda row: row["node"].lower())
        resources.sort(key=lambda row: (not row["problem"], row["vmid"]))
        return {
            "schema": "proxmox-ha-status.v1", "quorate": quorate, "fencing": fencing,
            "in_use": manager is not None or bool(nodes) or bool(resources),
            "manager": manager, "nodes": nodes[:MAX_ROWS], "count": len(resources),
            "truncated": len(resources) > MAX_ROWS, "resources": resources[:MAX_ROWS],
            "summary": {"nodes": len(nodes), "nodes_with_problems": sum(row["problem"] for row in nodes),
                        "resources": len(resources),
                        "resources_with_problems": sum(row["problem"] for row in resources)}}

    def ha_status(self) -> dict:
        """High availability: quorum, the HA manager, each node's HA state and each HA resource, in fixed words.

        Proxmox puts the manager's and nodes' state inside a status text that
        also carries the node's local time; only the mapped state word leaves
        the service. Resource comments are operator free text and stay here.
        """
        if not isinstance(rows := self._get("/cluster/ha/status/current"), list):
            raise _invalid()
        return self._result(self._ha_rows(rows))

    def node_network(self, node: str) -> dict:
        """One node's network interfaces: name, type, reported state, IP methods and bridge/bond ports.

        Addresses, netmasks, gateways, comments and alternative names (which
        can embed a MAC) never leave the service. Proxmox's `active` comes
        from the interfaces file it last parsed and can lag the link, so it is
        "reported", and an interface it says nothing about is None, not down.
        """
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        rows = self._rows(self._get(f"/nodes/{node}/network"), required=("iface", "type"))
        interfaces = []
        for row in rows:
            ports_field = "slaves" if row["type"] in ("bond", "OVSBond") else "bridge_ports"
            ports_text = _text(row, ports_field, limit=1024)
            ports = [port for port in (ports_text or "").split() if port != "none"]
            if any(not NODE_ID.fullmatch(port) for port in ports):
                raise _invalid()
            mtu = row.get("mtu")
            if mtu is not None and (type(mtu) is not int or not 576 <= mtu <= 65_520):
                raise _invalid()
            interfaces.append({
                "name": _text(row, "iface", required=True), "type": _text(row, "type", required=True),
                "active": _boolean(row, "active"), "autostart": _boolean(row, "autostart"),
                "ipv4": _text(row, "method"), "ipv6": _text(row, "method6"),
                "has_ipv4_address": bool(row.get("address") or row.get("cidr")),
                "ports": ports, "bond_mode": _text(row, "bond_mode"), "mtu": mtu,
                "vlan_aware": _boolean(row, "bridge_vlan_aware"),
            })
        groups = {"bridge": 0, "OVSBridge": 0, "bond": 1, "OVSBond": 1}
        interfaces.sort(key=lambda row: (groups.get(row["type"], 2), row["name"]))
        names = {row["name"] for row in interfaces}
        for row in interfaces:
            row["member_of"] = [parent["name"] for parent in interfaces if row["name"] in parent["ports"]]
        missing = sorted({port for row in interfaces for port in row["ports"]} - names)
        return self._result({
            "schema": "proxmox-node-network.v1", "node": node, "count": len(interfaces),
            "truncated": len(interfaces) > MAX_ROWS,
            "summary": {"interfaces": len(interfaces),
                        "bridges": sum(groups.get(row["type"]) == 0 for row in interfaces),
                        "bonds": sum(groups.get(row["type"]) == 1 for row in interfaces),
                        "physical": sum(row["type"] == "eth" for row in interfaces),
                        "reported_active": sum(row["active"] is True for row in interfaces),
                        "reported_inactive": sum(row["active"] is False for row in interfaces),
                        "not_reported": sum(row["active"] is None for row in interfaces)},
            "missing_ports": missing,
            "interfaces": interfaces[:MAX_ROWS]})

    def node_disks(self, node: str) -> dict:
        """Physical disks of one node: health and wear, never serials, WWNs or by-id links."""
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        disks = []
        for row in self._rows(self._get(f"/nodes/{node}/disks/list"), required=("devpath", "type")):
            device = _text(row, "devpath", required=True) or ""
            if not device.startswith("/dev/"):
                raise _invalid()
            size = _number(row, "size")
            wearout = row.get("wearout")
            # Proxmox's wearout is the life remaining in percent ("N/A" when unknown).
            if isinstance(wearout, (int, float)) and not isinstance(wearout, bool):
                if not 0 <= wearout <= 100:
                    raise _invalid()
                wear_used = round(100 - wearout)
            elif wearout in (None, "N/A"):
                wear_used = None
            else:
                raise _invalid()
            health = _disk_health(row.get("health"))
            issues = [issue for issue, present in (("smart_failed", health == "failed"),
                                                    ("worn_out", wear_used is not None and wear_used >= DISK_WORN))
                      if present]
            disks.append({"device": device, "type": _text(row, "type", required=True), "model": _text(row, "model"),
                          "size_bytes": size, "health": health, "wear_used_percent": wear_used,
                          "used_by": _text(row, "used"), "issues": issues})
        disks.sort(key=lambda row: (0 if "smart_failed" in row["issues"] else 1 if row["issues"] else 2,
                                    row["device"]))
        return self._result({
            "node": node, "count": len(disks), "truncated": len(disks) > MAX_ROWS,
            "summary": {"disks": len(disks),
                        "failing": sum("smart_failed" in row["issues"] for row in disks),
                        "worn": sum("worn_out" in row["issues"] for row in disks),
                        "health_unknown": sum(row["health"] == "unknown" for row in disks)},
            "disks": disks[:MAX_ROWS]})

    def disk_smart(self, node: str, device: str) -> dict:
        """One listed disk's SMART health and a fixed set of its numbers; never serials or the raw text.

        smartctl runs on the node for this read, so it is asked for one disk at a
        time. The device must be one the node's disk list reports.
        """
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        if isinstance(device, str) and not device.startswith("/"):
            device = "/dev/" + device
        if not isinstance(device, str) or not DISK_DEVICE.fullmatch(device):
            raise ValueError("Enter one exact disk device path, for example /dev/sda.")
        listed = {(_text(row, "devpath", required=True)): row
                  for row in self._rows(self._get(f"/nodes/{node}/disks/list"), required=("devpath", "type"))}
        if device not in listed:
            raise ValueError(f"{device} is not a disk Proxmox lists on node {node}.")
        answer = self._get(f"/nodes/{node}/disks/smart", {"disk": device})
        if not isinstance(answer, dict):
            raise _invalid()
        measures: dict[str, int | None] = dict.fromkeys(SMART_MEASURES)
        text, attributes = answer.get("text"), answer.get("attributes")
        if text is not None:
            if not isinstance(text, str) or len(text) > 65_536:
                raise _invalid()
            for line in text.splitlines():
                label, _, value = line.partition(":")
                key = SMART_TEXT_MEASURES.get(label.strip())
                if key is not None and (match := SMART_LEADING_INTEGER.match(value)):
                    measures[key] = int(match.group(1).replace(",", ""))
        wear_used = None
        if text is not None:
            match = re.search(r"Percentage Used(?: endurance indicator)?:\s*([0-9]{1,3})%", text)
            wear_used = int(match.group(1)) if match else None
        if attributes is not None:
            if not isinstance(attributes, list) or len(attributes) > 256:
                raise _invalid()
            for row in attributes:
                if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("raw"), str):
                    raise _invalid()
                key = SMART_ATA_MEASURES.get(row["name"])
                if key is not None and (match := SMART_LEADING_INTEGER.match(row["raw"])):
                    measures[key] = int(match.group(1).replace(",", ""))
        listed_row = listed[device]
        wearout = listed_row.get("wearout")
        if wear_used is None and isinstance(wearout, (int, float)) and not isinstance(wearout, bool) and 0 <= wearout <= 100:
            wear_used = round(100 - wearout)
        spare, threshold = measures["available_spare_percent"], measures["available_spare_threshold_percent"]
        issues = [issue for issue, present in (
            ("media_errors", bool(measures["media_errors"])),
            ("reallocated_sectors", bool(measures["reallocated_sectors"])),
            ("pending_sectors", bool(measures["pending_sectors"])),
            ("offline_uncorrectable", bool(measures["offline_uncorrectable"])),
            ("spare_below_threshold", spare is not None and threshold is not None and spare < threshold)) if present]
        return self._result({
            "schema": "proxmox-disk-smart.v1", "node": node, "device": device,
            "type": _text(listed_row, "type", required=True), "model": _text(listed_row, "model"),
            "health": _disk_health(answer.get("health")), "wear_used_percent": wear_used,
            "measures": measures, "issues": issues})

    def overview(self) -> dict:
        """One bounded dashboard snapshot from three provider reads.

        /cluster/resources without a type filter returns the node rows, the
        guest rows and the storage rows together, so nodes, guests and the
        database hosts come from the same read; backup coverage is the third.
        """
        cluster = self.cluster_status()["data"]
        rows = self._rows(self._get("/cluster/resources"), required=("type",))
        node_rows = [row for row in rows if row["type"] == "node"]
        if not all(isinstance(row.get(field), str) and row[field] for row in node_rows for field in ("node", "status")):
            raise _invalid()
        return self._result({
            "schema": "proxmox-overview.v1",
            "cluster": cluster,
            "nodes": self._node_rows(node_rows),
            "guests": self._guest_rows([row for row in rows if row["type"] in {"qemu", "lxc"}]),
            "databases": self._database_rows(rows),
        })

    def _backup_coverage(self) -> tuple[set[int] | None, str | None]:
        """VMIDs no backup job covers, or None with why coverage is unknown.

        The list needs Sys.Audit on /; a refusal or no answer leaves coverage
        unknown instead of failing the inventory. A malformed answer is refused.
        """
        try:
            rows = self._get("/cluster/backup-info/not-backed-up")
        except ProxmoxRefused as error:
            return None, (f"Proxmox refused the backup coverage read (HTTP {error.status_code}); "
                          "the token needs Sys.Audit on / to see backup jobs.")
        except ProviderResourceNotFound:
            return None, "This Proxmox version does not report backup coverage."
        except httpx.TransportError:
            return None, "Proxmox did not answer the backup coverage read."
        uncovered = set()
        for row in self._rows(rows):
            vmid = row.get("vmid")
            if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                raise _invalid()
            uncovered.add(vmid)
        return uncovered, None

    def _database_rows(self, rows: list[dict]) -> dict:
        """Guests tagged as database hosts, their backup coverage, issues and guest-disk storage."""
        databases = []
        for row in rows:
            if row["type"] not in ("qemu", "lxc"):
                continue
            vmid = row.get("vmid")
            if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                raise _invalid()
            node, status = _text(row, "node", required=True), _text(row, "status", required=True)
            is_database, engine = database_tags(_tags(row))
            # A template cannot run, so it hosts nothing.
            if not is_database or _boolean(row, "template"):
                continue
            databases.append({
                "vmid": vmid, "node": node, "type": row["type"], "name": _text(row, "name"), "engine": engine,
                "status": status, "lock": _text(row, "lock"),
                "memory_bytes": _number(row, "mem"), "memory_total_bytes": _number(row, "maxmem"),
                "disk_bytes": _guest_disk_used(row, row["type"]), "disk_total_bytes": _number(row, "maxdisk"),
                "uptime_seconds": _number(row, "uptime"), "backup_job": None, "issues": [],
            })
        uncovered, coverage_note = self._backup_coverage()
        for database in databases:
            if uncovered is not None:
                database["backup_job"] = database["vmid"] not in uncovered
            used, total = database["disk_bytes"], database["disk_total_bytes"]
            found = {"not_running" if database["status"] != "running" else "",
                     "locked" if database["lock"] else "",
                     "no_backup_job" if database["backup_job"] is False else "",
                     "disk_nearly_full" if used is not None and total and used / total >= DISK_NEARLY_FULL else ""}
            database["issues"] = [issue for issue in DATABASE_ISSUES if issue in found]
        # A stopped host first, then hosts with any other issue, then healthy ones.
        databases.sort(key=lambda row: (0 if "not_running" in row["issues"] else 1 if row["issues"] else 2,
                                        DATABASE_ISSUES.index(row["issues"][0]) if row["issues"] else 0,
                                        row["node"].lower(), row["vmid"]))
        hosting = {database["node"] for database in databases}
        storage = []
        for row in rows:
            if row["type"] != "storage":
                continue
            node = _text(row, "node", required=True)
            content = set((_text(row, "content", limit=512) or "").split(","))
            if node not in hosting or not GUEST_DISK_CONTENT & content:
                continue
            storage.append({"node": node, "storage": _text(row, "storage", required=True),
                            "type": _text(row, "plugintype"), "shared": _boolean(row, "shared"),
                            "status": _text(row, "status"), "used_bytes": _number(row, "disk"),
                            "total_bytes": _number(row, "maxdisk")})
        storage.sort(key=lambda row: (row["node"].lower(), row["storage"].lower()))
        return {
            "schema": "proxmox-database-inventory.v1",
            "count": len(databases),
            "truncated": len(databases) > MAX_ROWS,
            "summary": {
                "running": sum(row["status"] == "running" for row in databases),
                "not_running": sum(row["status"] != "running" for row in databases),
                "without_backup_job": None if uncovered is None else sum(row["backup_job"] is False
                                                                         for row in databases),
                "with_issues": sum(bool(row["issues"]) for row in databases),
            },
            "backup_coverage_known": uncovered is not None,
            "backup_coverage_note": coverage_note,
            "databases": databases[:MAX_ROWS],
            "storage": storage[:MAX_ROWS],
            "label_hint": None if databases else DATABASE_LABEL_HINT,
        }

    def database_inventory(self) -> dict:
        """Database hosts from operator tags, with observed state, capacity and backup coverage (two reads)."""
        rows = self._rows(self._get("/cluster/resources"), required=("type",))
        return self._result(self._database_rows(rows))

    def _backup_volumes(self, node: str, storage: str) -> list[dict]:
        rows = self._rows(self._get(f"/nodes/{node}/storage/{quote(storage, safe='')}/content",
                                    {"content": "backup"}))
        if len(rows) > MAX_BACKUP_VOLUMES:
            raise _invalid()
        volumes = []
        for row in rows:
            vmid, created = row.get("vmid"), row.get("ctime")
            if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                raise _invalid()
            if type(created) is not int or created < 0:
                raise _invalid()
            verification = row.get("verification")
            if verification is not None and not (isinstance(verification, dict)
                                                 and isinstance(verification.get("state"), str)):
                raise _invalid()
            volumes.append({"vmid": vmid, "time": created, "size_bytes": _number(row, "size"),
                            "protected": bool(row.get("protected")), "encrypted": bool(row.get("encrypted")),
                            "verified": verification["state"][:16] if verification else None})
        return volumes

    @staticmethod
    def _included_guests(answer) -> set[int]:
        """VMIDs a backup job includes, from Proxmox's own answer (a tree whose children are guests)."""
        children = answer.get("children") if isinstance(answer, dict) else None
        if not isinstance(children, list) or len(children) > MAX_BACKUP_VOLUMES:
            raise _invalid()
        vmids = set()
        for child in children:
            vmid = child.get("id") if isinstance(child, dict) else None
            if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                raise _invalid()
            vmids.add(vmid)
        return vmids

    def _last_run_before(self, schedule: str, cutoff: int) -> int | None:
        """The latest run of a backup schedule at or before cutoff, as Proxmox computes it.

        Proxmox computes calendar events only forward from a start time. The
        search starts 32 days back (one read for daily, weekly and monthly
        jobs), narrows when one page of events cannot reach the cutoff and
        widens to a year for a rarer schedule. A page that never reaches the
        cutoff gives its latest event: an earlier due time flags fewer guests,
        never more. None: no run within a year, or Proxmox cannot analyze it.
        """
        window, best = 32 * DAY_SECONDS, None
        for _ in range(4):
            try:
                rows = self._get("/cluster/jobs/schedule-analyze", {
                    "schedule": schedule, "starttime": cutoff - window, "iterations": MAX_SCHEDULE_EVENTS})
            except (ProxmoxRefused, ProviderResourceNotFound, httpx.TransportError):
                return None
            if not isinstance(rows, list) or len(rows) > MAX_SCHEDULE_EVENTS:
                raise _invalid()
            stamps = [row.get("timestamp") if isinstance(row, dict) else None for row in rows]
            if any(type(stamp) is not int or stamp < 0 for stamp in stamps):
                raise _invalid()
            before = [stamp for stamp in stamps if stamp <= cutoff]
            best = max([*before, *([best] if best is not None else [])], default=None)
            if len(stamps) < MAX_SCHEDULE_EVENTS or max(stamps) > cutoff:
                if before or window >= 366 * DAY_SECONDS:
                    return best
                window = 366 * DAY_SECONDS
            elif window > DAY_SECONDS:
                window = DAY_SECONDS
            elif window > 3600:
                window = 3600
            else:
                return best
        return best

    def _last_scheduled_runs(self, jobs: list[dict], vmids: set[int], now: float) -> dict[int, int]:
        """Each guest's most recent backup run that was due at least MISSED_RUN_GRACE_HOURS ago.

        Only enabled jobs with a schedule count. Proxmox says which guests a job
        includes and when its schedule fires; a job whose guests or schedule
        cannot be read is left out, and its guests are judged by age instead.
        """
        if len(jobs) > MAX_BACKUP_JOBS:
            raise _invalid()
        cutoff = int(now) - MISSED_RUN_GRACE_HOURS * 3600
        due_by_schedule: dict[str, int | None] = {}
        result: dict[int, int] = {}
        for job in jobs:
            job_id, schedule = job.get("id"), job.get("schedule")
            if (job.get("enabled", 1) in (0, False, "0") or not isinstance(schedule, str) or not schedule
                    or len(schedule) > 128 or not isinstance(job_id, str) or not BACKUP_JOB_ID.fullmatch(job_id)):
                continue
            try:
                members = self._included_guests(self._get(f"/cluster/backup/{job_id}/included_volumes")) & vmids
            except (ProxmoxRefused, ProviderResourceNotFound, httpx.TransportError):
                continue
            if not members:
                continue
            if schedule not in due_by_schedule:
                due_by_schedule[schedule] = self._last_run_before(schedule, cutoff)
            due = due_by_schedule[schedule]
            if due is None:
                continue
            for vmid in members:
                result[vmid] = max(result.get(vmid, 0), due)
        return result

    def backups(self) -> dict:
        """The newest backup of each guest across the backup storages, with job coverage and what is overdue.

        Volume ids, notes and paths stay inside the service. A shared storage is
        read once. A storage that cannot be read makes the listing incomplete,
        so no guest is then called never backed up or stale.
        """
        rows = self._rows(self._get("/cluster/resources"), required=("type",))
        guests = {}
        for row in rows:
            if row["type"] not in ("qemu", "lxc") or _boolean(row, "template"):
                continue
            vmid = row.get("vmid")
            if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
                raise _invalid()
            guests[vmid] = {"vmid": vmid, "node": _text(row, "node", required=True), "type": row["type"],
                            "name": _text(row, "name")}
        try:
            jobs_rows = self._rows(self._get("/cluster/backup"))
        except (ProxmoxRefused, ProviderResourceNotFound, httpx.TransportError):
            jobs_rows = None
        jobs = None if jobs_rows is None else {
            "count": len(jobs_rows), "enabled": sum(row.get("enabled", 1) not in (0, False, "0") for row in jobs_rows)}
        now = time.time()
        last_runs = self._last_scheduled_runs(jobs_rows or [], set(guests), now)
        uncovered, coverage_note = self._backup_coverage()
        nodes = sorted({_text(row, "node", required=True) or "" for row in rows if row["type"] == "node"})
        storages, unreadable, volumes, seen_shared = [], [], [], set()
        for node in nodes:
            if not NODE_ID.fullmatch(node):
                raise _invalid()
            for row in self._rows(self._get(f"/nodes/{node}/storage", {"content": "backup", "enabled": 1}),
                                  required=("storage", "type")):
                name = _text(row, "storage", required=True) or ""
                shared = bool(row.get("shared"))
                if shared and name in seen_shared:
                    continue
                if shared:
                    seen_shared.add(name)
                if len(storages) + len(unreadable) >= MAX_BACKUP_STORAGES:
                    raise _invalid()
                try:
                    found = self._backup_volumes(node, name)
                except (ProxmoxRefused, ProviderResourceNotFound, httpx.TransportError) as error:
                    reason = (str(error) if isinstance(error, ProxmoxRefused)
                              else "Proxmox did not answer the read." if isinstance(error, httpx.TransportError)
                              else "Proxmox did not find the storage.")
                    unreadable.append({"node": node, "storage": name, "reason": reason})
                    continue
                volumes.extend(dict(volume, storage=name) for volume in found)
                storages.append({"node": node, "storage": name, "type": _text(row, "type", required=True),
                                 "shared": shared, "backups": len(found)})
        complete = not unreadable
        result = []
        for vmid, guest in guests.items():
            own = [volume for volume in volumes if volume["vmid"] == vmid]
            newest = max(own, key=lambda volume: volume["time"], default=None)
            latest = None if newest is None else {
                "storage": newest["storage"], "time": newest["time"],
                "age_days": round(max(0.0, now - newest["time"]) / 86_400, 1),
                "size_bytes": newest["size_bytes"], "protected": newest["protected"],
                "encrypted": newest["encrypted"], "verified": newest["verified"]}
            backup_job = None if uncovered is None else vmid not in uncovered
            last_run = last_runs.get(vmid)
            # A guest its own job judges is never also judged by the fixed age.
            found = {"no_backup_job": backup_job is False,
                     "never_backed_up": complete and latest is None,
                     "missed_run": complete and latest is not None and last_run is not None
                     and latest["time"] < last_run,
                     "backup_stale": complete and latest is not None and last_run is None
                     and latest["age_days"] > BACKUP_STALE_DAYS}
            result.append(dict(guest, backup_job=backup_job, backups=len(own), last_scheduled_run=last_run,
                               latest=latest, issues=[issue for issue in BACKUP_ISSUES if found[issue]]))
        result.sort(key=lambda row: (BACKUP_ISSUES.index(row["issues"][0]) if row["issues"] else len(BACKUP_ISSUES),
                                     row["node"].lower(), row["vmid"]))
        return self._result({
            "schema": "proxmox-backups.v1",
            "jobs": jobs,
            "coverage_known": uncovered is not None,
            "coverage_note": coverage_note,
            "listing_complete": complete,
            "stale_after_days": BACKUP_STALE_DAYS,
            "missed_run_grace_hours": MISSED_RUN_GRACE_HOURS,
            "unreadable_storages": unreadable,
            "summary": {"guests": len(result),
                        "with_recent_backup": sum(row["latest"] is not None and not row["issues"]
                                                  for row in result),
                        "stale": sum("backup_stale" in row["issues"] for row in result),
                        "missed_run": sum("missed_run" in row["issues"] for row in result),
                        "never": sum("never_backed_up" in row["issues"] for row in result),
                        "without_backup_job": None if uncovered is None else sum(row["backup_job"] is False
                                                                                 for row in result)},
            "storages": storages,
            "count": len(result), "truncated": len(result) > MAX_ROWS,
            "guests": result[:MAX_ROWS],
        })

    # ------------------------------------------------------------------
    # Governed guest power actions
    # ------------------------------------------------------------------

    @staticmethod
    def _power_arguments(node, vmid, action) -> dict:
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
            raise ValueError("Enter one valid Proxmox VMID.")
        if not isinstance(action, str) or action not in POWER_ACTIONS:
            raise ValueError("The power action must be start, shutdown, reboot or force_stop.")
        return POWER_ACTIONS[action]

    def _guest_row(self, node: str, vmid: int) -> dict:
        """The exact guest as it is now: identity from the inventory, state from the guest itself.

        /cluster/resources is refreshed by Proxmox every few seconds, so its
        status can lag a just-finished action. Type, name and the template
        flag come from that inventory row; status and lock come from the
        guest's live status/current, which decides whether an action applies.
        """
        row = self._inventory_guest(node, vmid)
        live = self._get(f"/nodes/{node}/{row['type']}/{vmid}/status/current")
        if not isinstance(live, dict):
            raise _invalid()
        self._validate_guest_identity(live, node, vmid)
        return {"node": node, "vmid": vmid, "type": row["type"], "name": _text(row, "name"),
                "status": _text(live, "status", required=True), "lock": _text(live, "lock"),
                "template": bool(_boolean(row, "template"))}

    @staticmethod
    def _power_refusal(guest: dict, action: str) -> str | None:
        """Why Proxmox would refuse this action on the guest as it is now, or None."""
        label = f"{guest['type'].upper()} {guest['vmid']} ({guest['name'] or 'unnamed'})"
        if guest["template"]:
            return f"{label} is a template; templates cannot be powered on or off."
        if guest["lock"]:
            return f"{label} is locked ({guest['lock']}); wait until that operation finishes."
        if guest["status"] not in POWER_ACTIONS[action]["from"]:
            if action == "start" and guest["status"] == "running":
                return f"{label} is already running."
            return f"{label} is {guest['status']}, so it cannot be {action.replace('_', ' ')}."
        return None

    def power_target(self, node: str, vmid: int, action: str) -> dict:
        """Read-only: the exact guest an action would affect and whether it applies now."""
        self._power_arguments(node, vmid, action)
        guest = self._guest_row(node, vmid)
        reason = self._power_refusal(guest, action)
        return self._result({**guest, "action": action, "allowed": reason is None, "reason": reason})

    def _post(self, path: str, form: dict[str, str]) -> httpx.Response:
        headers = {"Authorization": f"PVEAPIToken={self.config.token_id}={self.config.token_secret}",
                   "Accept": "application/json"}
        if form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        extensions = {"sni_hostname": self.config.server_name} if self.config.server_name else {}
        body = "&".join(f"{key}={quote(value, safe='')}" for key, value in form.items())
        request = httpx.Request("POST", self.origin + "/api2/json" + path, headers=headers,
                                content=body.encode(), extensions=extensions)
        return self.client.send(request)

    @staticmethod
    def _post_refusal(response: httpx.Response) -> str:
        """The status and Proxmox's own short reason. Proxmox puts it in the
        HTTP reason phrase or a `message` field; never the request or token."""
        message = ""
        try:
            body = response.json()
            if isinstance(body, dict) and isinstance(body.get("message"), str):
                message = body["message"]
        except (ValueError, UnicodeError):
            pass
        if not message:
            message = response.reason_phrase or ""
        message = " ".join(message.split())[:200]
        return f"Proxmox refused the action (HTTP {response.status_code})" + (f": {message}" if message else ".")

    def guest_power(self, node: str, vmid: int, action: str, *, expected_type: str, expected_name: str | None,
                    sleep=time.sleep, clock=time.monotonic) -> dict:
        """Run one approved power action on the exact guest and prove its outcome.

        The guest must still be the type and name frozen at approval. One POST
        is sent; its task is followed to the end and the guest is read back.
        Success needs a finished task with exit status OK and the expected
        guest state. A POST that may have reached Proxmox without an answer,
        or a task still running at the deadline, is reported as an unknown
        outcome so it is reconciled instead of repeated.
        """
        spec = self._power_arguments(node, vmid, action)
        receipt = {"provider": "proxmox", "operation": "guest_power", "action": action, "node": node,
                   "vmid": vmid, "type": None, "name": None, "expected_status": spec["expect"],
                   "observed_status": None, "upid": None, "task_exit_status": None, "changed": False,
                   "verification_ok": False, "retry_safe": True}

        def outcome(ok: bool, status: str, error: str | None = None, **fields) -> dict:
            return {**receipt, **fields, "ok": ok, "status": status, "error": error,
                    "observed_at": datetime.now(timezone.utc).isoformat()}

        try:
            guest = self._guest_row(node, vmid)
        except ProviderResourceNotFound as error:
            return outcome(False, "not_found", str(error))
        except (httpx.HTTPError, RuntimeError):
            # The check before sending failed; nothing was sent, so a retry is safe.
            return outcome(False, "error", "Proxmox could not confirm the guest's current state; "
                           "nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["type"] != expected_type or guest["name"] != (expected_name or None):
            return outcome(False, "conflict", "The guest no longer matches the approved target "
                           f"({expected_type} {expected_name or 'unnamed'}); nothing was changed.")
        reason = self._power_refusal(guest, action)
        if reason is not None:
            return outcome(False, "conflict", reason + " Nothing was changed.", observed_status=guest["status"])

        path = f"/nodes/{node}/{guest['type']}/{vmid}/status/{spec['endpoint']}"
        try:
            response = self._post(path, spec["form"])
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return outcome(False, "error", "Proxmox could not be reached; the action was not sent.")
        except httpx.HTTPError:
            return outcome(False, "outcome_unknown", "Proxmox did not answer the action request; it may have run. "
                           "Check the guest before trying again.", retry_safe=False)
        if response.status_code in (401, 403):
            return outcome(False, "denied", self._post_refusal(response) + " Nothing was changed.")
        if response.status_code != 200:
            return outcome(False, "failed", self._post_refusal(response) + " Nothing was changed.")
        try:
            data = response.json().get("data")
        except (ValueError, UnicodeError, AttributeError):
            data = None
        match = UPID.fullmatch(data) if isinstance(data, str) and len(data) <= 512 else None
        if match is None or match.group(1) != node:
            return outcome(False, "outcome_unknown", "Proxmox accepted the action but did not return its task; "
                           "check the guest before trying again.", retry_safe=False)
        upid: str = match.group(0)
        receipt.update(upid=upid, changed=True, retry_safe=False)

        deadline = clock() + POWER_TASK_DEADLINE_SECONDS
        polls = 0
        task = None
        while True:
            try:
                task = self._get(f"/nodes/{node}/tasks/{quote(upid, safe='')}/status")
            except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
                task = None
            if isinstance(task, dict) and task.get("status") == "stopped":
                break
            if clock() >= deadline:
                return outcome(False, "outcome_unknown", f"Proxmox task {upid} had not finished after "
                               f"{int(POWER_TASK_DEADLINE_SECONDS)} seconds; check the guest before trying again.")
            sleep(POWER_POLL_SECONDS[min(polls, len(POWER_POLL_SECONDS) - 1)])
            polls += 1
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status

        try:
            current = self._get(f"/nodes/{node}/{guest['type']}/{vmid}/status/current")
            observed = _text(current, "status", required=True) if isinstance(current, dict) else None
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            observed = None
        receipt["observed_status"] = observed
        if exit_status != "OK":
            return outcome(False, "failed", f"Proxmox task {upid} ended with: {receipt['task_exit_status']}. "
                           f"The guest is now {observed or 'unknown'}.")
        if observed is None:
            return outcome(False, "outcome_unknown", "The Proxmox task finished, but the guest could not be read "
                           "back; check it before trying again.")
        if observed != spec["expect"]:
            return outcome(False, "failed", f"The Proxmox task finished, but the guest is {observed}, "
                           f"not {spec['expect']}.")
        return outcome(True, "executed", None, verification_ok=True)


def proxmox_integration_status(settings) -> dict:
    url = getattr(settings, "proxmox_base_url", None)
    token_id = getattr(settings, "proxmox_api_token_id", None)
    token_secret = getattr(settings, "proxmox_api_token_secret", None)
    return {"id": "proxmox", "name": "Proxmox VE", "mode": "governed_power_v1",
            "configured": bool(url and token_id and token_secret),
            "host": urlsplit(url).hostname if url else None,
            "verify_tls": getattr(settings, "proxmox_verify_tls", True)}


def build_proxmox_service(settings) -> ProxmoxService | None:
    if not proxmox_integration_status(settings)["configured"]:
        return None
    return ProxmoxService(ProxmoxConfig(settings.proxmox_base_url, settings.proxmox_api_token_id,
                                        settings.proxmox_api_token_secret, settings.proxmox_verify_tls,
                                        settings.proxmox_ca_file, settings.proxmox_server_name,
                                        settings.proxmox_timeout_seconds))
