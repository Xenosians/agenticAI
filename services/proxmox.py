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
from urllib.parse import quote, urlencode, urlsplit

import httpx

from services.lua_adapters.http_clients import client as lua_http_client
from services.provider_errors import ProviderResourceNotFound

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Reachability: a read is tried once on each configured origin (the tailnet
# name first, then any LAN address the owner added for the same host), and a
# dropped connection is tried again on a fresh one. A change is sent once: it
# never moves to another origin unless the connection itself never opened, so
# an unanswered POST is never repeated. The whole read retry stays under the
# tool's patience (3 attempts, connect timeout at most 5 s, short waits).
MAX_ALT_ORIGINS = 4
READ_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 0.5
CONNECT_TIMEOUT_SECONDS = 5.0
# A connection that never carried the request can be tried on the next origin:
# name resolution, TCP connect, TLS handshake or a pooled connection the server
# closed before reading it. A read timeout is not one of these (the server may
# still be working), and a change is never retried after it may have been sent.
CONNECT_FAILURES = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout,
                    httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError)
MAX_ROWS = 200
MAX_GUEST_FILESYSTEMS = 128
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
# Guest changes follow their Proxmox task to the end. A config write is
# synchronous; snapshots, clones, creates and deletes finish as a task.
CONFIG_TASK_DEADLINE_SECONDS = 120.0
SNAPSHOT_TASK_DEADLINE_SECONDS = 300.0
GUEST_TASK_DEADLINE_SECONDS = 600.0
BACKUP_TASK_DEADLINE_SECONDS = 900.0
CHANGE_POLL_SECONDS = (0.5, 1.0, 2.0, 5.0)
MAX_CORES = 128
MAX_MEMORY_MIB = 1_048_576
MAX_DISK_GIB = 65_536
GUEST_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
SNAPSHOT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,62}")
STORAGE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}")
TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:/-]{0,127}")
# One public SSH key on one line (a container's root login); never a private key.
SSH_PUBLIC_KEY = re.compile(r"(?:ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) [A-Za-z0-9+/=]{16,8192}(?: [A-Za-z0-9@._-]{1,64})?")
# A container's first network interface, only with the fields Janus sets: a bridge, an address (or DHCP)
# with its gateway, and the firewall flag.
LXC_NET0 = re.compile(r"name=eth0,bridge=[A-Za-z][A-Za-z0-9_.-]{0,15},ip=(?:dhcp|[0-9.]{7,15}/[0-9]{1,2})"
                      r"(?:,gw=[0-9.]{7,15})?(?:,firewall=[01])?")
MAX_SWAP_MIB = 1_048_576
DISK_ID = re.compile(r"(?:rootfs|mp[0-9]{1,3}|(?:scsi|virtio|sata|ide)[0-9]{1,3})")
TOKEN_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}@[A-Za-z0-9_.-]{1,64}![A-Za-z0-9_.-]{1,64}")
TLS_SERVER_NAME = re.compile(
    r"(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


@dataclass(frozen=True)
class ProxmoxOrigin:
    """One HTTPS origin that answers for the same Proxmox host.

    ``server_name`` is the certificate name TLS verifies, for an origin whose
    host is a bare LAN address (strict verification stays on).
    """
    url: str
    server_name: str | None = None


def _https_origin(value: str) -> str:
    parsed = urlsplit(str(value).strip())
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
    return str(value).strip().rstrip("/")


@dataclass(frozen=True)
class ProxmoxConfig:
    base_url: str
    token_id: str
    token_secret: str = field(repr=False)
    verify_tls: bool = True
    ca_file: Path | None = None
    server_name: str | None = None
    timeout_seconds: float = 10.0
    # Ready-to-use fallbacks for the same host on another network (a LAN
    # address while the tailnet is down). Empty when only the base URL applies.
    alt_origins: tuple[ProxmoxOrigin, ...] = ()
    read_attempts: int = READ_ATTEMPTS
    retry_wait_seconds: float = RETRY_WAIT_SECONDS

    def origin(self) -> str:
        return _https_origin(self.base_url)


def _invalid() -> RuntimeError:
    return RuntimeError("Proxmox returned an invalid answer.")


class ProxmoxRefused(RuntimeError):
    """Proxmox answered a read with an error status other than 404."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Proxmox refused the read (HTTP {status_code}).")


class ProxmoxConflict(RuntimeError):
    """The guest no longer matches what the owner approved; nothing was sent."""


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


def _guest_byte_count(item: dict, field: str) -> int:
    """A byte counter returned by QEMU Guest Agent, bounded to signed 64-bit."""
    value = item.get(field)
    if type(value) is not int or not 0 <= value <= (2 ** 63 - 1):
        raise _invalid()
    return value


def _optional_guest_byte_count(item: dict, field: str) -> int | None:
    """QEMU Guest Agent filesystem counters are optional in its wire schema."""
    if field not in item:
        return None
    return _guest_byte_count(item, field)


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
# One node's maintenance state (node_maintenance): most urgent issue first.
MAINTENANCE_ISSUES = ("service_not_running", "certificate_expired", "certificate_expiring",
                      "enterprise_repository_without_subscription", "no_update_repository",
                      "reboot_for_newer_kernel", "test_repository_enabled")
CERTIFICATE_EXPIRING_DAYS = 30
PVE_VERSION = re.compile(r"pve-manager/([0-9]+(?:\.[0-9]+){1,3})(?:/[0-9a-f]+)?")
KERNEL_RELEASE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+-[0-9]+-pve")
# Installed kernels are proxmox-kernel-<release>-signed (or older pve-kernel-<release>); the
# proxmox-kernel-<series> meta packages and proxmox-kernel-helper are not kernels.
KERNEL_PACKAGE = re.compile(r"(?:proxmox|pve)-kernel-([0-9]+\.[0-9]+\.[0-9]+-[0-9]+-pve)(?:-signed)?")
SERVICE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,63}")
REPOSITORY_HANDLE = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
# Certificate files pveproxy can serve, in the order it prefers them.
SERVED_CERTIFICATES = ("pveproxy-ssl.pem", "pve-ssl.pem")
CERTIFICATE_ROLES = {"pve-root-ca.pem": "cluster_ca", "pve-ssl.pem": "node", "pveproxy-ssl.pem": "custom"}
PROXMOX_CA_NAME = "/CN=Proxmox Virtual Environment/"
KEY_TYPES = {"rsaEncryption": "RSA", "id-ecPublicKey": "EC", "ED25519": "Ed25519", "ED448": "Ed448"}
SUBSCRIPTION_STATES = {"active": "active", "notfound": "none", "new": "not_checked", "expired": "expired",
                       "invalid": "invalid", "suspended": "suspended"}

# One node's load over a window (node_load): Proxmox RRD timeframes and thresholds; most urgent issue first.
LOAD_TIMEFRAMES = ("hour", "day", "week", "month")
LOAD_ISSUES = ("memory_low", "memory_stalls", "io_stalls", "cpu_busy", "swapping")
LOAD_CPU_BUSY = 0.9
LOAD_MEMORY_LOW = 0.1
LOAD_IO_STALL_PERCENT = 10.0
LOAD_SWAPPING = 0.1
MAX_LOAD_SAMPLES = 2000
LOAD_FIELDS = ("maxcpu", "cpu", "iowait", "loadavg", "memtotal", "memused", "memavailable", "swaptotal",
               "swapused", "pressurecpusome", "pressureiosome", "pressureiofull", "pressurememorysome",
               "pressurememoryfull", "netin", "netout", "rootused", "roottotal")

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


def _tag_set(tags: str | None) -> tuple[str, ...]:
    """The tags as a stable set: Proxmox accepts commas but stores them semicolon-separated."""
    return tuple(sorted({part.strip() for part in re.split(r"[,;]", tags or "") if part.strip()}))


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


DISK_SIZE = re.compile(r"(?:^|,)size=([0-9]{1,12})([KMGT]?)(?:,|$)")
SIZE_UNITS = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}


def _disk_size_bytes(value) -> int | None:
    """The size a Proxmox disk property string records, in bytes, or None."""
    if not isinstance(value, str) or len(value) > 2048:
        return None
    match = DISK_SIZE.search(value)
    if match is None:
        return None
    return int(match.group(1)) * SIZE_UNITS[match.group(2)]


class ProxmoxService:
    def __init__(self, config: ProxmoxConfig, client: httpx.Client | None = None):
        self.config = config
        self.origin = config.origin()
        if not TOKEN_ID.fullmatch(config.token_id) or not config.token_secret or len(config.token_secret) > 4096:
            raise ValueError("Proxmox API token configuration is invalid.")
        if config.server_name is not None and not TLS_SERVER_NAME.fullmatch(config.server_name):
            raise ValueError("Proxmox TLS server name is invalid.")
        origins = [ProxmoxOrigin(self.origin, config.server_name)]
        if len(config.alt_origins) > MAX_ALT_ORIGINS:
            raise ValueError("At most four Proxmox fallback origins can be configured.")
        for alt in config.alt_origins:
            if alt.server_name is not None and not TLS_SERVER_NAME.fullmatch(alt.server_name):
                raise ValueError("Proxmox TLS server name is invalid.")
            url = _https_origin(alt.url)
            if all(existing.url != url for existing in origins):
                origins.append(ProxmoxOrigin(url, alt.server_name))
        self.origins = tuple(origins)
        verify = str(config.ca_file) if config.ca_file is not None else config.verify_tls
        self.client = client or lua_http_client(
            verify=verify, timeout=httpx.Timeout(config.timeout_seconds,
                                                 connect=min(config.timeout_seconds, CONNECT_TIMEOUT_SECONDS)))
        self.owns_client = client is None

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _wait(self, attempt: int) -> None:
        wait = self.config.retry_wait_seconds * (2 ** attempt)
        if wait > 0:
            time.sleep(min(wait, 3.0))

    def _read(self, origin: ProxmoxOrigin, path: str, params: dict | None) -> list | dict:
        # The cluster certificate names the host (for example
        # hummingbird.local.pve) while the URL may use its LAN address. Keep
        # strict verification and tell TLS which name to verify against.
        extensions = {"sni_hostname": origin.server_name} if origin.server_name else {}
        request = httpx.Request("GET", origin.url + "/api2/json" + path, params=params,
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

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        """One read, retried across this host's origins while the connection never opened.

        The attempts rotate through the configured origins, so a tailnet name
        that stopped resolving does not stop the read when a LAN origin still
        answers. A read has no effect to repeat, so each attempt is safe; the
        attempt count and the waits stay bounded.
        """
        attempts = max(1, min(int(self.config.read_attempts), READ_ATTEMPTS))
        last: Exception | None = None
        for attempt in range(attempts):
            origin = self.origins[attempt % len(self.origins)]
            try:
                return self._read(origin, path, params)
            except CONNECT_FAILURES as error:
                last = error
                if attempt + 1 < attempts:
                    self._wait(attempt)
        raise last if last is not None else RuntimeError("Proxmox read failed.")

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
        guest = self._inventory_guest(node, vmid)
        guest_type = guest["type"]
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
        is_database, _ = database_tags(_tags(guest))
        if guest_type == "qemu" and is_database:
            status["database_filesystems"] = self._database_filesystem_observation(
                node, vmid, status["status"])
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

    def node_maintenance(self, node: str) -> dict:
        """One node's maintenance state: version, kernels, services, certificates and update source.

        Each section is its own read; one Proxmox refuses (or, on an older
        release, does not have) is listed in `unreadable` and is None, never
        healthy. Certificate subjects,
        issuers, SANs, fingerprints and PEM, the subscription key and server
        id, and repository paths never leave the service. Pending package
        updates need Sys.Modify (apt/update), which Janus does not hold.
        """
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        unreadable: list[str] = []

        def section(name: str, path: str):
            try:
                return self._get(path)
            except (ProxmoxRefused, ProviderResourceNotFound):
                unreadable.append(name)
                return None

        status = self._get(f"/nodes/{node}/status")
        if not isinstance(status, dict):
            raise _invalid()
        version = PVE_VERSION.fullmatch(_text(status, "pveversion", required=True))
        if not version:
            raise _invalid()
        current = status.get("current-kernel")
        running = _text(current, "release") if isinstance(current, dict) else None
        if running is None:
            # Before Proxmox VE 8 only kversion ("Linux <release> #1 SMP ...") names the running kernel.
            words = (_text(status, "kversion", limit=400) or "").split()
            running = words[1] if len(words) > 1 else None
        if running is not None and not KERNEL_RELEASE.fullmatch(running):
            raise _invalid()
        boot = status.get("boot-info")
        boot = {"mode": _text(boot, "mode"), "secure_boot": _boolean(boot, "secureboot")} if isinstance(boot, dict) else None

        kernels = set()
        for row in self._rows(self._get(f"/nodes/{node}/apt/versions"), required=("Package",)):
            match = KERNEL_PACKAGE.fullmatch(row["Package"])
            if match and row.get("CurrentState") == "Installed":
                kernels.add(match.group(1))
        def release_key(release: str):
            return tuple(int(part) for part in re.findall(r"[0-9]+", release))
        newest = max(kernels | {running} - {None}, key=release_key, default=None)

        cluster = section("cluster", "/cluster/status")
        in_cluster = None if cluster is None else any(
            isinstance(row, dict) and row.get("type") == "cluster" for row in self._rows(cluster))

        services = section("services", f"/nodes/{node}/services")
        service_summary = None
        if services is not None:
            rows = self._rows(services, required=("name", "active-state", "unit-state"))
            not_running, expected_stopped, turned_off, missing, running_count = [], [], [], 0, 0
            for row in rows:
                name = row["name"]
                if not SERVICE_NAME.fullmatch(name):
                    raise _invalid()
                active, unit_state = row["active-state"], row["unit-state"]
                if unit_state == "not-found":
                    missing += 1
                elif active == "active":
                    running_count += 1
                elif unit_state == "disabled":
                    turned_off.append(name)
                elif name == "corosync" and in_cluster is False:
                    # corosync starts only when the node has a cluster configuration.
                    expected_stopped.append(name)
                else:
                    not_running.append({"name": name, "state": _text(row, "active-state", required=True)})
            service_summary = {"installed": len(rows) - missing, "running": running_count,
                               "not_running": sorted(not_running, key=lambda row: row["name"]),
                               "expected_stopped": sorted(expected_stopped), "turned_off": sorted(turned_off),
                               "not_installed": missing}

        certificates = section("certificates", f"/nodes/{node}/certificates/info")
        certificate_rows = None
        if certificates is not None:
            now = time.time()
            certificate_rows = []
            names = set()
            for row in self._rows(certificates, required=("filename",)):
                not_after = row.get("notafter")
                if type(not_after) is not int:
                    raise _invalid()
                file = row["filename"]
                names.add(file)
                issuer, subject = row.get("issuer"), row.get("subject")
                key_type = KEY_TYPES.get(row.get("public-key-type"), "other")
                bits = row.get("public-key-bits")
                certificate_rows.append({
                    "file": file, "role": CERTIFICATE_ROLES.get(file, "other"), "served": False,
                    "issuer": "self_signed" if issuer == subject else
                              "proxmox_ca" if isinstance(issuer, str) and issuer.startswith(PROXMOX_CA_NAME) else
                              "other_ca",
                    "key": f"{key_type} {bits}" if type(bits) is int else key_type,
                    "days_left": math.floor((not_after - now) / DAY_SECONDS)})
            served = next((name for name in SERVED_CERTIFICATES if name in names), None)
            for row in certificate_rows:
                row["served"] = row["file"] == served
            certificate_rows.sort(key=lambda row: (not row["served"], row["days_left"]))

        subscription = section("subscription", f"/nodes/{node}/subscription")
        if subscription is not None:
            if not isinstance(subscription, dict):
                raise _invalid()
            subscription = SUBSCRIPTION_STATES.get(_text(subscription, "status") or "", "unknown")

        repositories = section("repositories", f"/nodes/{node}/apt/repositories")
        repository_summary = None
        if repositories is not None:
            if not isinstance(repositories, dict) or not isinstance(repositories.get("standard-repos"), list):
                raise _invalid()
            enabled, turned_off_repos = [], []
            for row in self._rows(repositories["standard-repos"], required=("handle",)):
                handle, state = row["handle"], row.get("status")
                if not REPOSITORY_HANDLE.fullmatch(handle) or state not in (None, 0, 1) or type(state) is bool:
                    raise _invalid()
                if state == 1:
                    enabled.append(handle)
                elif state == 0:
                    turned_off_repos.append(handle)
            errors = repositories.get("errors")
            repository_summary = {"enabled": enabled, "turned_off": turned_off_repos,
                                  "errors": len(errors) if isinstance(errors, list) else 0}

        found = {
            "service_not_running": bool(service_summary and service_summary["not_running"]),
            "certificate_expired": any(row["served"] and row["days_left"] < 0 for row in certificate_rows or []),
            "certificate_expiring": any(
                0 <= row["days_left"] < CERTIFICATE_EXPIRING_DAYS for row in certificate_rows or []),
            "enterprise_repository_without_subscription": bool(
                repository_summary and "enterprise" in repository_summary["enabled"]
                and subscription is not None and subscription != "active"),
            "no_update_repository": bool(repository_summary and not set(repository_summary["enabled"])
                                         & {"enterprise", "no-subscription", "test"}),
            "reboot_for_newer_kernel": bool(running and newest and newest != running),
            "test_repository_enabled": bool(repository_summary and "test" in repository_summary["enabled"]),
        }
        return self._result({
            "schema": "proxmox-node-maintenance.v1", "node": node, "version": version.group(1),
            "kernel": {"running": running, "newest_installed": newest, "installed": len(kernels)},
            "boot": boot, "in_cluster": in_cluster, "services": service_summary,
            "certificates": certificate_rows, "subscription": subscription, "repositories": repository_summary,
            "issues": [issue for issue in MAINTENANCE_ISSUES if found[issue]],
            "unreadable": unreadable,
        })

    def _load_series(self, node: str, timeframe: str, cf: str) -> list[dict]:
        rows = self._rows(self._get(f"/nodes/{node}/rrddata", {"timeframe": timeframe, "cf": cf}))
        if len(rows) > MAX_LOAD_SAMPLES:
            raise _invalid()
        for row in rows:
            if type(row.get("time")) is not int:
                raise _invalid()
            for field in LOAD_FIELDS:
                _number(row, field)
        return rows

    def node_load(self, node: str, timeframe: str) -> dict:
        """One node's typical and peak load over a window, from Proxmox's own history (RRD).

        The AVERAGE series gives typical values and the MAX series peaks.
        Minutes without a sample are gaps, never zero; a window without any
        sample reports no load at all.
        """
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        if timeframe not in LOAD_TIMEFRAMES:
            raise ValueError("Choose a window of hour, day, week or month.")
        average = self._load_series(node, timeframe, "AVERAGE")
        peak = self._load_series(node, timeframe, "MAX")

        def values(rows: list[dict], field: str) -> list:
            return [row[field] for row in rows if row.get(field) is not None]

        def mean(rows: list[dict], field: str):
            found = values(rows, field)
            return sum(found) / len(found) if found else None

        def highest(rows: list[dict], field: str):
            found = values(rows, field)
            return max(found) if found else None

        def percent(value):
            return None if value is None else round(value * 100, 1)

        def rounded(value, digits=2):
            # Proxmox sends RRD values as long fractions (0.516333333333333).
            return None if value is None else round(value, digits)

        sampled = [row for row in average if row.get("cpu") is not None or row.get("memused") is not None]
        times = [row["time"] for row in average]
        step = min((b - a for a, b in zip(times, times[1:]) if b > a), default=None)
        data = {"schema": "proxmox-node-load.v1", "node": node, "timeframe": timeframe, "step_seconds": step,
                "samples": len(sampled), "expected_samples": len(average),
                "from": datetime.fromtimestamp(times[0], timezone.utc).isoformat() if times else None,
                "to": datetime.fromtimestamp(times[-1], timezone.utc).isoformat() if times else None,
                "cpu": None, "io_wait": None, "load": None, "memory": None, "swap": None, "pressure": None,
                "network": None, "root_disk": None, "issues": []}
        if not sampled:
            return self._result(data)
        last = sampled[-1]
        memory_total = highest(average, "memtotal")
        swap_total = highest(average, "swaptotal")
        lowest_available = min(values(average, "memavailable") + values(peak, "memavailable"), default=None)
        data.update({
            "cpu": {"cores": highest(average, "maxcpu"), "average_percent": percent(mean(average, "cpu")),
                    "peak_percent": percent(highest(peak, "cpu"))},
            "io_wait": {"average_percent": percent(mean(average, "iowait")),
                        "peak_percent": percent(highest(peak, "iowait"))},
            "load": {"average": rounded(mean(average, "loadavg")), "peak": rounded(highest(peak, "loadavg"))},
            "memory": {"total_bytes": memory_total, "average_used_bytes": mean(average, "memused"),
                       "peak_used_bytes": highest(peak, "memused"), "lowest_available_bytes": lowest_available},
            "swap": {"total_bytes": swap_total, "peak_used_bytes": highest(peak, "swapused")},
            "pressure": {"cpu_some": rounded(highest(peak, "pressurecpusome")),
                         "io_some": rounded(highest(peak, "pressureiosome")),
                         "io_full": rounded(highest(peak, "pressureiofull")),
                         "memory_some": rounded(highest(peak, "pressurememorysome")),
                         "memory_full": rounded(highest(peak, "pressurememoryfull"))},
            "network": {"average_in_bytes_per_second": mean(average, "netin"),
                        "peak_in_bytes_per_second": highest(peak, "netin"),
                        "average_out_bytes_per_second": mean(average, "netout"),
                        "peak_out_bytes_per_second": highest(peak, "netout")},
            "root_disk": {"used_bytes": last.get("rootused"), "total_bytes": last.get("roottotal")},
        })
        pressure, swap_peak, cpu_peak = data["pressure"], data["swap"]["peak_used_bytes"], highest(peak, "cpu")
        found = {
            "memory_low": bool(memory_total and lowest_available is not None
                               and lowest_available < LOAD_MEMORY_LOW * memory_total),
            "memory_stalls": bool(pressure["memory_full"]),
            "io_stalls": (pressure["io_full"] or 0) >= LOAD_IO_STALL_PERCENT,
            "cpu_busy": (cpu_peak or 0) >= LOAD_CPU_BUSY,
            "swapping": bool(swap_total and swap_peak and swap_peak >= LOAD_SWAPPING * swap_total),
        }
        data["issues"] = [issue for issue in LOAD_ISSUES if found[issue]]
        return self._result(data)

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

    def _database_filesystem_observation(self, node: str, vmid: int, guest_status: str) -> dict:
        """Bounded filesystem-only observation for an already identified DB QEMU guest."""
        base = {"schema": "proxmox-database-filesystems.v1", "scope": "guest_filesystem_capacity_only"}
        if guest_status != "running":
            return {**base, "state": "guest_not_running", "count": 0, "filesystems": [],
                    "note": "Filesystem usage requires a running VM and an available guest agent."}
        try:
            rows = self._rows(self._get(f"/nodes/{node}/qemu/{vmid}/agent/get-fsinfo"))
        except ProxmoxRefused as error:
            note = f"Proxmox refused the guest filesystem read (HTTP {error.status_code})."
        except ProviderResourceNotFound:
            note = "Proxmox did not find the guest filesystem read endpoint or guest."
        except httpx.TransportError:
            note = "Proxmox did not answer the guest filesystem read."
        else:
            note = None
        if note is not None:
            return {**base, "state": "unavailable", "count": 0, "filesystems": [], "note": note}
        if len(rows) > MAX_GUEST_FILESYSTEMS:
            raise _invalid()
        filesystems = []
        for row in rows:
            mountpoint = _text(row, "mountpoint", required=True, limit=512)
            filesystem = _text(row, "type", required=True, limit=64)
            used = _optional_guest_byte_count(row, "used-bytes")
            total = _optional_guest_byte_count(row, "total-bytes")
            if used is not None and total is not None and used > total:
                raise _invalid()
            filesystems.append({
                "mountpoint": mountpoint, "filesystem": filesystem,
                "used_bytes": used, "total_bytes": total,
                "free_bytes": total - used if used is not None and total is not None else None,
                "usage_percent": round(used / total * 100, 1)
                if used is not None and total not in (None, 0) else None,
            })
        filesystems.sort(key=lambda row: (row["mountpoint"].casefold(), row["filesystem"].casefold()))
        partial = any(row["used_bytes"] is None or row["total_bytes"] is None for row in filesystems)
        return {**base, "state": "partial" if partial else "observed" if filesystems else "no_filesystems_reported",
                "count": len(filesystems), "truncated": False, "filesystems": filesystems,
                "note": "QEMU Guest Agent did not report all optional filesystem size counters."
                if partial else None,
                "near_full_count": sum(row["usage_percent"] is not None and
                                       row["usage_percent"] >= DISK_NEARLY_FULL * 100
                                       for row in filesystems)}

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

    def guest_change_target(self, node: str, vmid: int) -> dict:
        """Read-only: the exact guest a change would affect and its config digest.

        The digest lets the approval freeze the configuration the owner saw, so
        a concurrent edit before the change is refused instead of overwritten.
        """
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
            raise ValueError("Enter one valid Proxmox VMID.")
        guest = self._guest_row(node, vmid)
        config = self._get(f"/nodes/{node}/{guest['type']}/{vmid}/config")
        if not isinstance(config, dict):
            raise _invalid()
        self._validate_guest_identity(config, node, vmid)
        digest = _text(config, "digest", limit=128)
        return self._result({**guest, "digest": digest})

    def _send(self, method: str, path: str, form: dict[str, str] | None = None) -> httpx.Response:
        headers = {"Authorization": f"PVEAPIToken={self.config.token_id}={self.config.token_secret}",
                   "Accept": "application/json"}
        body = ""
        if form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            body = "&".join(f"{key}={quote(value, safe='')}" for key, value in form.items())
        # A change moves to another origin only while the connection never
        # opened (nothing could have reached Proxmox yet). Once a request may
        # have been sent it is never repeated here: the caller reports an
        # unknown outcome and reconciles by reading the guest.
        unsafe = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.ReadError, httpx.WriteError,
                  httpx.RemoteProtocolError)
        last: Exception | None = None
        for origin in self.origins:
            extensions = {"sni_hostname": origin.server_name} if origin.server_name else {}
            request = httpx.Request(method, origin.url + "/api2/json" + path, headers=headers,
                                    content=body.encode(), extensions=extensions)
            try:
                return self.client.send(request)
            except CONNECT_FAILURES as error:
                if isinstance(error, unsafe):
                    raise
                last = error
        raise last if last is not None else RuntimeError("Proxmox change could not be sent.")

    def _post(self, path: str, form: dict[str, str]) -> httpx.Response:
        return self._send("POST", path, form)

    def _put(self, path: str, form: dict[str, str]) -> httpx.Response:
        return self._send("PUT", path, form)

    def _delete(self, path: str, params: dict[str, str] | None = None) -> httpx.Response:
        query = f"?{urlencode(params)}" if params else ""
        return self._send("DELETE", path + query)

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

    # ------------------------------------------------------------------
    # Governed guest changes (owner decisions, 17 November)
    #
    # Every change runs on the exact guest frozen at approval, refuses a
    # concurrent edit (config digest) or a different guest identity, sends
    # one request, follows its task to the end and reads the expected state
    # back. An answer that cannot be confirmed is reported as an unknown
    # outcome and is never retried here.
    # ------------------------------------------------------------------

    @staticmethod
    def _change_receipt(operation: str, node: str, vmid: int) -> dict:
        return {"provider": "proxmox", "operation": operation, "node": node, "vmid": vmid,
                "type": None, "name": None, "upid": None, "task_exit_status": None,
                "changed": False, "verification_ok": False, "retry_safe": True,
                "observed_at": None, "error": None}

    @staticmethod
    def _change_outcome(receipt: dict, ok: bool, status: str, error: str | None = None, **fields) -> dict:
        return {**receipt, **fields, "ok": ok, "status": status, "error": error,
                "observed_at": datetime.now(timezone.utc).isoformat()}

    def _follow_task(self, node: str, upid: str, *, deadline_seconds: float,
                     sleep=time.sleep, clock=time.monotonic) -> dict | None:
        """Wait for one Proxmox task; None means it had not finished by the deadline."""
        deadline = clock() + deadline_seconds
        polls = 0
        while True:
            try:
                task = self._get(f"/nodes/{node}/tasks/{quote(upid, safe='')}/status")
            except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
                task = None
            if isinstance(task, dict) and task.get("status") == "stopped":
                return task
            if clock() >= deadline:
                return None
            sleep(CHANGE_POLL_SECONDS[min(polls, len(CHANGE_POLL_SECONDS) - 1)])
            polls += 1

    def _task_response(self, response: httpx.Response, node: str, receipt: dict) -> tuple[str | None, dict | None]:
        """The UPID of an accepted task, or the receipt of a request that did not start one."""
        if response.status_code in (401, 403):
            return None, self._change_outcome(receipt, False, "denied",
                                             self._post_refusal(response) + " Nothing was changed.")
        if response.status_code != 200:
            return None, self._change_outcome(receipt, False, "failed",
                                             self._post_refusal(response) + " Nothing was changed.")
        try:
            data = response.json().get("data")
        except (ValueError, UnicodeError, AttributeError):
            data = None
        match = UPID.fullmatch(data) if isinstance(data, str) and len(data) <= 512 else None
        if match is None or match.group(1) != node:
            return None, self._change_outcome(receipt, False, "outcome_unknown",
                                             "Proxmox accepted the change but did not return its task; "
                                             "check the guest before trying again.", retry_safe=False)
        receipt.update(upid=match.group(0), changed=True, retry_safe=False)
        return match.group(0), None

    def _match_guest(self, node: str, vmid: int, expected_type: str, expected_name: str | None) -> dict:
        guest = self._guest_row(node, vmid)
        if guest["type"] != expected_type or guest["name"] != (expected_name or None):
            raise ProxmoxConflict(
                f"The guest no longer matches the approved target "
                f"({expected_type} {expected_name or 'unnamed'}); nothing was changed.")
        return guest

    def _config_for_change(self, node: str, guest: dict, expected_digest: str | None) -> tuple[dict, str | None]:
        config = self._get(f"/nodes/{node}/{guest['type']}/{guest['vmid']}/config")
        if not isinstance(config, dict):
            raise _invalid()
        self._validate_guest_identity(config, node, guest["vmid"])
        digest = _text(config, "digest", limit=128)
        if expected_digest is not None and digest != expected_digest:
            raise ProxmoxConflict("The guest's configuration changed after approval; nothing was changed.")
        return config, digest

    def _guest_exists(self, node: str, vmid: int) -> dict | None:
        """The guest using this VMID anywhere in the cluster, or None.

        Proxmox VMIDs are unique across the cluster, so a VMID used on another
        node blocks a new guest just as one on the chosen node does.
        """
        rows = self._rows(self._get("/cluster/resources", {"type": "vm"}))
        for row in rows:
            if row.get("vmid") == vmid and row.get("type") in {"qemu", "lxc"}:
                return row
        return None

    def _confirm_new_guest(self, receipt: dict, node: str, vmid: int, guest_type: str | None,
                           name: str) -> dict | None:
        """After a finished create, clone or restore task: prove the new guest is the one asked for.

        The type comes from the cluster inventory, the name from the guest's own configuration.
        /cluster/resources lists a brand-new guest without its name for a few seconds (seen live
        on 20 Nov 2026), so the inventory row alone cannot judge a new guest's identity. Returns
        the failed or unknown receipt, or None when the guest is exactly the one requested.
        """
        try:
            created = self._guest_exists(node, vmid)
            config = None
            if created is not None and created.get("type") in ("lxc", "qemu"):
                config = self._get(f"/nodes/{node}/{created['type']}/{vmid}/config")
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the guest list could not be read back. "
                                        "Check the node before trying again.")
        if created is None:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but guest {vmid} is not listed.")
        if guest_type is not None and created.get("type") != guest_type:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but guest {vmid} is not a {guest_type} guest.")
        if not isinstance(config, dict):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the new guest's configuration could not be "
                                        "read back. Check the node before trying again.")
        observed = config.get("hostname" if created["type"] == "lxc" else "name")
        if observed != name:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but guest {vmid} is not named '{name}'.")
        return None

    def new_guest_target(self, node: str, newid: int) -> dict:
        """Read-only: the exact node a new guest would land on, and that its VMID is free."""
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        if type(newid) is not int or not (1 <= newid <= MAX_VMID):
            raise ValueError("Enter one valid, free Proxmox VMID for the new guest.")
        existing = self._guest_exists(node, newid)
        if existing is not None:
            raise ProxmoxConflict(
                f"VMID {newid} is already used on node {existing.get('node')}; choose a free VMID.")
        rows = self._rows(self._get("/nodes"), required=("node", "status"))
        for row in rows:
            if row.get("node") == node:
                return self._result({"node": node, "vmid": newid, "free": True})
        raise ProviderResourceNotFound(f"Proxmox node {node} was not found.")

    def next_vmid(self) -> int:
        """Read-only: the next free VMID Proxmox itself offers for the whole cluster.

        Proxmox owns the numbering (GET /cluster/nextid); Janus never invents a
        range or assumes a lab convention, so the same code works on any cluster.
        """
        response = self._send("GET", "/cluster/nextid")
        if response.status_code == 404:
            raise ProviderResourceNotFound("Proxmox did not find the requested resource.")
        if response.status_code != 200:
            raise ProxmoxRefused(response.status_code)
        try:
            data = response.json().get("data")
        except (ValueError, UnicodeError, AttributeError):
            raise _invalid() from None
        if type(data) is int:
            vmid = data
        elif isinstance(data, str) and data.isdigit():
            vmid = int(data)
        else:
            raise _invalid()
        if not (1 <= vmid <= MAX_VMID):
            raise _invalid()
        return vmid

    def guest_set_resources(self, node: str, vmid: int, *, cores: int | None, memory_mib: int | None,
                            expected_type: str, expected_name: str | None, expected_digest: str | None,
                            sleep=time.sleep, clock=time.monotonic) -> dict:
        """Set one guest's cores and/or memory on the exact approved guest."""
        if (cores is not None and (type(cores) is not int or not (1 <= cores <= MAX_CORES))) or \
                (memory_mib is not None and (type(memory_mib) is not int or not (16 <= memory_mib <= MAX_MEMORY_MIB))) or \
                (cores is None and memory_mib is None):
            raise ValueError("Enter a core count (1-128) or memory in MiB (16-1048576).")
        receipt = self._change_receipt("guest_set_resources", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        try:
            config, _ = self._config_for_change(node, guest, expected_digest)
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not read the guest's configuration; nothing was sent.")
        previous = {"cores": _number(config, "cores"), "memory_mib": _memory_mib(config, guest["type"])}
        form = {}
        if cores is not None:
            form["cores"] = str(cores)
        if memory_mib is not None:
            form["memory"] = str(memory_mib)
        try:
            response = self._put(f"/nodes/{node}/{guest['type']}/{vmid}/config", form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; the change was not sent.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the change request; it may have applied. "
                                        "Check the guest before trying again.", retry_safe=False)
        if response.status_code in (401, 403):
            return self._change_outcome(receipt, False, "denied",
                                        self._post_refusal(response) + " Nothing was changed.")
        if response.status_code != 200:
            return self._change_outcome(receipt, False, "failed",
                                        self._post_refusal(response) + " Nothing was changed.")
        receipt.update(changed=True, retry_safe=False)
        try:
            current = self._get(f"/nodes/{node}/{guest['type']}/{vmid}/config")
            observed = {"cores": _number(current, "cores") if isinstance(current, dict) else None,
                        "memory_mib": _memory_mib(current, guest["type"]) if isinstance(current, dict) else None}
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox accepted the change, but the guest could not be read back. "
                                        "Check the guest before trying again.")
        receipt.update(previous=previous, observed=observed,
                       cores=cores if cores is not None else observed["cores"],
                       memory_mib=memory_mib if memory_mib is not None else observed["memory_mib"])
        expected = {"cores": cores, "memory_mib": memory_mib}
        if any(value is not None and observed[key] != value for key, value in expected.items()):
            return self._change_outcome(receipt, False, "failed",
                                        "Proxmox accepted the change but the guest does not show it "
                                        f"(cores {observed['cores']}, memory {observed['memory_mib']} MiB).")
        return self._change_outcome(receipt, True, "executed", verification_ok=True)

    def guest_set_options(self, node: str, vmid: int, *, tags: str | None, on_boot: bool | None,
                          expected_type: str, expected_name: str | None, expected_digest: str | None,
                          console_mode: str | None = None, sleep=time.sleep, clock=time.monotonic) -> dict:
        """Set one guest's tags, start-at-boot flag and/or (containers) console mode on the exact approved guest."""
        normalized = None
        if tags is not None:
            parts = [part.strip() for part in re.split(r"[,;]", tags) if part.strip()]
            if len(parts) > 32 or any(not TAG.fullmatch(part) for part in parts):
                raise ValueError("Enter tags as a comma-separated list of letters, digits, '_', '.', ':', '/' or '-'.")
            normalized = ",".join(dict.fromkeys(parts))
        if console_mode is not None and (console_mode not in ("tty", "console", "shell") or expected_type != "lxc"):
            raise ValueError("A container's console mode is tty, console or shell.")
        if normalized is None and on_boot is None and console_mode is None:
            raise ValueError("Enter tags or a start-at-boot choice.")
        receipt = self._change_receipt("guest_set_options", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        try:
            config, _ = self._config_for_change(node, guest, expected_digest)
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not read the guest's configuration; nothing was sent.")
        previous = {"tags": _text(config, "tags", limit=2048), "on_boot": _boolean(config, "onboot")}
        form = {}
        if normalized is not None:
            form["tags"] = normalized
        if on_boot is not None:
            form["onboot"] = "1" if on_boot else "0"
        if console_mode is not None:
            form["cmode"] = console_mode
        try:
            response = self._put(f"/nodes/{node}/{guest['type']}/{vmid}/config", form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; the change was not sent.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the change request; it may have applied. "
                                        "Check the guest before trying again.", retry_safe=False)
        if response.status_code in (401, 403):
            return self._change_outcome(receipt, False, "denied",
                                        self._post_refusal(response) + " Nothing was changed.")
        if response.status_code != 200:
            return self._change_outcome(receipt, False, "failed",
                                        self._post_refusal(response) + " Nothing was changed.")
        receipt.update(changed=True, retry_safe=False)
        try:
            current = self._get(f"/nodes/{node}/{guest['type']}/{vmid}/config")
            observed = {"tags": _text(current, "tags", limit=2048) if isinstance(current, dict) else None,
                        "on_boot": _boolean(current, "onboot") if isinstance(current, dict) else None,
                        # Proxmox omits the console mode while it is the default, tty.
                        "console_mode": (_text(current, "cmode", limit=16) or "tty") if isinstance(current, dict)
                        else None}
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox accepted the change, but the guest could not be read back. "
                                        "Check the guest before trying again.")
        receipt.update(previous=previous, observed=observed, tags=normalized if normalized is not None else observed["tags"],
                       on_boot=on_boot if on_boot is not None else observed["on_boot"])
        if (normalized is not None and _tag_set(observed["tags"]) != _tag_set(normalized)) or \
                (on_boot is not None and observed["on_boot"] != on_boot) or \
                (console_mode is not None and observed["console_mode"] != console_mode):
            return self._change_outcome(receipt, False, "failed",
                                        "Proxmox accepted the change but the guest does not show it.")
        return self._change_outcome(receipt, True, "executed", verification_ok=True)

    def guest_resize_disk(self, node: str, vmid: int, *, disk: str, grow_gib: int,
                          expected_type: str, expected_name: str | None, expected_digest: str | None,
                          sleep=time.sleep, clock=time.monotonic) -> dict:
        """Grow one disk of the exact approved guest and prove the new size (grow only)."""
        if not isinstance(disk, str) or not DISK_ID.fullmatch(disk):
            raise ValueError("Enter one disk of the guest exactly as its configuration lists it (for example scsi0).")
        if type(grow_gib) is not int or not (1 <= grow_gib <= MAX_DISK_GIB):
            raise ValueError(f"Enter how many GiB to add (1-{MAX_DISK_GIB}).")
        receipt = self._change_receipt("guest_resize_disk", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        try:
            config, _ = self._config_for_change(node, guest, expected_digest)
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not read the guest's configuration; nothing was sent.")
        previous = _disk_size_bytes(config.get(disk))
        if previous is None:
            return self._change_outcome(receipt, False, "conflict",
                                        f"Disk {disk} is not part of this guest or has no size to grow; "
                                        "nothing was changed.")
        try:
            response = self._put(f"/nodes/{node}/{guest['type']}/{vmid}/resize",
                                 {"disk": disk, "size": f"+{grow_gib}G"})
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; the disk was not resized.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the resize request; it may have applied. "
                                        "Check the guest before trying again.", retry_safe=False)
        if response.status_code in (401, 403):
            return self._change_outcome(receipt, False, "denied",
                                        self._post_refusal(response) + " Nothing was changed.")
        if response.status_code != 200:
            return self._change_outcome(receipt, False, "failed",
                                        self._post_refusal(response) + " Nothing was changed.")
        receipt.update(changed=True, retry_safe=False)
        try:
            current = self._get(f"/nodes/{node}/{guest['type']}/{vmid}/config")
            observed = _disk_size_bytes(current.get(disk)) if isinstance(current, dict) else None
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox accepted the resize, but disk {disk} could not be read back. "
                                        "Check the guest before trying again.")
        expected = previous + grow_gib * 1024 ** 3
        receipt.update(disk=disk, grow_gib=grow_gib, previous_bytes=previous, new_bytes=observed)
        if observed != expected:
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox accepted the resize, but disk {disk} does not show "
                                        f"{expected} bytes.")
        return self._change_outcome(receipt, True, "executed", verification_ok=True)

    def guest_snapshot_create(self, node: str, vmid: int, *, snapshot: str, description: str | None,
                              expected_type: str, expected_name: str | None,
                              sleep=time.sleep, clock=time.monotonic) -> dict:
        """Take one named snapshot of the exact approved guest and prove it exists."""
        if not isinstance(snapshot, str) or not SNAPSHOT_NAME.fullmatch(snapshot) or snapshot == "current":
            raise ValueError("Enter a snapshot name starting with a letter (letters, digits, '_' or '-').")
        if description is not None and (not isinstance(description, str) or len(description) > 256
                                        or any(ord(char) < 32 for char in description)):
            raise ValueError("Keep the snapshot description to one line of at most 256 characters.")
        receipt = self._change_receipt("guest_snapshot_create", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        form = {"snapname": snapshot}
        if description:
            form["description"] = description
        try:
            response = self._post(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot", form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; no snapshot was requested.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the snapshot request; it may have run. "
                                        "Check the guest before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=SNAPSHOT_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the guest "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        try:
            rows = self._rows(self._get(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot"))
            names = {_text(row, "name", required=True) for row in rows}
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the snapshots could not be read back.")
        if snapshot not in names:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but snapshot '{snapshot}' is not listed.")
        return self._change_outcome(receipt, True, "executed", snapshot=snapshot, verification_ok=True)

    def guest_snapshot_delete(self, node: str, vmid: int, *, snapshot: str, expected_type: str,
                              expected_name: str | None, sleep=time.sleep, clock=time.monotonic) -> dict:
        """Delete one named snapshot of the exact approved guest and prove it is gone."""
        if not isinstance(snapshot, str) or not SNAPSHOT_NAME.fullmatch(snapshot) or snapshot == "current":
            raise ValueError("Enter one existing snapshot name.")
        receipt = self._change_receipt("guest_snapshot_delete", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        try:
            response = self._delete(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot/"
                                    f"{quote(snapshot, safe='')}")
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; no snapshot was deleted.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the deletion; it may have run. "
                                        "Check the guest before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=SNAPSHOT_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the guest "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        try:
            rows = self._rows(self._get(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot"))
            names = {_text(row, "name", required=True) for row in rows}
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the snapshots could not be read back.")
        if snapshot in names:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but snapshot '{snapshot}' is still listed.")
        return self._change_outcome(receipt, True, "executed", snapshot=snapshot, verification_ok=True)

    def guest_snapshot_rollback(self, node: str, vmid: int, *, snapshot: str, expected_type: str,
                                expected_name: str | None, sleep=time.sleep, clock=time.monotonic) -> dict:
        """Roll the exact approved guest back to one snapshot and prove the current state follows it."""
        if not isinstance(snapshot, str) or not SNAPSHOT_NAME.fullmatch(snapshot) or snapshot == "current":
            raise ValueError("Enter one existing snapshot name.")
        receipt = self._change_receipt("guest_snapshot_rollback", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        try:
            response = self._post(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot/"
                                  f"{quote(snapshot, safe='')}/rollback", {})
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; nothing was rolled back.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the rollback; it may have run. "
                                        "Check the guest before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=SNAPSHOT_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the guest "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        try:
            rows = self._rows(self._get(f"/nodes/{node}/{guest['type']}/{vmid}/snapshot"))
            current = next((row for row in rows if row.get("name") == "current"), None)
            parent = current.get("parent") if isinstance(current, dict) else None
        except (httpx.HTTPError, RuntimeError, ProviderResourceNotFound):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the snapshots could not be read back. "
                                        "Check the guest before trying again.")
        receipt["current_parent"] = parent
        if parent != snapshot:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but the guest does not follow snapshot "
                                        f"'{snapshot}'.")
        return self._change_outcome(receipt, True, "executed", snapshot=snapshot, verification_ok=True)

    def guest_delete(self, node: str, vmid: int, *, expected_type: str, expected_name: str | None,
                     sleep=time.sleep, clock=time.monotonic) -> dict:
        """Delete the exact approved, stopped guest and prove it is gone."""
        receipt = self._change_receipt("guest_delete", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["status"] == "running":
            return self._change_outcome(receipt, False, "conflict",
                                        "The guest is running; shut it down before deleting it.")
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        try:
            response = self._delete(f"/nodes/{node}/{guest['type']}/{vmid}", {"purge": "1"})
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; nothing was deleted.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the deletion; it may have run. "
                                        "Check the guest before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=GUEST_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the guest "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        try:
            remaining = self._guest_exists(node, vmid)
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the guest list could not be read back.")
        if remaining is not None:
            return self._change_outcome(receipt, False, "failed",
                                        f"The Proxmox task finished, but guest {vmid} is still listed.")
        return self._change_outcome(receipt, True, "executed", deleted=True, verification_ok=True)

    def guest_create_lxc(self, node: str, vmid: int, *, hostname: str, template: str, cores: int,
                         memory_mib: int, disk_gib: int, storage: str, pool: str | None,
                         description: str | None, swap_mib: int | None = None, net0: str | None = None,
                         nameserver: str | None = None, ssh_public_keys: str | None = None,
                         nesting: bool = False, console_mode: str | None = None, on_boot: bool = False,
                         tags: str | None = None, sleep=time.sleep, clock=time.monotonic) -> dict:
        """Create one stopped LXC container from a template on the exact node, then prove it exists.

        The optional settings are sent only when given, each in its own fixed shape: swap, the first network
        interface, nameservers, one public SSH key for root, nesting (systemd in an unprivileged container),
        the console mode, start at boot and tags. A password is never part of the request.
        """
        if type(vmid) is not int or not (1 <= vmid <= MAX_VMID):
            raise ValueError("Enter one valid Proxmox VMID.")
        if not isinstance(hostname, str) or not GUEST_NAME.fullmatch(hostname):
            raise ValueError("Enter a hostname of letters, digits and hyphens, starting and ending with one.")
        if not isinstance(template, str) or len(template) > 300 or \
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}:vztmpl/[A-Za-z0-9][A-Za-z0-9_.+-]{0,199}", template):
            raise ValueError("Choose a template volume from the node's storage content.")
        if type(cores) is not int or not (1 <= cores <= MAX_CORES):
            raise ValueError("Enter a core count (1-128).")
        if type(memory_mib) is not int or not (16 <= memory_mib <= MAX_MEMORY_MIB):
            raise ValueError("Enter memory in MiB (16-1048576).")
        if type(disk_gib) is not int or not (1 <= disk_gib <= MAX_DISK_GIB):
            raise ValueError("Enter a disk size in GiB (1-65536).")
        if not isinstance(storage, str) or not STORAGE_ID.fullmatch(storage):
            raise ValueError("Enter one exact storage name.")
        if pool is not None and (not isinstance(pool, str) or not STORAGE_ID.fullmatch(pool)):
            raise ValueError("Enter one exact pool name.")
        if description is not None and (not isinstance(description, str) or len(description) > 256
                                        or any(ord(char) < 32 for char in description)):
            raise ValueError("Keep the description to one line of at most 256 characters.")
        optional: dict[str, str] = {}
        if swap_mib is not None:
            if type(swap_mib) is not int or not (0 <= swap_mib <= MAX_SWAP_MIB):
                raise ValueError("Enter swap in MiB (0-1048576).")
            optional["swap"] = str(swap_mib)
        if net0 is not None:
            if not isinstance(net0, str) or not LXC_NET0.fullmatch(net0):
                raise ValueError("Enter the network as name=eth0,bridge=<bridge>,ip=<address/prefix or dhcp>"
                                 "[,gw=<gateway>][,firewall=0|1].")
            for field_name in ("ip", "gw"):
                value = dict(part.split("=", 1) for part in net0.split(",")).get(field_name)
                if value not in (None, "dhcp"):
                    try:
                        (ipaddress.ip_interface if field_name == "ip" else ipaddress.ip_address)(value)
                    except ValueError:
                        raise ValueError(f"The network's {field_name} is not an IPv4 address.") from None
            optional["net0"] = net0
        if nameserver is not None:
            servers = nameserver.split(" ") if isinstance(nameserver, str) else []
            try:
                valid = 1 <= len(servers) <= 3 and all(ipaddress.ip_address(server) for server in servers)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("Enter one to three nameserver addresses separated by spaces.")
            optional["nameserver"] = nameserver
        if ssh_public_keys is not None:
            if not isinstance(ssh_public_keys, str) or not SSH_PUBLIC_KEY.fullmatch(ssh_public_keys):
                raise ValueError("Enter one public SSH key on one line.")
            optional["ssh-public-keys"] = ssh_public_keys
        if type(nesting) is not bool or type(on_boot) is not bool:
            raise ValueError("Nesting and start at boot are on or off.")
        if nesting:
            optional["features"] = "nesting=1"
        if console_mode is not None:
            if console_mode not in ("tty", "shell"):
                raise ValueError("The console mode is tty or shell.")
            optional["cmode"] = console_mode
        if on_boot:
            optional["onboot"] = "1"
        if tags is not None:
            parts = [part.strip() for part in tags.split(",") if part.strip()] if isinstance(tags, str) else []
            if not parts or len(parts) > 32 or any(not TAG.fullmatch(part) for part in parts):
                raise ValueError("Enter tags as a comma-separated list of letters, digits, '_', '.', ':', '/' or '-'.")
            optional["tags"] = ",".join(dict.fromkeys(parts))
        receipt = self._change_receipt("guest_create_lxc", node, vmid)
        receipt.update(name=hostname)
        try:
            existing = self._guest_exists(node, vmid)
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest list; nothing was created.")
        if existing is not None:
            return self._change_outcome(receipt, False, "conflict",
                                        f"VMID {vmid} is already used on node {existing.get('node')}; "
                                        "nothing was created.")
        form = {"vmid": str(vmid), "ostemplate": template, "hostname": hostname, "cores": str(cores),
                "memory": str(memory_mib), "rootfs": f"{storage}:{disk_gib}", "unprivileged": "1"}
        if pool:
            form["pool"] = pool
        if description:
            form["description"] = description
        form.update(optional)
        try:
            response = self._post(f"/nodes/{node}/lxc", form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; nothing was created.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the creation request; it may have run. "
                                        "Check the node before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=GUEST_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the node "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        mismatch = self._confirm_new_guest(receipt, node, vmid, "lxc", hostname)
        if mismatch is not None:
            return mismatch
        return self._change_outcome(receipt, True, "executed", type="lxc", name=hostname,
                                    core_count=cores, memory_mib=memory_mib, verification_ok=True)

    def guest_clone(self, node: str, vmid: int, *, newid: int, name: str, full: bool, pool: str | None,
                    expected_type: str, expected_name: str | None,
                    sleep=time.sleep, clock=time.monotonic) -> dict:
        """Clone the exact approved guest to one new VMID and prove the new guest exists."""
        if type(newid) is not int or not (1 <= newid <= MAX_VMID) or newid == vmid:
            raise ValueError("Enter a new, different Proxmox VMID.")
        if not isinstance(name, str) or not GUEST_NAME.fullmatch(name):
            raise ValueError("Enter a guest name of letters, digits and hyphens, starting and ending with one.")
        if not isinstance(full, bool):
            raise ValueError("Choose a full or linked clone.")
        if pool is not None and (not isinstance(pool, str) or not STORAGE_ID.fullmatch(pool)):
            raise ValueError("Enter one exact pool name.")
        receipt = self._change_receipt("guest_clone", node, newid)
        receipt.update(source_vmid=vmid, name=name)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the source guest; nothing was sent.")
        receipt.update(type=guest["type"], source_name=guest["name"])
        try:
            existing = self._guest_exists(node, newid)
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest list; nothing was cloned.")
        if existing is not None:
            return self._change_outcome(receipt, False, "conflict",
                                        f"VMID {newid} is already used on node {existing.get('node')}; "
                                        "nothing was cloned.")
        form = {"newid": str(newid), "full": "1" if full else "0"}
        form["hostname" if guest["type"] == "lxc" else "name"] = name
        if pool:
            form["pool"] = pool
        try:
            response = self._post(f"/nodes/{node}/{guest['type']}/{vmid}/clone", form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; nothing was cloned.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the clone request; it may have run. "
                                        "Check the node before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=GUEST_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the node "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        mismatch = self._confirm_new_guest(receipt, node, newid, guest["type"], name)
        if mismatch is not None:
            return mismatch
        return self._change_outcome(receipt, True, "executed", new_vmid=newid,
                                    new_name=name, verification_ok=True)

    def backup_now(self, node: str, vmid: int, *, storage: str, mode: str = "snapshot",
                   expected_type: str, expected_name: str | None,
                   sleep=time.sleep, clock=time.monotonic) -> dict:
        """Start one backup of the exact approved guest and prove a new volume exists."""
        if not isinstance(storage, str) or not STORAGE_ID.fullmatch(storage):
            raise ValueError("Enter one exact backup storage name.")
        if mode not in {"snapshot", "suspend", "stop"}:
            raise ValueError("The backup mode must be snapshot, suspend or stop.")
        receipt = self._change_receipt("backup_now", node, vmid)
        try:
            guest = self._match_guest(node, vmid, expected_type, expected_name)
        except ProviderResourceNotFound as error:
            return self._change_outcome(receipt, False, "not_found", str(error))
        except ProxmoxConflict as error:
            return self._change_outcome(receipt, False, "conflict", str(error))
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest's current state; nothing was sent.")
        receipt.update(type=guest["type"], name=guest["name"])
        if guest["lock"]:
            return self._change_outcome(receipt, False, "conflict",
                                        f"The guest is locked ({guest['lock']}); nothing was changed.")
        started = int(time.time())
        form = {"vmid": str(vmid), "storage": storage, "mode": mode, "compress": "zstd"}
        try:
            response = self._post(f"/nodes/{node}/vzdump", form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; no backup was started.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the backup request; it may have run. "
                                        "Check the node before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=BACKUP_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the storage "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        try:
            rows = self._rows(self._get(f"/nodes/{node}/storage/{quote(storage, safe='')}/content",
                                        {"content": "backup"}))
            new = [row for row in rows if row.get("vmid") == vmid
                   and isinstance(row.get("ctime"), int) and row["ctime"] >= started - 60]
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "The Proxmox task finished, but the backup storage could not be read "
                                        "back. Check the storage before trying again.")
        receipt["new_backups"] = len(new)
        if not new:
            return self._change_outcome(receipt, False, "failed",
                                        "The Proxmox task finished, but no new backup volume is listed.")
        volumes = [row.get("volid") for row in new if isinstance(row.get("volid"), str)]
        return self._change_outcome(receipt, True, "executed", storage=storage, volume=volumes[0] if volumes else None,
                                    verification_ok=True)

    def backup_restore(self, node: str, *, volume: str, newid: int, name: str, storage: str | None,
                       pool: str | None, sleep=time.sleep, clock=time.monotonic) -> dict:
        """Restore one backup volume into a NEW guest and prove the new guest exists."""
        if type(newid) is not int or not (1 <= newid <= MAX_VMID):
            raise ValueError("Enter one valid new Proxmox VMID.")
        if not isinstance(name, str) or not GUEST_NAME.fullmatch(name):
            raise ValueError("Enter a guest name of letters, digits and hyphens, starting and ending with one.")
        if not isinstance(volume, str) or len(volume) > 400 or \
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}:backup/vzdump-(lxc|qemu)-[^\s]{1,300}", volume):
            raise ValueError("Choose one backup volume from the storage content.")
        match = re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}:backup/vzdump-(lxc|qemu)-[^\s]{1,300}", volume)
        guest_type = match.group(1)
        if storage is not None and (not isinstance(storage, str) or not STORAGE_ID.fullmatch(storage)):
            raise ValueError("Enter one exact target storage name.")
        if pool is not None and (not isinstance(pool, str) or not STORAGE_ID.fullmatch(pool)):
            raise ValueError("Enter one exact pool name.")
        receipt = self._change_receipt("backup_restore", node, newid)
        receipt.update(type=guest_type, name=name, volume=volume)
        try:
            existing = self._guest_exists(node, newid)
        except (httpx.HTTPError, RuntimeError):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not confirm the guest list; nothing was restored.")
        if existing is not None:
            return self._change_outcome(receipt, False, "conflict",
                                        f"VMID {newid} is already used on node {existing.get('node')}; "
                                        "nothing was restored.")
        if guest_type == "lxc":
            path, form = f"/nodes/{node}/lxc", {"vmid": str(newid), "ostemplate": volume, "restore": "1",
                                                "hostname": name}
        else:
            path, form = f"/nodes/{node}/qemu", {"vmid": str(newid), "archive": volume, "name": name}
        if storage:
            form["storage"] = storage
        if pool:
            form["pool"] = pool
        try:
            response = self._post(path, form)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self._change_outcome(receipt, False, "error",
                                        "Proxmox could not be reached; nothing was restored.")
        except httpx.HTTPError:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        "Proxmox did not answer the restore request; it may have run. "
                                        "Check the node before trying again.", retry_safe=False)
        upid, failure = self._task_response(response, node, receipt)
        if failure is not None:
            return failure
        task = self._follow_task(node, upid, deadline_seconds=GUEST_TASK_DEADLINE_SECONDS,
                                 sleep=sleep, clock=clock)
        if task is None:
            return self._change_outcome(receipt, False, "outcome_unknown",
                                        f"Proxmox task {upid} had not finished in time; check the node "
                                        "before trying again.")
        exit_status = task.get("exitstatus") if isinstance(task.get("exitstatus"), str) else None
        receipt["task_exit_status"] = exit_status[:300] if exit_status else exit_status
        if exit_status != "OK":
            return self._change_outcome(receipt, False, "failed",
                                        f"Proxmox task {upid} ended with: {receipt['task_exit_status']}")
        mismatch = self._confirm_new_guest(receipt, node, newid, guest_type, name)
        if mismatch is not None:
            return mismatch
        return self._change_outcome(receipt, True, "executed", new_vmid=newid, new_name=name,
                                    new_type=guest_type, verification_ok=True)


def proxmox_integration_status(settings) -> dict:
    url = getattr(settings, "proxmox_base_url", None)
    token_id = getattr(settings, "proxmox_api_token_id", None)
    token_secret = getattr(settings, "proxmox_api_token_secret", None)
    return {"id": "proxmox", "name": "Proxmox VE", "mode": "governed_power_v1",
            "configured": bool(url and token_id and token_secret),
            "host": urlsplit(url).hostname if url else None,
            "verify_tls": getattr(settings, "proxmox_verify_tls", True)}


def alt_origins(value: str | None) -> tuple[ProxmoxOrigin, ...]:
    """The configured fallback origins, in order; an unreadable entry is refused."""
    entries = [part.strip() for part in (value or "").replace("\n", ",").split(",") if part.strip()]
    if len(entries) > MAX_ALT_ORIGINS:
        raise ValueError("At most four Proxmox fallback origins can be configured.")
    result = []
    for entry in entries:
        url, _, server_name = entry.partition("|")
        result.append(ProxmoxOrigin(_https_origin(url), server_name.strip() or None))
    return tuple(result)


def build_proxmox_service(settings) -> ProxmoxService | None:
    if not proxmox_integration_status(settings)["configured"]:
        return None
    return ProxmoxService(ProxmoxConfig(
        settings.proxmox_base_url, settings.proxmox_api_token_id,
        settings.proxmox_api_token_secret, settings.proxmox_verify_tls,
        settings.proxmox_ca_file, settings.proxmox_server_name,
        settings.proxmox_timeout_seconds,
        alt_origins(getattr(settings, "proxmox_alt_urls", None)),
        getattr(settings, "proxmox_read_attempts", READ_ATTEMPTS),
        getattr(settings, "proxmox_retry_wait_seconds", RETRY_WAIT_SECONDS)))
