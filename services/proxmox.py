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
                raise RuntimeError(f"Proxmox refused the read (HTTP {response.status_code}).")
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
                           "disk_bytes": _number(row, "disk"), "disk_total_bytes": _number(row, "maxdisk"),
                           "uptime_seconds": _number(row, "uptime")})
        guests.sort(key=lambda row: (row["node"].lower(), row["vmid"]))
        return {"count": len(guests), "truncated": len(guests) > MAX_ROWS, "guests": guests[:MAX_ROWS]}

    def nodes(self) -> dict:
        rows = self._rows(self._get("/nodes"), required=("node", "status"))
        return self._result(self._node_rows(rows))

    def guests(self) -> dict:
        rows = self._rows(self._get("/cluster/resources", {"type": "vm"}), required=("type", "node", "status"))
        return self._result(self._guest_rows(rows))

    def _guest_type(self, node: str, vmid: int) -> str:
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
        return matches[0]["type"]

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
            "disk_bytes": _number(data, "disk"),
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

    def overview(self) -> dict:
        """One bounded dashboard snapshot from two provider reads.

        /cluster/resources without a type filter returns the node rows and the
        guest rows together, so nodes and guests come from the same read (one
        round trip fewer than /nodes plus a filtered resources read).
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
        row = matches[0]
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
