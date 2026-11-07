"""Bounded, read-only Proxmox VE API reads for the operations specialist.

Every request uses the shared Lua-gated HTTP client. The token and host come
from machine configuration; tool arguments can select only validated node IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import ipaddress
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from services.lua_adapters.http_clients import client as lua_http_client

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ROWS = 200
MAX_TASKS = 50
NODE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
TOKEN_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}@[A-Za-z0-9_.-]{1,64}![A-Za-z0-9_.-]{1,64}")


@dataclass(frozen=True)
class ProxmoxConfig:
    base_url: str
    token_id: str
    token_secret: str = field(repr=False)
    verify_tls: bool = True
    ca_file: Path | None = None
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


class ProxmoxService:
    def __init__(self, config: ProxmoxConfig, client: httpx.Client | None = None):
        self.config = config
        self.origin = config.origin()
        if not TOKEN_ID.fullmatch(config.token_id) or not config.token_secret or len(config.token_secret) > 4096:
            raise ValueError("Proxmox API token configuration is invalid.")
        verify = str(config.ca_file) if config.ca_file is not None else config.verify_tls
        self.client = client or lua_http_client(verify=verify, timeout=config.timeout_seconds)
        self.owns_client = client is None

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _get(self, path: str, params: dict | None = None) -> list | dict:
        request = httpx.Request("GET", self.origin + "/api2/json" + path, params=params,
                                headers={"Authorization": f"PVEAPIToken={self.config.token_id}={self.config.token_secret}",
                                         "Accept": "application/json"})
        with self.client.stream(request.method, request.url, headers=request.headers) as response:
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
                   "name": _text(row, "name"), "online": row.get("online") if type(row.get("online")) is bool else None}
                  for row in rows]
        return self._result({"count": len(result), "truncated": len(result) > MAX_ROWS, "members": result[:MAX_ROWS]})

    def nodes(self) -> dict:
        rows = self._rows(self._get("/nodes"), required=("node", "status"))
        nodes = [{"node": _text(row, "node", required=True), "status": _text(row, "status", required=True),
                  "uptime_seconds": _number(row, "uptime"), "cpu_fraction": _number(row, "cpu"),
                  "cpu_count": _number(row, "maxcpu"), "memory_bytes": _number(row, "mem"),
                  "memory_total_bytes": _number(row, "maxmem"), "disk_bytes": _number(row, "disk"),
                  "disk_total_bytes": _number(row, "maxdisk")}
                 for row in rows]
        nodes.sort(key=lambda row: row["node"].lower())
        return self._result({"count": len(nodes), "truncated": len(nodes) > MAX_ROWS, "nodes": nodes[:MAX_ROWS]})

    def guests(self) -> dict:
        rows = self._rows(self._get("/cluster/resources", {"type": "vm"}), required=("type", "node", "status"))
        guests = []
        for row in rows:
            vmid = row.get("vmid")
            if type(vmid) is not int or vmid < 1:
                raise _invalid()
            guests.append({"vmid": vmid, "node": _text(row, "node", required=True),
                           "type": _text(row, "type", required=True), "name": _text(row, "name"),
                           "status": _text(row, "status", required=True), "cpu_fraction": _number(row, "cpu"),
                           "memory_bytes": _number(row, "mem"), "memory_total_bytes": _number(row, "maxmem"),
                           "disk_bytes": _number(row, "disk"), "disk_total_bytes": _number(row, "maxdisk"),
                           "uptime_seconds": _number(row, "uptime")})
        guests.sort(key=lambda row: (row["node"].lower(), row["vmid"]))
        return self._result({"count": len(guests), "truncated": len(guests) > MAX_ROWS, "guests": guests[:MAX_ROWS]})

    def node_storage(self, node: str) -> dict:
        if not isinstance(node, str) or not NODE_ID.fullmatch(node):
            raise ValueError("Enter one exact Proxmox node name.")
        rows = self._rows(self._get(f"/nodes/{node}/storage"), required=("storage", "type"))
        storage = [{"storage": _text(row, "storage", required=True), "type": _text(row, "type", required=True),
                    "content": _text(row, "content"), "active": row.get("active") if type(row.get("active")) is bool else None,
                    "enabled": row.get("enabled") if type(row.get("enabled")) is bool else None,
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
        """Read one bounded dashboard snapshot from the provider's read APIs."""
        cluster = self.cluster_status()["data"]
        nodes = self.nodes()["data"]
        guests = self.guests()["data"]
        return self._result({
            "schema": "proxmox-overview.v1",
            "cluster": cluster,
            "nodes": nodes,
            "guests": guests,
        })


def proxmox_integration_status(settings) -> dict:
    url = getattr(settings, "proxmox_base_url", None)
    token_id = getattr(settings, "proxmox_api_token_id", None)
    token_secret = getattr(settings, "proxmox_api_token_secret", None)
    return {"id": "proxmox", "name": "Proxmox VE", "mode": "read_only_v1",
            "configured": bool(url and token_id and token_secret),
            "host": urlsplit(url).hostname if url else None,
            "verify_tls": getattr(settings, "proxmox_verify_tls", True)}


def build_proxmox_service(settings) -> ProxmoxService | None:
    if not proxmox_integration_status(settings)["configured"]:
        return None
    return ProxmoxService(ProxmoxConfig(settings.proxmox_base_url, settings.proxmox_api_token_id,
                                        settings.proxmox_api_token_secret, settings.proxmox_verify_tls,
                                        settings.proxmox_ca_file, settings.proxmox_timeout_seconds))
