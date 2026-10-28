"""One labelled observation from the fixed read adapters (FR-093).

Reads the allow-listed containers, the configured health endpoints and the
configured OpenWrt router through the same services the read tools use. Each
source is contained: a failure marks only that source unavailable, keeps a
stable error code and drops provider messages, URLs and paths.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re

import httpx

from services.engineering.topology import NODE_ID

INTERFACE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,31}")
MAX_INTERFACE_READS = 16
MAX_TEXT = 200
KNOWN_ERROR_CODES = {"not_configured", "provider_error", "provider_unavailable", "provider_timeout",
                     "container_not_allowed", "not_found"}


def _text(value) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split())[:MAX_TEXT]


def _scalar(value):
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int) and abs(value) <= 2 ** 53:
        return value
    return _text(value)


def _failure(source: str, result: dict) -> dict:
    code = result.get("error_code")
    if code == "not_configured":
        return {"id": source, "state": "not_configured", "error_code": "not_configured"}
    return {"id": source, "state": "unavailable",
            "error_code": code if code in KNOWN_ERROR_CODES else "provider_error"}


def _exception_code(exc: Exception) -> str:
    return "provider_timeout" if isinstance(exc, httpx.TimeoutException) else "provider_unavailable"


def _read(source: str, read):
    """(result, source row). The adapter's own message never leaves this function."""
    try:
        result = read()
    except Exception as exc:  # noqa: BLE001 - every adapter failure is contained to its source
        return None, {"id": source, "state": "unavailable", "error_code": _exception_code(exc)}
    if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("data"), dict):
        return None, _failure(source, result if isinstance(result, dict) else {})
    return result["data"], {"id": source, "state": "observed", "error_code": None}


def _containers(service, nodes: list, absent: list) -> dict:
    if service is None:
        return {"id": "containers", "state": "not_configured", "error_code": "not_configured"}
    data, row = _read("containers", service.container_status)
    for item in (data or {}).get("containers", []):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        identity = "container:" + item["name"]
        if not NODE_ID.fullmatch(identity):
            continue
        if item.get("found") is False:
            absent.append(identity)
        elif item.get("found") is True:
            nodes.append({"id": identity, "kind": "container", "label": item["name"], "properties": {
                "state": _scalar(item.get("state")), "health": _scalar(item.get("health")),
                "restart_count": _scalar(item.get("restart_count")), "image": _scalar(item.get("image"))}})
    return row


def _endpoints(service, nodes: list) -> dict:
    if service is None:
        return {"id": "endpoints", "state": "not_configured", "error_code": "not_configured"}
    data, row = _read("endpoints", service.service_health)
    for item in (data or {}).get("services", []):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        identity = "endpoint:" + item["name"]
        if NODE_ID.fullmatch(identity) and isinstance(item.get("up"), bool):
            # The probe URL is configuration, not observation; it is never returned.
            nodes.append({"id": identity, "kind": "service_endpoint", "label": item["name"],
                          "properties": {"up": item["up"], "http_status": _scalar(item.get("http_status"))}})
    return row


def _interface_names(topology: dict | None, defaults: list[str]) -> list[str]:
    names = list(dict.fromkeys(defaults))
    for node in (topology or {}).get("nodes", []):
        if isinstance(node, dict) and isinstance(node.get("id"), str) and node["id"].startswith("interface:"):
            name = node["id"].split(":", 1)[1]
            if name not in names:
                names.append(name)
    return names


def _address(data: dict) -> str | None:
    rows = data.get("ipv4-address")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        address, mask = rows[0].get("address"), rows[0].get("mask")
        if isinstance(address, str) and isinstance(mask, int):
            return _text(f"{address}/{mask}")
    return None


def _router(service, topology, defaults, nodes: list, edges: list, unread: list) -> dict:
    if service is None:
        return {"id": "router", "state": "not_configured", "error_code": "not_configured"}
    board, row = _read("router", service.system_info)
    if board is None:
        return row
    hostname = _text(board.get("hostname")) or "router"
    router_id = "router:" + hostname
    if not NODE_ID.fullmatch(router_id):
        router_id = "router:router"
    release = board.get("release") if isinstance(board.get("release"), dict) else {}
    nodes.append({"id": router_id, "kind": "router", "label": hostname, "properties": {
        "hostname": hostname, "kernel": _text(board.get("kernel")),
        "release": _text(release.get("version")), "target": _text(release.get("target"))}})
    reads = 0
    for name in _interface_names(topology, defaults):
        identity = "interface:" + name
        if not INTERFACE_NAME.fullmatch(name) or not NODE_ID.fullmatch(identity) or reads >= MAX_INTERFACE_READS:
            unread.append(identity)
            continue
        reads += 1
        data, status = _read("router", lambda name=name: service.interface_status(name))
        if data is None:
            # A refused or unknown interface is unread, never evidence that it is absent.
            unread.append(identity)
            continue
        nodes.append({"id": identity, "kind": "interface", "label": name, "properties": {
            "up": data.get("up") if isinstance(data.get("up"), bool) else None,
            "available": data.get("available") if isinstance(data.get("available"), bool) else None,
            "proto": _text(data.get("proto")), "device": _text(data.get("device")),
            "ipv4_address": _address(data)}})
        edges.append({"from": identity, "to": router_id, "relation": "hosted_on"})
    return row


def observe(*, containers, endpoints, router, topology: dict | None, default_interfaces: list[str],
            now: datetime | None = None) -> dict:
    """Read every source once; unreadable sources are labelled, never guessed."""
    nodes, edges, absent, unread = [], [], [], []
    sources = [
        _containers(containers, nodes, absent),
        _endpoints(endpoints, nodes),
        _router(router, topology, default_interfaces, nodes, edges, unread),
    ]
    body = {"sources": sources, "nodes": nodes, "edges": edges, "absent": sorted(absent), "unread": sorted(unread)}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    observed_at = (now or datetime.now(timezone.utc)).isoformat()
    return {"schema": "infrastructure-observation.v1", "state_kind": "observed", "observed_at": observed_at,
            "observation_sha256": digest, **body}
