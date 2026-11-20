"""Governed tailnet device, DNS and key changes (owner decision 20 Nov 2026, ~14:10 WIB, item 1).

The pattern of the grant change (services/tailnet_policy_change.py), applied to the rest of the
control API Janus's client may now write (scopes devices:core, devices:routes, dns, auth_keys):

* ``prepare_*`` runs before approval and writes nothing. It reads the target, refuses what cannot
  apply, and freezes the target's identity and the value being replaced.
* ``apply_*`` runs after approval. It re-reads the target and refuses if anything moved, sends
  exactly one request (never through the read retry), and reads the result back. Once a request
  may have reached Tailscale, a lost answer is ``outcome_unknown`` with ``retry_safe`` false.

Device keys and auth-key secrets never appear in a result; only names, ids, tags and counts do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import re
from typing import Any

import httpx

from services.tailnet import API, ENROLMENT_TAG, KEY_ID, MAX_RESPONSE_BYTES, NODE_ID, TailnetService, TailscaleRefused

# A device's settings Janus may change, each with the API path below /device/{id} and its body.
DEVICE_SETTINGS = ("routes", "authorized", "tags", "key_expiry", "name", "remove")
DNS_SETTINGS = ("nameservers", "magic_dns", "search_paths", "split_dns")
DEVICE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
DOMAIN = re.compile(r"(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)*[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")
MAX_ROUTES = 32
MAX_NAMESERVERS = 8
MAX_TAGS = 8


class TailnetChangeRefused(ValueError):
    """The change cannot apply; nothing was sent."""


def _refused(status: str, error: str) -> dict[str, Any]:
    return {"ok": False, "status": status, "error": error}


def _cidr(value) -> str:
    try:
        network = ipaddress.ip_network(value, strict=True)
    except (TypeError, ValueError):
        raise TailnetChangeRefused(f"{value!r} is not a subnet route such as 10.77.0.0/24.") from None
    return str(network)


def _address(value) -> str:
    try:
        return str(ipaddress.ip_address(value))
    except (TypeError, ValueError):
        raise TailnetChangeRefused(f"{value!r} is not a nameserver address.") from None


def _tag(value) -> str:
    if not isinstance(value, str) or not ENROLMENT_TAG.fullmatch(value):
        raise TailnetChangeRefused("Tags are named tag:<name>, for example tag:janus-lab.")
    return value


@dataclass(frozen=True)
class DeviceChange:
    """One setting of one device, named by the device's tailnet name exactly as Tailscale lists it."""
    device: str
    setting: str
    value: dict = field(default_factory=dict)

    def validated(self) -> "DeviceChange":
        if not isinstance(self.device, str) or not self.device.strip() or len(self.device) > 120:
            raise TailnetChangeRefused("Name one tailnet device exactly as the device list shows it.")
        if self.setting not in DEVICE_SETTINGS:
            raise TailnetChangeRefused("A device change sets routes, authorized, tags, key expiry or name, "
                                       "or removes the device.")
        value = dict(self.value or {})
        if self.setting == "routes":
            routes = value.get("routes")
            if not isinstance(routes, list) or len(routes) > MAX_ROUTES:
                raise TailnetChangeRefused(f"Name the routes to enable (at most {MAX_ROUTES}); an empty list "
                                           "turns every route off.")
            value = {"routes": sorted({_cidr(route) for route in routes})}
        elif self.setting == "authorized":
            if type(value.get("authorized")) is not bool:
                raise TailnetChangeRefused("Say whether the device is authorized (true) or not (false).")
            value = {"authorized": value["authorized"]}
        elif self.setting == "tags":
            tags = value.get("tags")
            if not isinstance(tags, list) or not tags or len(tags) > MAX_TAGS:
                raise TailnetChangeRefused(f"Name one to {MAX_TAGS} tags, for example tag:janus-lab.")
            value = {"tags": sorted({_tag(tag) for tag in tags})}
        elif self.setting == "key_expiry":
            if type(value.get("key_expiry_disabled")) is not bool:
                raise TailnetChangeRefused("Say whether the device key never expires (true) or expires (false).")
            value = {"key_expiry_disabled": value["key_expiry_disabled"]}
        elif self.setting == "name":
            name = value.get("name")
            if not isinstance(name, str) or not DEVICE_NAME.fullmatch(name):
                raise TailnetChangeRefused("A device name is lowercase letters, digits and '-', up to 63.")
            value = {"name": name}
        else:
            value = {}
        return DeviceChange(self.device.strip(), self.setting, value)


def _devices(service: TailnetService) -> list[dict]:
    rows = service._get("/tailnet/-/devices", {"fields": "all"}).get("devices")
    if not isinstance(rows, list):
        raise TailscaleRefused("Tailscale answered the device list in an unexpected form.")
    return [row for row in rows if isinstance(row, dict)]


def _find(service: TailnetService, name: str) -> dict:
    wanted = name.lower()
    matches = [row for row in _devices(service)
               if wanted in {str(row.get("hostname") or "").lower(),
                             str(row.get("name") or "").lower(),
                             str(row.get("name") or "").split(".", 1)[0].lower()}]
    if not matches:
        raise TailnetChangeRefused(f"There is no device named {name} on the tailnet.")
    if len(matches) > 1:
        raise TailnetChangeRefused(f"{len(matches)} devices are named {name}; rename one first.")
    row = matches[0]
    if not isinstance(row.get("nodeId"), str) or not NODE_ID.fullmatch(row["nodeId"]):
        raise TailscaleRefused("Tailscale listed the device without its id.")
    return row


def _short_name(row: dict) -> str:
    return str(row.get("name") or "").split(".", 1)[0]


def _current(row: dict, setting: str) -> dict:
    if setting == "routes":
        return {"enabled_routes": sorted(row.get("enabledRoutes") or [])}
    if setting == "authorized":
        return {"authorized": row.get("authorized") is True}
    if setting == "tags":
        return {"tags": sorted(row.get("tags") or [])}
    if setting == "key_expiry":
        return {"key_expiry_disabled": row.get("keyExpiryDisabled") is True}
    if setting == "name":
        return {"name": _short_name(row)}
    return {"present": True}


def _identity(row: dict) -> dict:
    return {"device_id": row["nodeId"], "device_name": str(row.get("hostname") or "")[:120],
            "device_tags": sorted(row.get("tags") or [])}


def prepare_device_change(service: TailnetService, change: DeviceChange) -> dict[str, Any]:
    """Read-only: the exact device and the value being replaced, frozen for the approval."""
    try:
        change = change.validated()
        row = _find(service, change.device)
        if change.setting == "routes":
            advertised = set(row.get("advertisedRoutes") or [])
            missing = [route for route in change.value["routes"] if route not in advertised]
            if missing:
                return _refused("denied", f"{change.device} does not advertise {', '.join(missing)}; the device "
                                          "must advertise a route (tailscale set --advertise-routes) before it can "
                                          "be approved.")
        if _current(row, change.setting) == _target(change) and change.setting != "remove":
            return _refused("denied", f"{change.device} already has that setting; nothing would change.")
    except TailnetChangeRefused as error:
        return _refused("denied", str(error))
    return {"ok": True, "status": "ready", "frozen": {**_identity(row), "previous": _current(row, change.setting)}}


def _target(change: DeviceChange) -> dict:
    if change.setting == "routes":
        return {"enabled_routes": change.value["routes"]}
    return dict(change.value)


def _send_once(service: TailnetService, method: str, path: str, body: dict | None) -> httpx.Response:
    """Exactly one write; the read retry in TailnetService._send never applies to it."""
    response = service.client.request(method, API + path, json=body, headers={
        "Authorization": f"Bearer {service._access_token()}", "Accept": "application/json"})
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RuntimeError("Tailscale's answer exceeds the bounded response size.")
    return response


def _device_request(change: DeviceChange, device_id: str) -> tuple[str, str, dict | None]:
    path = f"/device/{device_id}"
    value = change.value
    return {
        "routes": ("POST", path + "/routes", {"routes": value.get("routes")}),
        "authorized": ("POST", path + "/authorized", {"authorized": value.get("authorized")}),
        "tags": ("POST", path + "/tags", {"tags": value.get("tags")}),
        "key_expiry": ("POST", path + "/key", {"keyExpiryDisabled": value.get("key_expiry_disabled")}),
        "name": ("POST", path + "/name", {"name": value.get("name")}),
        "remove": ("DELETE", path, None),
    }[change.setting]


def apply_device_change(service: TailnetService, change: DeviceChange, *, device_id: str, device_name: str,
                        device_tags: list[str], previous: dict) -> dict[str, Any]:
    """The approved change: refused if the device moved, sent once, then read back."""
    base = {"provider": "tailscale", "operation": f"tailnet_device_{change.setting}", "device": device_name,
            "changed": False, "verification_ok": False, "retry_safe": True}
    try:
        change = change.validated()
        row = _find(service, change.device)
    except TailnetChangeRefused as error:
        return {**base, "ok": False, "status": "conflict", "error": f"{error} Nothing was sent."}
    if _identity(row) != {"device_id": device_id, "device_name": device_name, "device_tags": sorted(device_tags)} \
            or _current(row, change.setting) != previous:
        return {**base, "ok": False, "status": "conflict",
                "error": f"{device_name} changed since the approval; nothing was sent. Propose the change again."}
    method, path, body = _device_request(change, device_id)
    try:
        response = _send_once(service, method, path, body)
    except (httpx.TransportError, RuntimeError):
        return {**base, "ok": False, "status": "outcome_unknown", "retry_safe": False,
                "error": "Tailscale did not confirm the change. Read the device list before trying again."}
    if response.status_code not in (200, 204):
        return {**base, "ok": False, "status": "failed",
                "error": f"Tailscale refused the change (HTTP {response.status_code})."}
    base.update(changed=True, retry_safe=False)
    try:
        rows = _devices(service)
    except (httpx.TransportError, RuntimeError, TailscaleRefused):
        return {**base, "ok": False, "status": "outcome_unknown",
                "error": "Tailscale accepted the change, but Janus could not read the device back."}
    after = next((item for item in rows if item.get("nodeId") == device_id), None)
    if change.setting == "remove":
        verified, observed = after is None, {"present": after is not None}
    else:
        observed = _current(after, change.setting) if after is not None else {}
        verified = after is not None and observed == _target(change)
    if not verified:
        return {**base, "ok": False, "status": "outcome_unknown", "observed": observed,
                "error": "Tailscale accepted the change, but the device read back does not show it."}
    return {**base, "ok": True, "status": "executed", "verification_ok": True, "observed": observed,
            "previous": previous, "error": None}


# ---- DNS ----

def _dns_value(setting: str, value: dict) -> dict:
    if setting == "nameservers":
        servers = value.get("nameservers")
        if not isinstance(servers, list) or len(servers) > MAX_NAMESERVERS:
            raise TailnetChangeRefused(f"Name up to {MAX_NAMESERVERS} nameserver addresses.")
        return {"nameservers": [_address(server) for server in servers]}
    if setting == "magic_dns":
        if type(value.get("magic_dns")) is not bool:
            raise TailnetChangeRefused("Say whether MagicDNS is on (true) or off (false).")
        return {"magic_dns": value["magic_dns"]}
    if setting == "search_paths":
        paths = value.get("search_paths")
        if not isinstance(paths, list) or len(paths) > 10 or not all(
                isinstance(path, str) and DOMAIN.fullmatch(path) for path in paths):
            raise TailnetChangeRefused("Name up to 10 search domains, for example lab.internal.")
        return {"search_paths": list(paths)}
    if setting == "split_dns":
        domain, servers = value.get("domain"), value.get("nameservers")
        if not isinstance(domain, str) or not DOMAIN.fullmatch(domain):
            raise TailnetChangeRefused("Name one domain, for example lab.internal.")
        if servers is not None and (not isinstance(servers, list) or not servers or len(servers) > MAX_NAMESERVERS):
            raise TailnetChangeRefused("Name the domain's nameservers, or none to remove its split DNS.")
        return {"domain": domain, "nameservers": None if servers is None else [_address(s) for s in servers]}
    raise TailnetChangeRefused("A DNS change sets nameservers, MagicDNS, search domains or one split-DNS domain.")


def _dns_read(service: TailnetService, setting: str, value: dict) -> dict:
    if setting == "nameservers":
        return {"nameservers": list(service._get("/tailnet/-/dns/nameservers").get("dns") or [])}
    if setting == "magic_dns":
        return {"magic_dns": service._get("/tailnet/-/dns/preferences").get("magicDNS") is True}
    if setting == "search_paths":
        return {"search_paths": list(service._get("/tailnet/-/dns/searchpaths").get("searchPaths") or [])}
    split = service._get("/tailnet/-/dns/split-dns")
    servers = split.get(value["domain"])
    return {"domain": value["domain"], "nameservers": None if servers is None else list(servers)}


def _dns_request(setting: str, value: dict) -> tuple[str, str, dict]:
    if setting == "nameservers":
        return "POST", "/tailnet/-/dns/nameservers", {"dns": value["nameservers"]}
    if setting == "magic_dns":
        return "POST", "/tailnet/-/dns/preferences", {"magicDNS": value["magic_dns"]}
    if setting == "search_paths":
        return "POST", "/tailnet/-/dns/searchpaths", {"searchPaths": value["search_paths"]}
    return "PATCH", "/tailnet/-/dns/split-dns", {value["domain"]: value["nameservers"]}


def prepare_dns_change(service: TailnetService, setting: str, value: dict) -> dict[str, Any]:
    """Read-only: validate the new value and freeze the current one."""
    try:
        wanted = _dns_value(setting, value)
        current = _dns_read(service, setting, wanted)
    except TailnetChangeRefused as error:
        return _refused("denied", str(error))
    if current == wanted:
        return _refused("denied", "The tailnet DNS already has that setting; nothing would change.")
    return {"ok": True, "status": "ready", "frozen": {"previous": current}}


def apply_dns_change(service: TailnetService, setting: str, value: dict, *, previous: dict) -> dict[str, Any]:
    base = {"provider": "tailscale", "operation": f"tailnet_dns_{setting}", "changed": False,
            "verification_ok": False, "retry_safe": True}
    try:
        wanted = _dns_value(setting, value)
    except TailnetChangeRefused as error:
        return {**base, "ok": False, "status": "conflict", "error": f"{error} Nothing was sent."}
    if _dns_read(service, setting, wanted) != previous:
        return {**base, "ok": False, "status": "conflict",
                "error": "The tailnet DNS changed since the approval; nothing was sent. Propose it again."}
    method, path, body = _dns_request(setting, wanted)
    try:
        response = _send_once(service, method, path, body)
    except (httpx.TransportError, RuntimeError):
        return {**base, "ok": False, "status": "outcome_unknown", "retry_safe": False,
                "error": "Tailscale did not confirm the DNS change. Read the DNS settings before trying again."}
    if response.status_code != 200:
        return {**base, "ok": False, "status": "failed",
                "error": f"Tailscale refused the DNS change (HTTP {response.status_code})."}
    base.update(changed=True, retry_safe=False)
    try:
        observed = _dns_read(service, setting, wanted)
    except (httpx.TransportError, RuntimeError, TailscaleRefused):
        return {**base, "ok": False, "status": "outcome_unknown",
                "error": "Tailscale accepted the DNS change, but Janus could not read it back."}
    if observed != wanted:
        return {**base, "ok": False, "status": "outcome_unknown", "observed": observed,
                "error": "Tailscale accepted the DNS change, but the setting read back differs."}
    return {**base, "ok": True, "status": "executed", "verification_ok": True, "observed": observed,
            "previous": previous, "error": None}


# ---- keys ----

def prepare_key_revoke(service: TailnetService, key_id: str) -> dict[str, Any]:
    """Read-only: the key exists, it is not the client Janus itself runs on, and what it is for."""
    if not isinstance(key_id, str) or not KEY_ID.fullmatch(key_id):
        return _refused("denied", "Name one key by its id (letters and digits) as the key list shows it.")
    if key_id == service.config.client_id:
        return _refused("denied", "That key is the client Janus itself uses; Janus never revokes its own access.")
    try:
        key = service._get(f"/tailnet/-/keys/{key_id}")
    except TailscaleRefused:
        return _refused("denied", f"There is no key {key_id} on the tailnet.")
    tags = (((key.get("capabilities") or {}).get("devices") or {}).get("create") or {}).get("tags") or []
    return {"ok": True, "status": "ready", "frozen": {
        "key_type": str(key.get("keyType") or "auth")[:20], "key_description": str(key.get("description") or "")[:50],
        "key_tags": sorted(str(tag)[:60] for tag in tags)}}


def apply_key_revoke(service: TailnetService, key_id: str, *, key_type: str, key_description: str,
                     key_tags: list[str]) -> dict[str, Any]:
    base = {"provider": "tailscale", "operation": "tailnet_key_revoke", "key": key_id, "changed": False,
            "verification_ok": False, "retry_safe": True}
    current = prepare_key_revoke(service, key_id)
    if not current["ok"]:
        return {**base, "ok": False, "status": "conflict", "error": current["error"] + " Nothing was sent."}
    if current["frozen"] != {"key_type": key_type, "key_description": key_description, "key_tags": sorted(key_tags)}:
        return {**base, "ok": False, "status": "conflict",
                "error": "The key changed since the approval; nothing was sent."}
    try:
        response = _send_once(service, "DELETE", f"/tailnet/-/keys/{key_id}", None)
    except (httpx.TransportError, RuntimeError):
        return {**base, "ok": False, "status": "outcome_unknown", "retry_safe": False,
                "error": "Tailscale did not confirm the revocation. Read the key list before trying again."}
    if response.status_code not in (200, 204):
        return {**base, "ok": False, "status": "failed",
                "error": f"Tailscale refused the revocation (HTTP {response.status_code})."}
    base.update(changed=True, retry_safe=False)
    try:
        service._get(f"/tailnet/-/keys/{key_id}")
    except TailscaleRefused:
        return {**base, "ok": True, "status": "executed", "verification_ok": True, "error": None}
    except (httpx.TransportError, RuntimeError):
        pass
    return {**base, "ok": False, "status": "outcome_unknown",
            "error": "Tailscale accepted the revocation, but Janus could not confirm the key is gone."}
