"""Bounded Tailscale API provider: tailnet devices, DNS, policy and device posture reads, and the
policy file's governed text write (read with its ETag, validated by Tailscale, written with If-Match).

Signs in with the deployment's own trust credential (devices:core:read,
dns:read, devices:posture_attributes:read and policy_file); the token lasts an
hour and is reused until shortly before it expires. Every answer is
normalized: device addresses, node and machine keys and user details never
leave this module. The policy is summarized with users, host aliases and
addresses replaced by fixed words.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
import re
import threading
from typing import Callable

import httpx

from services.lua_adapters.http_clients import client as lua_http_client

API = "https://api.tailscale.com/api/v2"
MAX_DEVICES = 200
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Renew this long before the token's stated expiry, so no read uses an expiring token.
TOKEN_MARGIN = timedelta(minutes=5)
# A node key that lapses within this window is flagged, so it can be renewed before the device drops off.
KEY_EXPIRING_WITHIN = timedelta(days=14)
# Advertising either default route offers the device as an exit node.
EXIT_ROUTES = frozenset({"0.0.0.0/0", "::/0"})
# Fixed order, most urgent first: what an operator should look at on a device.
ISSUE_ORDER = ("key_expired", "key_expiring", "waiting_for_approval", "tailnet_lock_error",
               "routes_waiting_for_approval", "blocks_incoming", "udp_blocked", "update_available")
# Public resolvers Tailscale reaches over DNS-over-HTTPS (Cloudflare, Google, Quad9).
DOH_RESOLVERS = frozenset({
    "1.1.1.1", "1.0.0.1", "2606:4700:4700::1111", "2606:4700:4700::1001",
    "8.8.8.8", "8.8.4.4", "2001:4860:4860::8888", "2001:4860:4860::8844",
    "9.9.9.9", "149.112.112.112", "2620:fe::fe", "2620:fe::9",
})
NEVER = "0001-01-01T00:00:00Z"
# Posture is one read per device, asked for by chat only.
MAX_POSTURE_DEVICES = 50
MAX_POLICY_RULES = 200
NODE_ID = re.compile(r"[A-Za-z0-9]{1,64}")
# Policy identifiers that name no person or address stay as written.
POLICY_NAME_PREFIXES = ("autogroup:", "tag:", "group:", "ipset:", "svc:")
ACL_PORTS = re.compile(r"\*|[0-9]{1,5}(?:-[0-9]{1,5})?(?:,[0-9]{1,5}(?:-[0-9]{1,5})?)*")
POLICY_SECTIONS = ("acls", "grants", "ssh", "tagOwners", "groups", "hosts", "tests")
# The fixed node attributes Tailscale reports; custom and integration attributes are only counted.
POSTURE_ATTRIBUTES = {"node:os": ("os", 40), "node:osVersion": ("os_version", 60),
                      "node:tsVersion": ("tailscale_version", 40), "node:tsAutoUpdate": ("auto_update", None),
                      "node:tsReleaseTrack": ("release_track", 20),
                      "node:tsStateEncrypted": ("state_encrypted", None)}


@dataclass(frozen=True)
class TailnetConfig:
    client_id: str
    client_secret: str
    timeout_seconds: float = 10.0


class TailscaleRefused(RuntimeError):
    """Tailscale answered with a status other than 200; the answer itself is never repeated."""


class PolicyConflict(RuntimeError):
    """The policy changed since it was read (If-Match failed): nothing was written or overwritten."""


def _invalid() -> RuntimeError:
    return RuntimeError("Tailscale returned an invalid answer.")


def _strings(value, limit: int = 1000) -> list[str]:
    if not isinstance(value, list) or len(value) > limit or not all(isinstance(item, str) for item in value):
        raise _invalid()
    return value


def _principal(value) -> str:
    """A policy source or destination with every user, host alias and address replaced by a fixed word."""
    if not isinstance(value, str) or not value or len(value) > 255:
        raise _invalid()
    if value == "*" or value.startswith(POLICY_NAME_PREFIXES):
        return value[:120]
    if "@" in value:
        return "a user"
    candidate = value[1:-1] if value.startswith("[") and value.endswith("]") else value
    try:
        ipaddress.ip_network(candidate, strict=False)
    except ValueError:
        return "a host alias"
    return "an address"


def _acl_destination(value) -> tuple[str, str]:
    """An ACL destination "host:ports" as its masked host and its ports."""
    if not isinstance(value, str):
        raise _invalid()
    host, _, ports = value.rpartition(":")
    if not host or not ACL_PORTS.fullmatch(ports):
        raise _invalid()
    return _principal(host), ports


def _rule(kind: str, row) -> dict:
    if not isinstance(row, dict):
        raise _invalid()
    sources = [_principal(item) for item in _strings(row.get("src"))]
    if kind == "acl":
        pairs = [_acl_destination(item) for item in _strings(row.get("dst"))]
        destinations, ports = [host for host, _ in pairs], [port for _, port in pairs]
        capabilities = 0
    else:
        destinations = [_principal(item) for item in _strings(row.get("dst"))]
        ports = [item[:40] for item in _strings(_shape(row.get("ip"), list))]
        capabilities = len(_shape(row.get("app"), dict))
    return {"kind": kind, "sources": sources, "destinations": destinations, "ports": ports,
            "app_capabilities": capabilities}


def _ssh_user(value) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise _invalid()
    return value[:120] if value == "root" or value.startswith("autogroup:") else "a named user"


def _text(value, limit: int = 200) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _invalid()
    return value[:limit]


def _time(value) -> datetime | None:
    if value in (None, "", NEVER):
        return None
    if not isinstance(value, str):
        raise _invalid()
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise _invalid() from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _flag(value) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise _invalid()


def _shape(value, kind: type):
    """A missing field reads as empty; one of the wrong type is an invalid answer, even when it is empty."""
    if value is None:
        return kind()
    if not isinstance(value, kind):
        raise _invalid()
    return value


def _address(value) -> str:
    if not isinstance(value, str):
        raise _invalid()
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        raise _invalid() from None


def _routes(value) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(route, str) for route in value):
        raise _invalid()
    return value


def _connectivity(value) -> dict | None:
    """How the device reaches its peers, from its last report; endpoints and addresses are not kept."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _invalid()
    supports = _shape(value.get("clientSupports"), dict)
    latency = _shape(value.get("latency"), dict)
    mapping = [_flag(supports.get(name)) for name in ("upnp", "pmp", "pcp")]
    relay = None
    for region, row in latency.items():
        if not isinstance(row, dict):
            raise _invalid()
        milliseconds = row.get("latencyMs")
        if milliseconds is not None and (isinstance(milliseconds, bool) or not isinstance(milliseconds, (int, float))):
            raise _invalid()
        if _flag(row.get("preferred")) is True:
            relay = (region[:60], None if milliseconds is None else round(float(milliseconds), 1))
    return {
        "udp": _flag(supports.get("udp")),
        # The router changes the public port per destination ("hard NAT"): more connections are relayed.
        "hard_nat": _flag(value.get("mappingVariesByDestIP")),
        "port_mapping": True if True in mapping else None if None in mapping else False,
        "nearest_relay": relay and relay[0],
        "relay_latency_ms": relay and relay[1],
    }


class TailnetService:
    def __init__(self, config: TailnetConfig, client: httpx.Client | None = None,
                 clock: Callable[[], datetime] | None = None):
        self.config = config
        self.client = client or lua_http_client(timeout=config.timeout_seconds)
        self.owns_client = client is None
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._token: tuple[str, datetime] | None = None
        self._lock = threading.Lock()

    def close(self):
        if self.owns_client:
            self.client.close()

    def _send(self, request: httpx.Request) -> dict | list:
        with self.client.stream(request.method, request.url, headers=request.headers,
                                content=request.content) as response:
            if response.status_code != 200:
                # The answer is never echoed: it can repeat what was sent.
                raise TailscaleRefused(f"Tailscale refused the read (HTTP {response.status_code}).")
            payload = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                payload.extend(chunk)
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise RuntimeError("Tailscale's answer exceeds the bounded response size.")
        try:
            return httpx.Response(200, content=bytes(payload)).json()
        except ValueError:
            raise _invalid() from None

    def _access_token(self) -> str:
        with self._lock:
            now = self.clock()
            if self._token is not None and now < self._token[1]:
                return self._token[0]
            grant = self._send(httpx.Request("POST", API + "/oauth/token", data={
                "client_id": self.config.client_id, "client_secret": self.config.client_secret,
                "grant_type": "client_credentials"}))
            token, lifetime = (grant.get("access_token"), grant.get("expires_in")) if isinstance(grant, dict) else (None, None)
            if not isinstance(token, str) or not token or type(lifetime) is not int or lifetime <= 0:
                raise _invalid()
            self._token = (token, now + timedelta(seconds=lifetime) - TOKEN_MARGIN)
            return token

    def _get(self, path: str, params: dict | None = None) -> dict:
        body = self._send(httpx.Request("GET", API + path, params=params, headers={
            "Authorization": f"Bearer {self._access_token()}", "Accept": "application/json"}))
        if not isinstance(body, dict):
            raise _invalid()
        return body

    def _result(self, data: dict) -> dict:
        return {"ok": True, "status": "success", "provider": "tailscale",
                "observed_at": self.clock().isoformat(), "data": data}

    def _device(self, item) -> dict:
        if not isinstance(item, dict) or not isinstance(item.get("hostname"), str) or not item["hostname"]:
            raise _invalid()
        tags = _shape(item.get("tags"), list)
        if not all(isinstance(tag, str) for tag in tags):
            raise _invalid()
        never = _flag(item.get("keyExpiryDisabled")) is True
        expires = None if never else _time(item.get("expires"))
        version = _text(item.get("clientVersion"), 60)
        advertised, enabled = _routes(item.get("advertisedRoutes")), _routes(item.get("enabledRoutes"))
        waiting = None if advertised is None or enabled is None else len(set(advertised) - set(enabled))
        lock_error = _text(item.get("tailnetLockError"))
        device = {
            "name": _text(item["hostname"], 120),
            "dns_name": _text(item.get("name"), 255),
            "os": _text(item.get("os"), 40),
            # "1.104.4-t9382131983" is version and build; the version is what an operator compares.
            "client_version": version.split("-", 1)[0] if version else None,
            "update_available": _flag(item.get("updateAvailable")),
            "online": _flag(item.get("connectedToControl")),
            "last_seen": _text(item.get("lastSeen"), 40),
            "key_expires": None if expires is None else item["expires"],
            "key_expiry_disabled": never,
            "key_expires_in_days": None if expires is None else int((expires - self.clock()) / timedelta(days=1)),
            "authorized": _flag(item.get("authorized")),
            "shared_from_another_tailnet": _flag(item.get("isExternal")),
            "tags": [tag[:60] for tag in tags[:20]],
            # "Shields up": the device accepts no incoming tailnet connections.
            "blocks_incoming": _flag(item.get("blocksIncomingConnections")),
            # Counts only; the routes themselves describe private networks.
            "routes_advertised": None if advertised is None else len(advertised),
            "routes_waiting_for_approval": waiting,
            "exit_node_offered": None if advertised is None else bool(EXIT_ROUTES & set(advertised)),
            # Whether tailnet lock reports a problem; its text is not repeated.
            "tailnet_lock_error": None if lock_error is None else bool(lock_error),
            "connectivity": _connectivity(item.get("clientConnectivity")),
        }
        found = set()
        if expires is not None:
            found.add("key_expired" if expires <= self.clock() else
                      "key_expiring" if expires - self.clock() < KEY_EXPIRING_WITHIN else "")
        found.add("waiting_for_approval" if device["authorized"] is False else "")
        found.add("tailnet_lock_error" if device["tailnet_lock_error"] else "")
        found.add("routes_waiting_for_approval" if waiting else "")
        found.add("blocks_incoming" if device["blocks_incoming"] else "")
        found.add("udp_blocked" if device["connectivity"] and device["connectivity"]["udp"] is False else "")
        found.add("update_available" if device["update_available"] else "")
        device["issues"] = [issue for issue in ISSUE_ORDER if issue in found]
        return device

    def devices(self) -> dict:
        # Every field: reachability (UDP, NAT, nearest relay), routes and shields-up come only with fields=all.
        body = self._get("/tailnet/-/devices", {"fields": "all"})
        rows = body.get("devices")
        if not isinstance(rows, list):
            raise _invalid()
        devices = [self._device(item) for item in rows]
        devices.sort(key=lambda row: row["name"].lower())
        summary = {"online": sum(row["online"] is True for row in devices),
                   "offline": sum(row["online"] is False for row in devices),
                   "with_issues": sum(bool(row["issues"]) for row in devices)}
        return self._result({"count": len(devices), "truncated": len(devices) > MAX_DEVICES,
                             "summary": summary, "devices": devices[:MAX_DEVICES]})

    def dns_settings(self) -> dict:
        body = self._get("/tailnet/-/dns/configuration")
        preferences = _shape(body.get("preferences"), dict)
        nameservers = _shape(body.get("nameservers"), list)
        split = _shape(body.get("splitDNS"), dict)
        search = _shape(body.get("searchPaths"), list)
        if not all(isinstance(item, str) for item in search):
            raise _invalid()
        resolvers = []
        for item in nameservers:
            if not isinstance(item, dict):
                raise _invalid()
            address = _address(item.get("address"))
            resolvers.append({"address": address, "encrypted": address in DOH_RESOLVERS})
        domains = []
        for domain, servers in sorted(split.items()):
            servers = _shape(servers, list)
            if not isinstance(domain, str):
                raise _invalid()
            domains.append({"domain": domain[:255], "nameservers": [
                _address(server.get("address") if isinstance(server, dict) else None) for server in servers]})
        return self._result({
            "magic_dns": _flag(preferences.get("magicDNS")) is True,
            "override_local_dns": _flag(preferences.get("overrideLocalDNS")) is True,
            "nameservers": resolvers, "split_dns": domains, "search_domains": [item[:255] for item in search[:20]],
        })


    def policy(self) -> dict:
        """Who may reach what under the tailnet policy, with users, host aliases and addresses masked."""
        body = self._get("/tailnet/-/acl")
        try:
            settings = self._get("/tailnet/-/settings")
            managed = _flag(settings.get("aclsExternallyManagedOn"))
        except (TailscaleRefused, httpx.TransportError):
            managed = None
        rules = [_rule("acl", row) for row in _shape(body.get("acls"), list)]
        rules += [_rule("grant", row) for row in _shape(body.get("grants"), list)]
        ssh = []
        for row in _shape(body.get("ssh"), list):
            if not isinstance(row, dict) or not isinstance(row.get("action"), str):
                raise _invalid()
            ssh.append({"action": row["action"][:20],
                        "sources": [_principal(item) for item in _strings(row.get("src"))],
                        "destinations": [_principal(item) for item in _strings(row.get("dst"))],
                        "users": [_ssh_user(item) for item in _strings(_shape(row.get("users"), list))]})
        owners = []
        for tag, names in sorted(_shape(body.get("tagOwners"), dict).items()):
            if not isinstance(tag, str):
                raise _invalid()
            owners.append({"tag": tag[:120], "owners": [_principal(item) for item in _strings(names)]})
        groups = []
        for name, members in sorted(_shape(body.get("groups"), dict).items()):
            if not isinstance(name, str):
                raise _invalid()
            groups.append({"name": name[:120], "members": len(_strings(members))})
        hosts = _shape(body.get("hosts"), dict)
        if not all(isinstance(value, str) for value in hosts.values()):
            raise _invalid()
        # Every member reaches every device on every port.
        open_to_all = any(set(rule["sources"]) & {"*", "autogroup:member"}
                          and any(host == "*" and port == "*" for host, port in zip(rule["destinations"], rule["ports"]))
                          if rule["kind"] == "acl" else
                          set(rule["sources"]) & {"*", "autogroup:member"} and "*" in rule["destinations"]
                          and "*" in rule["ports"] for rule in rules)
        return self._result({
            "schema": "tailnet-policy.v1",
            "externally_managed": managed,
            "truncated": len(rules) > MAX_POLICY_RULES or len(ssh) > MAX_POLICY_RULES,
            "rules": rules[:MAX_POLICY_RULES], "ssh": ssh[:MAX_POLICY_RULES],
            "tag_owners": owners[:MAX_POLICY_RULES], "groups": groups[:MAX_POLICY_RULES],
            "hosts": len(hosts), "tests": len(_shape(body.get("tests"), list)),
            "other_sections": sorted(str(key)[:60] for key in body if key not in POLICY_SECTIONS)[:40],
            "summary": {"rules": len(rules), "ssh_rules": len(ssh), "tags": len(owners), "groups": len(groups),
                        "open_to_all_members": bool(open_to_all)},
        })

    # ---- The policy file as text: read with its ETag, validate, write with If-Match (D2) ----

    def _policy_request(self, method: str, path: str, *, content: str | None = None,
                        headers: dict | None = None) -> httpx.Response:
        sent = {"Authorization": f"Bearer {self._access_token()}", **(headers or {})}
        response = self.client.request(method, API + path, headers=sent,
                                       content=None if content is None else content.encode("utf-8"))
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise RuntimeError("Tailscale's answer exceeds the bounded response size.")
        return response

    def policy_text(self) -> tuple[str, str]:
        """The policy file exactly as stored (HuJSON, comments kept) and its ETag."""
        response = self._policy_request("GET", "/tailnet/-/acl", headers={"Accept": "application/hujson"})
        if response.status_code != 200:
            raise TailscaleRefused(f"Tailscale refused the read (HTTP {response.status_code}).")
        etag = response.headers.get("etag")
        if not etag:
            raise TailscaleRefused("Tailscale returned the policy without an ETag, so it cannot be changed safely.")
        return response.text, etag

    def validate_policy(self, text: str) -> str | None:
        """Tailscale's own check of a changed policy (it writes nothing): None when valid, else why not."""
        response = self._policy_request("POST", "/tailnet/-/acl/validate", content=text,
                                        headers={"Content-Type": "application/hujson",
                                                 "Accept": "application/json"})
        if response.status_code != 200:
            raise TailscaleRefused(f"Tailscale refused to validate the policy (HTTP {response.status_code}).")
        try:
            body = response.json() if response.content.strip() else {}
        except ValueError:
            raise _invalid() from None
        message = body.get("message") if isinstance(body, dict) else None
        if not message:
            return None
        # Only the short reason is repeated; details can name users and addresses.
        reason = str(message).strip().rstrip(".")[:160]
        return f"Tailscale rejected the changed policy: {reason}."

    def write_policy(self, text: str, etag: str) -> str | None:
        """Write the policy only if it is still the version read (If-Match); returns the new ETag."""
        if not etag:
            raise ValueError("A policy write needs the ETag of the version it changes.")
        response = self._policy_request("POST", "/tailnet/-/acl", content=text,
                                        headers={"Content-Type": "application/hujson", "If-Match": etag,
                                                 "Accept": "application/hujson"})
        if response.status_code == 412:
            raise PolicyConflict("The tailnet policy changed since it was read; nothing was written.")
        if response.status_code != 200:
            raise TailscaleRefused(f"Tailscale refused the policy write (HTTP {response.status_code}).")
        return response.headers.get("etag")

    def device_posture(self) -> dict:
        """Each device's fixed posture attributes (OS, Tailscale version, auto-update, encrypted state)."""
        rows = self._get("/tailnet/-/devices").get("devices")
        if not isinstance(rows, list):
            raise _invalid()
        devices = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("hostname"), str) or not row["hostname"] \
                    or not isinstance(row.get("nodeId"), str) or not NODE_ID.fullmatch(row["nodeId"]):
                raise _invalid()
            devices.append((row["hostname"][:120], row["nodeId"]))
        devices.sort(key=lambda device: device[0].lower())
        result = []
        for name, node_id in devices[:MAX_POSTURE_DEVICES]:
            row: dict = {"name": name, **{field: None for field, _ in POSTURE_ATTRIBUTES.values()},
                         "other_attributes": None}
            try:
                attributes = _shape(self._get(f"/device/{node_id}/attributes").get("attributes"), dict)
            except (TailscaleRefused, httpx.TransportError):
                result.append(row)
                continue
            for key, value in attributes.items():
                if key in POSTURE_ATTRIBUTES:
                    field, limit = POSTURE_ATTRIBUTES[key]
                    row[field] = _flag(value) if limit is None else _text(value, limit)
            row["other_attributes"] = sum(key not in POSTURE_ATTRIBUTES for key in attributes)
            result.append(row)
        return self._result({
            "schema": "tailnet-posture.v1", "count": len(devices), "truncated": len(devices) > MAX_POSTURE_DEVICES,
            "devices": result,
            "summary": {"auto_update_off": sum(row["auto_update"] is False for row in result),
                        "state_not_encrypted": sum(row["state_encrypted"] is False for row in result)}})


def tailnet_integration_status(settings) -> dict:
    return {"configured": bool(getattr(settings, "tailscale_client_id", None)
                               and getattr(settings, "tailscale_client_secret", None))}


def build_tailnet_service(settings) -> TailnetService | None:
    if not tailnet_integration_status(settings)["configured"]:
        return None
    return TailnetService(TailnetConfig(settings.tailscale_client_id, settings.tailscale_client_secret,
                                        settings.tailscale_timeout_seconds))
