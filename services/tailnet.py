"""Bounded read-only Tailscale API provider: tailnet devices and DNS settings.

Signs in with the deployment's own OAuth client, which should hold only the
devices:core:read and dns:read scopes; the token lasts an hour and is reused
until shortly before it expires. Every answer is normalized: device addresses,
node and machine keys and user details never leave this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import ipaddress
import threading
from typing import Callable

import httpx

from services.lua_adapters.http_clients import client as lua_http_client

API = "https://api.tailscale.com/api/v2"
MAX_DEVICES = 200
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Renew this long before the token's stated expiry, so no read uses an expiring token.
TOKEN_MARGIN = timedelta(minutes=5)
# Public resolvers Tailscale reaches over DNS-over-HTTPS (Cloudflare, Google, Quad9).
DOH_RESOLVERS = frozenset({
    "1.1.1.1", "1.0.0.1", "2606:4700:4700::1111", "2606:4700:4700::1001",
    "8.8.8.8", "8.8.4.4", "2001:4860:4860::8888", "2001:4860:4860::8844",
    "9.9.9.9", "149.112.112.112", "2620:fe::fe", "2620:fe::9",
})
NEVER = "0001-01-01T00:00:00Z"


@dataclass(frozen=True)
class TailnetConfig:
    client_id: str
    client_secret: str
    timeout_seconds: float = 10.0


def _invalid() -> RuntimeError:
    return RuntimeError("Tailscale returned an invalid answer.")


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


def _address(value) -> str:
    if not isinstance(value, str):
        raise _invalid()
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        raise _invalid() from None


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
                raise RuntimeError(f"Tailscale refused the read (HTTP {response.status_code}).")
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

    def _get(self, path: str) -> dict:
        body = self._send(httpx.Request("GET", API + path, headers={
            "Authorization": f"Bearer {self._access_token()}", "Accept": "application/json"}))
        if not isinstance(body, dict):
            raise _invalid()
        return body

    def _result(self, data: dict) -> dict:
        return {"ok": True, "status": "observed", "provider": "tailscale",
                "observed_at": self.clock().isoformat(), "data": data}

    def _device(self, item) -> dict:
        if not isinstance(item, dict) or not isinstance(item.get("hostname"), str) or not item["hostname"]:
            raise _invalid()
        tags = item.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise _invalid()
        never = _flag(item.get("keyExpiryDisabled")) is True
        expires = None if never else _time(item.get("expires"))
        version = _text(item.get("clientVersion"), 60)
        return {
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
        }

    def devices(self) -> dict:
        body = self._get("/tailnet/-/devices")
        rows = body.get("devices")
        if not isinstance(rows, list):
            raise _invalid()
        devices = [self._device(item) for item in rows]
        devices.sort(key=lambda row: row["name"].lower())
        return self._result({"count": len(devices), "truncated": len(devices) > MAX_DEVICES,
                             "devices": devices[:MAX_DEVICES]})

    def dns_settings(self) -> dict:
        body = self._get("/tailnet/-/dns/configuration")
        preferences = body.get("preferences") or {}
        nameservers = body.get("nameservers") or []
        split = body.get("splitDNS") or {}
        search = body.get("searchPaths") or []
        if (not isinstance(preferences, dict) or not isinstance(nameservers, list) or not isinstance(split, dict)
                or not isinstance(search, list) or not all(isinstance(item, str) for item in search)):
            raise _invalid()
        resolvers = []
        for item in nameservers:
            if not isinstance(item, dict):
                raise _invalid()
            address = _address(item.get("address"))
            resolvers.append({"address": address, "encrypted": address in DOH_RESOLVERS})
        domains = []
        for domain, servers in sorted(split.items()):
            servers = servers or []
            if not isinstance(domain, str) or not isinstance(servers, list):
                raise _invalid()
            domains.append({"domain": domain[:255], "nameservers": [
                _address(server.get("address") if isinstance(server, dict) else None) for server in servers]})
        return self._result({
            "magic_dns": _flag(preferences.get("magicDNS")) is True,
            "override_local_dns": _flag(preferences.get("overrideLocalDNS")) is True,
            "nameservers": resolvers, "split_dns": domains, "search_domains": [item[:255] for item in search[:20]],
        })


def tailnet_integration_status(settings) -> dict:
    return {"configured": bool(getattr(settings, "tailscale_client_id", None)
                               and getattr(settings, "tailscale_client_secret", None))}


def build_tailnet_service(settings) -> TailnetService | None:
    if not tailnet_integration_status(settings)["configured"]:
        return None
    return TailnetService(TailnetConfig(settings.tailscale_client_id, settings.tailscale_client_secret,
                                        settings.tailscale_timeout_seconds))
