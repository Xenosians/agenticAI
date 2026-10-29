"""Bounded read-only OpenWrt ubus provider. Credentials never enter tool arguments."""
from __future__ import annotations

from services.lua_adapters.http_clients import client as lua_http_client

from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import json
import re
from urllib.parse import urlsplit

import httpx


# Bounds for client listings; a router can know about thousands of hosts.
MAX_DHCP_LEASES = 200
MAX_WIRELESS_CLIENTS = 200
WIRELESS_DEVICE_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}")


@dataclass(frozen=True)
class OpenWrtConfig:
    base_url: str
    username: str
    password: str
    verify_tls: bool = True
    timeout_seconds: float = 10.0
    allow_loopback_http: bool = False

    def origin(self) -> str:
        parsed = urlsplit(self.base_url.strip())
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("OpenWrt URL must be a management origin without credentials or path.")
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback and self.allow_loopback_http):
            raise ValueError("OpenWrt requires HTTPS; explicit loopback HTTP is allowed for the local lab only.")
        return self.base_url.strip().rstrip("/")


class OpenWrtService:
    def __init__(self, config: OpenWrtConfig, client: httpx.Client | None = None):
        self.config = config
        self.origin = config.origin()
        self.client = client or lua_http_client(verify=config.verify_tls, timeout=config.timeout_seconds)
        self.owns_client = client is None
        self.session: str | None = None

    def close(self):
        if self.owns_client:
            self.client.close()

    def _rpc(self, session: str, obj: str, method: str, arguments: dict) -> dict:
        with self.client.stream("POST", self.origin + "/ubus", json={
                "jsonrpc": "2.0", "id": 1, "method": "call",
                "params": [session, obj, method, arguments],
        }) as response:
            response.raise_for_status()
            payload = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                payload.extend(chunk)
                if len(payload) > 1_048_576:
                    raise RuntimeError("OpenWrt response exceeds the bounded response size.")
            body = json.loads(payload)
        result = body.get("result") if isinstance(body, dict) else None
        if not isinstance(body, dict) or body.get("id") != 1 or body.get("error") or not isinstance(result, list) or len(result) != 2 or result[0] != 0 or not isinstance(result[1], dict):
            raise RuntimeError("OpenWrt rejected the read or returned an invalid RPC response.")
        return result[1]

    def _login(self) -> str:
        login = self._rpc("0" * 32, "session", "login", {
            "username": self.config.username, "password": self.config.password,
        })
        session = login.get("ubus_rpc_session")
        if not isinstance(session, str) or not re.fullmatch(r"[0-9a-f]{32}", session):
            raise RuntimeError("OpenWrt login returned no valid session.")
        return session

    @staticmethod
    def _observed(data: dict) -> dict:
        return {"ok": True, "status": "success", "provider": "openwrt",
                "observed_at": datetime.now(timezone.utc).isoformat(), "data": data}

    def _read(self, obj: str, method: str, arguments: dict) -> dict:
        # All callers below are fixed read capabilities. No generic RPC tool exists.
        return self._observed(self._rpc(self._login(), obj, method, arguments))

    def system_info(self) -> dict:
        result = self._read("system", "board", {})
        if not result["data"].get("hostname") or not isinstance(result["data"].get("release"), dict):
            raise RuntimeError("OpenWrt board response is missing identity fields.")
        return result

    def interface_status(self, interface: str) -> dict:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,31}", interface):
            raise ValueError("Invalid logical OpenWrt interface name.")
        return self._read("network.interface." + interface, "status", {})

    def firewall_config(self) -> dict:
        result = self._read("uci", "get", {"config": "firewall"})
        values = result["data"].get("values")
        if not isinstance(values, dict):
            raise RuntimeError("OpenWrt firewall response has no configuration values.")
        # Configuration is observed configuration, not proof of active packet filtering.
        result["scope"] = "configured_firewall_rules; not dataplane verification"
        return result

    def dhcp_leases(self) -> dict:
        data = self._read("luci-rpc", "getDHCPLeases", {})
        raw = data["data"]
        ipv4, ipv6 = raw.get("dhcp_leases", []), raw.get("dhcp6_leases", [])
        if not isinstance(ipv4, list) or not isinstance(ipv6, list):
            raise RuntimeError("OpenWrt lease response has no lease lists.")
        leases = [_lease(item, "ipv4") for item in ipv4 if isinstance(item, dict)]
        leases += [_lease(item, "ipv6") for item in ipv6 if isinstance(item, dict)]
        data["data"] = {"leases": leases[:MAX_DHCP_LEASES], "count": len(leases),
                        "truncated": len(leases) > MAX_DHCP_LEASES}
        # Leases come from the router's own DHCP server only; static addresses and
        # hosts served by another DHCP server are not listed.
        data["scope"] = "dhcp_server_leases"
        return data

    def wireless_clients(self) -> dict:
        session = self._login()
        devices = self._rpc(session, "iwinfo", "devices", {}).get("devices")
        if not isinstance(devices, list):
            raise RuntimeError("OpenWrt wireless device response is invalid.")
        interfaces = []
        for device in devices:
            if not isinstance(device, str) or not WIRELESS_DEVICE_PATTERN.fullmatch(device):
                continue
            stations = self._rpc(session, "iwinfo", "assoclist", {"device": device}).get("results", [])
            clients = [_station(item) for item in stations if isinstance(item, dict)] if isinstance(stations, list) else []
            interfaces.append({"device": device, "clients": clients[:MAX_WIRELESS_CLIENTS],
                               "count": len(clients), "truncated": len(clients) > MAX_WIRELESS_CLIENTS})
        result = self._observed({"interfaces": interfaces,
                                 "count": sum(item["count"] for item in interfaces)})
        if not interfaces:
            result["note"] = "The router reports no wireless interfaces."
        return result


def _lease(item: dict, family: str) -> dict:
    return {
        "family": family,
        "hostname": item.get("hostname") or None,
        "address": item.get("ipaddr") or item.get("ip6addr") or (item.get("ip6addrs") or [None])[0],
        "mac": item.get("macaddr") or None,
        "duid": item.get("duid") or None,
        "expires_in_seconds": item.get("expires") if isinstance(item.get("expires"), int) else None,
    }


def _station(item: dict) -> dict:
    def rate(direction: str) -> int | None:
        value = item.get(direction)
        return value.get("rate") if isinstance(value, dict) and isinstance(value.get("rate"), int) else None

    return {
        "mac": item.get("mac"),
        "signal_dbm": item.get("signal") if isinstance(item.get("signal"), int) else None,
        "noise_dbm": item.get("noise") if isinstance(item.get("noise"), int) else None,
        "inactive_ms": item.get("inactive") if isinstance(item.get("inactive"), int) else None,
        "rx_rate_kbps": rate("rx"),
        "tx_rate_kbps": rate("tx"),
    }


def openwrt_integration_status(settings) -> dict:
    url = settings.openwrt_base_url
    return {"id": "openwrt", "name": "OpenWrt", "mode": "read_only_v1",
            "configured": bool(url and settings.openwrt_username and settings.openwrt_password),
            "host": urlsplit(url).hostname if url else None,
            "verify_tls": settings.openwrt_verify_tls}


def build_openwrt_service(settings) -> OpenWrtService | None:
    if not openwrt_integration_status(settings)["configured"]:
        return None
    return OpenWrtService(OpenWrtConfig(
        settings.openwrt_base_url, settings.openwrt_username, settings.openwrt_password,
        settings.openwrt_verify_tls, settings.openwrt_timeout_seconds,
        settings.openwrt_allow_loopback_http,
    ))
