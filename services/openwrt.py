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
MAX_KNOWN_HOSTS = 200
WIRELESS_DEVICE_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}")
MAC_PATTERN = re.compile(r"[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}")
MIB = 1024 * 1024


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

    def router_health(self) -> dict:
        result = self._read("system", "info", {})
        raw = result["data"]
        memory, root = raw.get("memory"), raw.get("root")
        if (not isinstance(raw.get("uptime"), int) or not _int_list(raw.get("load"), 3)
                or not isinstance(memory, dict) or not _ints(memory, "total", "available")):
            raise RuntimeError("OpenWrt system info is missing uptime, load or memory.")
        data = {
            "uptime_seconds": raw["uptime"],
            # ubus reports load averages scaled by 65536.
            "load_average": [round(value / 65536, 2) for value in raw["load"]],
            "memory_total_mib": round(memory["total"] / MIB, 1),
            "memory_available_mib": round(memory["available"] / MIB, 1),
            "memory_used_percent": _percent(memory["total"] - memory["available"], memory["total"]),
        }
        if isinstance(root, dict) and _ints(root, "total", "used"):
            data["root_filesystem_used_percent"] = _percent(root["used"], root["total"])
        result["data"] = data
        return result

    def interfaces(self) -> dict:
        result = self._read("network.interface", "dump", {})
        rows = result["data"].get("interface")
        if not isinstance(rows, list):
            raise RuntimeError("OpenWrt interface dump has no interface list.")
        interfaces = [_interface(item) for item in rows if isinstance(item, dict) and isinstance(item.get("interface"), str)]
        result["data"] = {"interfaces": interfaces, "count": len(interfaces)}
        return result

    def known_hosts(self) -> dict:
        result = self._read("luci-rpc", "getHostHints", {})
        hints = result["data"]
        if not isinstance(hints, dict):
            raise RuntimeError("OpenWrt host hints are not a host map.")
        hosts = [_host(mac, item) for mac, item in hints.items() if MAC_PATTERN.fullmatch(mac) and isinstance(item, dict)]
        result["data"] = {"hosts": hosts[:MAX_KNOWN_HOSTS], "count": len(hosts), "truncated": len(hosts) > MAX_KNOWN_HOSTS}
        # Names and addresses the router has seen (neighbour table, leases, static
        # entries); a host that has not talked to the router is not listed.
        result["scope"] = "router_neighbour_table_and_leases"
        return result

    def dns_config(self) -> dict:
        result = self._read("uci", "get", {"config": "dhcp", "type": "dnsmasq"})
        sections = result["data"].get("values")
        if not isinstance(sections, dict):
            raise RuntimeError("OpenWrt DNS response has no configuration values.")
        resolvers = [_resolver(item) for item in sections.values() if isinstance(item, dict)]
        result["data"] = {"resolvers": resolvers, "count": len(resolvers)}
        # Saved settings only: this does not show which answers a lookup returns.
        result["scope"] = "configured_dns_settings; not live resolution"
        return result


def _ints(mapping: dict, *keys: str) -> bool:
    return all(isinstance(mapping.get(key), int) and mapping[key] >= 0 for key in keys)


def _int_list(value, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(item, int) for item in value)


def _percent(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 1) if whole else None


def _strings(value) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _addresses(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [f"{item['address']}/{item['mask']}" for item in value
            if isinstance(item, dict) and isinstance(item.get("address"), str) and isinstance(item.get("mask"), int)]


def _route(item: dict) -> str | None:
    if not isinstance(item.get("target"), str) or not isinstance(item.get("mask"), int):
        return None
    target = f"{item['target']}/{item['mask']}"
    return f"{target} via {item['nexthop']}" if isinstance(item.get("nexthop"), str) else target


def _interface(item: dict) -> dict:
    routes = item.get("route") if isinstance(item.get("route"), list) else []
    return {
        "name": item["interface"],
        "up": item.get("up") is True,
        "protocol": item.get("proto") if isinstance(item.get("proto"), str) else None,
        "device": item.get("l3_device") if isinstance(item.get("l3_device"), str) else None,
        "uptime_seconds": item.get("uptime") if isinstance(item.get("uptime"), int) else None,
        "ipv4": _addresses(item.get("ipv4-address")),
        "ipv6": _addresses(item.get("ipv6-address")),
        "routes": [text for text in (_route(route) for route in routes if isinstance(route, dict)) if text],
        "dns_servers": _strings(item.get("dns-server")),
        "dns_search": _strings(item.get("dns-search")),
    }


def _host(mac: str, item: dict) -> dict:
    return {"mac": mac.lower(), "name": item.get("name") if isinstance(item.get("name"), str) else None,
            "ipv4": _strings(item.get("ipaddrs")), "ipv6": _strings(item.get("ip6addrs"))}


UCI_TRUE = {"1", "true", "yes", "on", "enabled"}
UCI_FALSE = {"0", "false", "no", "off", "disabled"}


def _uci_flag(item: dict, option: str) -> bool | None:
    """A UCI boolean option; None when it is not set (the router's default applies)."""
    value = item.get(option)
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    return True if value in UCI_TRUE else False if value in UCI_FALSE else None


def _resolver(item: dict) -> dict:
    servers = item.get("server")
    cache = item.get("cachesize")
    return {
        "domain": item.get("domain") if isinstance(item.get("domain"), str) else None,
        # Without noresolv, dnsmasq also asks the resolvers the internet connection supplies.
        "uses_wan_resolvers": _uci_flag(item, "noresolv") is not True,
        "upstream_servers": [servers] if isinstance(servers, str) else _strings(servers),
        "rebind_protection": _uci_flag(item, "rebind_protection"),
        "dnssec": _uci_flag(item, "dnssec"),
        "drops_unqualified_names": _uci_flag(item, "domainneeded"),
        "drops_private_reverse_lookups": _uci_flag(item, "boguspriv"),
        "answers_local_networks_only": _uci_flag(item, "localservice"),
        "cache_size": int(cache) if isinstance(cache, str) and cache.isdigit() else None,
    }


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
