import json


def format_network_result(result: dict) -> str:
    if not result.get("ok"):
        return "Network provider read did not succeed."
    text = "Observed OpenWrt data: " + json.dumps(result.get("data"), ensure_ascii=False)
    return text + (" " + result["note"] if result.get("note") else "")


def _duration(seconds: int) -> str:
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    parts = [f"{days} day{'s' if days != 1 else ''}"] if days else []
    parts += [f"{hours} h"] if hours or days else []
    return " ".join(parts + [f"{rest // 60} min"])


def format_router_health(result: dict) -> str:
    if not result.get("ok"):
        return "Network provider read did not succeed."
    data = result["data"]
    text = (f"Router up {_duration(data['uptime_seconds'])}; load "
            + " / ".join(str(value) for value in data["load_average"])
            + f"; memory {data['memory_used_percent']}% used of {data['memory_total_mib']} MiB")
    if data.get("root_filesystem_used_percent") is not None:
        text += f"; storage {data['root_filesystem_used_percent']}% used"
    return text + "."


def format_interfaces(result: dict) -> str:
    if not result.get("ok"):
        return "Network provider read did not succeed."
    lines = [f"{result['data']['count']} interface(s):"]
    for item in result["data"]["interfaces"]:
        how = ", ".join(value for value in (item["protocol"], item["device"]) if value)
        line = f"{item['name']}: {'up' if item['up'] else 'down'}" + (f" ({how})" if how else "")
        addresses = item["ipv4"] + item["ipv6"]
        if addresses:
            line += " " + ", ".join(addresses)
        if item["routes"]:
            line += "; routes " + ", ".join(item["routes"])
        if item["dns_servers"]:
            line += "; DNS " + ", ".join(item["dns_servers"])
        lines.append(line)
    return "\n".join(lines)


def format_known_hosts(result: dict) -> str:
    if not result.get("ok"):
        return "Network provider read did not succeed."
    data = result["data"]
    lines = [f"{data['count']} known host(s)" + (" (list cut to the first 200)" if data["truncated"] else "") + ":"]
    for host in data["hosts"]:
        lines.append(f"{host['name'] or 'unnamed'} ({host['mac']}) " + ", ".join(host["ipv4"] + host["ipv6"]))
    return "\n".join(lines)


def _setting(value: bool | None, on: str, off: str) -> str:
    return "not set" if value is None else on if value else off


def format_dns_config(result: dict) -> str:
    if not result.get("ok"):
        return "Network provider read did not succeed."
    resolvers = result["data"]["resolvers"]
    if not resolvers:
        return "The router has no DNS resolver configured."
    lines = []
    for item in resolvers:
        upstream = (["the internet connection's resolvers"] if item["uses_wan_resolvers"] else []) + item["upstream_servers"]
        text = (f"DNS resolver{' (domain ' + item['domain'] + ')' if item['domain'] else ''}: upstream: "
                + (", ".join(upstream) or "none") + "; rebind protection "
                + _setting(item["rebind_protection"], "on", "off") + "; DNSSEC "
                + _setting(item["dnssec"], "on", "off"))
        if item["cache_size"] is not None:
            text += f"; cache {item['cache_size']} names"
        lines.append(text + ".")
    return "\n".join(lines)


def _tailscale(value: bool | None, on: str, off: str) -> str:
    return on if value else off


def format_tailnet_devices(result: dict) -> str:
    if not result.get("ok"):
        return "Tailscale read did not succeed."
    data = result["data"]
    lines = [f"{data['count']} device(s) on the tailnet" + (" (list cut to the first 200)" if data["truncated"] else "") + ":"]
    for device in data["devices"]:
        about = ", ".join(part for part in (device["os"], device["client_version"] and f"Tailscale {device['client_version']}") if part)
        state = ("online" if device["online"] else
                 "offline since " + device["last_seen"][:10] if device["last_seen"] else "offline")
        if device["key_expiry_disabled"]:
            key = "key never expires"
        elif device["key_expires_in_days"] is None:
            key = "key expiry unknown"
        elif device["key_expires_in_days"] < 0:
            key = f"key expired {-device['key_expires_in_days']} days ago"
        else:
            key = f"key expires in {device['key_expires_in_days']} days"
        parts = [state, key]
        if device["update_available"]:
            parts.append("update available")
        if device["authorized"] is False:
            parts.append("waiting for approval")
        if device["shared_from_another_tailnet"]:
            parts.append("shared from another tailnet")
        if device["tags"]:
            parts.append("tags " + ", ".join(device["tags"]))
        lines.append(f"{device['name']}" + (f" ({about})" if about else "") + ": " + "; ".join(parts))
    return "\n".join(lines)


def format_tailnet_dns(result: dict) -> str:
    if not result.get("ok"):
        return "Tailscale read did not succeed."
    data = result["data"]
    text = f"Tailnet DNS: MagicDNS {_tailscale(data['magic_dns'], 'on', 'off')}; "
    resolvers = ", ".join(f"{row['address']} ({'encrypted, DNS-over-HTTPS' if row['encrypted'] else 'plain DNS'})"
                          for row in data["nameservers"])
    if data["override_local_dns"] and resolvers:
        text += "devices use the tailnet's resolvers instead of the local network's: " + resolvers
    elif resolvers:
        text += "devices use the local network's resolvers, with these as fallback: " + resolvers
    else:
        text += "devices use the local network's resolvers"
    for row in data["split_dns"]:
        text += f"; {row['domain']} via " + (", ".join(row["nameservers"]) or "no resolver")
    if data["search_domains"]:
        text += "; search domains " + ", ".join(data["search_domains"])
    return text + "."


def _read(description: str, resource_type: str, permission: str, formatter) -> dict:
    return {
        "description": description, "resource_type": resource_type, "operation_kind": "read", "effect": "read",
        "permission": permission, "risk": "read", "requires_approval": False, "grounded_arguments": [],
        "parameters": {}, "condition_fields": ["ok", "status"], "result_formatter": formatter,
    }


NETWORKING_TOOLS = {
    "network_system_info": {
        "description": "Read configured OpenWrt router/firewall hostname, hardware and firmware version.",
        "resource_type": "network.device", "operation_kind": "read", "effect": "read",
        "permission": "network.device.read", "risk": "read", "requires_approval": False,
        "grounded_arguments": [], "parameters": {}, "condition_fields": ["ok", "status"],
        "result_formatter": format_network_result,
    },
    "network_interface_status": {
        "description": "Read status and addresses of an explicitly named logical OpenWrt interface. Never infer the interface.",
        "resource_type": "network.interface", "operation_kind": "read", "effect": "read",
        "permission": "network.interface.read", "risk": "read", "requires_approval": False,
        "grounded_arguments": ["interface"],
        "parameters": {"interface": {"type": "str", "description": "Exact logical interface supplied by the user, e.g. lan or wan."}},
        "condition_fields": ["ok", "status"], "result_formatter": format_network_result,
    },
    "network_firewall_config": {
        "description": "Read OpenWrt configured firewall zones and rules without changing, applying or committing configuration. Does not certify active packet filtering.",
        "resource_type": "network.firewall", "operation_kind": "read", "effect": "read",
        "permission": "network.firewall.read", "risk": "read", "requires_approval": False,
        "grounded_arguments": [], "parameters": {}, "condition_fields": ["ok", "status"],
        "result_formatter": format_network_result,
    },
    "network_dhcp_leases": {
        "description": "List the current DHCP leases handed out by the OpenWrt router (hostname, address, MAC, remaining lease time), IPv4 and IPv6. Only clients of the router's own DHCP server; does not discover static hosts or change anything.",
        "resource_type": "network.dhcp_lease", "operation_kind": "read", "effect": "read",
        "permission": "network.dhcp_lease.read", "risk": "read", "requires_approval": False,
        "grounded_arguments": [], "parameters": {}, "condition_fields": ["ok", "status"],
        "result_formatter": format_network_result,
    },
    "network_wireless_clients": {
        "description": "List wireless stations currently associated with each OpenWrt wireless interface (MAC, signal, rates, idle time). Read only; does not disconnect, block or change wireless settings.",
        "resource_type": "network.wireless_client", "operation_kind": "read", "effect": "read",
        "permission": "network.wireless_client.read", "risk": "read", "requires_approval": False,
        "grounded_arguments": [], "parameters": {}, "condition_fields": ["ok", "status"],
        "result_formatter": format_network_result,
    },
    "network_router_health": _read(
        "Read the OpenWrt router's uptime, load average, memory use and storage use. Read only.",
        "network.device.health", "network.device.health.read", format_router_health),
    "network_interfaces": _read(
        "List every logical OpenWrt interface with its state, protocol, addresses, routes and DNS servers. "
        "Use this when the user does not name one interface. Read only.",
        "network.interface", "network.interface.list", format_interfaces),
    "network_known_hosts": _read(
        "List the hosts the OpenWrt router knows by MAC address, with their names and IPv4/IPv6 addresses "
        "(from its neighbour table, leases and static entries). Read only; does not scan the network.",
        "network.host", "network.host.read", format_known_hosts),
    "network_dns_config": _read(
        "Read the OpenWrt router's saved DNS resolver settings: which upstream servers it asks, and whether "
        "rebind protection and DNSSEC are on. Use this to check how the network resolves names. Read only; "
        "it does not perform lookups or change settings.",
        "network.dns", "network.dns.read", format_dns_config),
}

# Tailscale network (tailnet) reads, through the deployment's read-only Tailscale credential.
TAILNET_TOOLS = {
    "network_tailnet_devices": _read(
        "List the devices on the Tailscale network (tailnet) with their operating system, Tailscale version, "
        "whether each is online, when its key expires, pending updates and tags. Use this for remote-access "
        "devices and VPN reachability. Read only; it does not approve, expire or remove devices.",
        "network.tailnet.device", "network.tailnet.device.read", format_tailnet_devices),
    "network_tailnet_dns": _read(
        "Read the Tailscale network's DNS settings: MagicDNS, which resolvers devices use and whether they "
        "are encrypted (DNS-over-HTTPS), split DNS domains and search domains. Use this to check whether "
        "remote devices are protected from DNS tampering. Read only.",
        "network.tailnet.dns", "network.tailnet.dns.read", format_tailnet_dns),
}


def _addresses(values: list[str] | None) -> str:
    return ", ".join(values) if values else "no address (name not found)"


def format_dns_integrity(result: dict) -> str:
    if not result.get("ok"):
        return "DNS check did not succeed."
    data = result["data"]
    rewritten = data["rewritten"]
    head = f"DNS check of {data['names_checked']} name(s) through {data['resolver']}: "
    if not rewritten:
        head += "no rewritten answers."
    else:
        head += f"plain DNS answers were rewritten for {len(rewritten)} ({', '.join(rewritten)})."
        if data["system_resolver_protected"]:
            head += " This host's own resolver gave the true answers, so the services here are not affected."
        elif data["system_follows_rewrite"]:
            head += (" This host's own resolver also returned rewritten answers; switch it to encrypted DNS "
                     "(DNS-over-HTTPS) or a tailnet resolver.")
    lines = [head]
    for row in data["names"]:
        verdict = row["verdict"]
        if verdict == "consistent":
            line = f"{row['name']}: consistent ({_addresses(row['encrypted'])})"
            if row["system_resolver"] == "different_answer":
                line += (f"; this host's resolver gives {_addresses(row['system'])} "
                         "(another resolver's view, as for geo-routed names)")
            lines.append(line)
        elif verdict == "unknown":
            lines.append(f"{row['name']}: could not be checked ({'; '.join(row['errors']) or 'no answer'})")
        else:
            lines.append(f"{row['name']}: plain DNS rewritten to {_addresses(row['plain'])}; DNS-over-HTTPS gives "
                         f"{_addresses(row['encrypted'])}; this host's resolver gives {_addresses(row['system'])}")
    return "\n".join(lines)


# Checks how names resolve from the AI host itself; independent of the router and the tailnet.
DNS_CHECK_TOOLS = {
    "network_dns_integrity_check": {
        **_read(
            "Check whether DNS answers are being rewritten (ISP DNS hijacking or interception): ask each name by "
            "plain DNS, by this host's own resolver and by encrypted DNS-over-HTTPS, and compare the addresses. "
            "Use this when sites resolve to the wrong place or a provider may be tampering with DNS. Read only.",
            "network.dns.integrity", "network.dns.integrity.read", format_dns_integrity),
        "grounded_arguments": ["names"],
        "parameters": {"names": {
            "type": "list[str]",
            "description": "Optional host names the user named, e.g. vimeo.com; at most 10. "
                           "Omit them to check the configured reference names."}},
    },
}
# The networking specialist's reads; readiness keeps OpenWrt, Tailscale and the DNS check as separate providers.
OPENWRT_TOOLS = dict(NETWORKING_TOOLS)
NETWORKING_TOOLS.update(TAILNET_TOOLS)
NETWORKING_TOOLS.update(DNS_CHECK_TOOLS)
