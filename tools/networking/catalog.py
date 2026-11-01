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
}
