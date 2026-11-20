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
    if not result["data"]["interfaces"]:
        return "The router reports no logical interfaces."
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
    if not data["hosts"]:
        return "The router knows no hosts."
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


def _reachability(connectivity: dict | None) -> list[str]:
    """How the device reaches peers, from its last report. Nothing is said when it sent none."""
    if not connectivity:
        return []
    relay = []
    if connectivity["nearest_relay"]:
        latency = connectivity["relay_latency_ms"]
        relay = [f"nearest relay {connectivity['nearest_relay']}" + (f" {latency} ms" if latency is not None else "")]
    degraded = (["UDP blocked, so traffic goes through a relay"] if connectivity["udp"] is False else []) + (
        ["hard NAT (more connections relayed)"] if connectivity["hard_nat"] else [])
    if degraded:
        return degraded + relay
    if connectivity["udp"]:
        return [", ".join(["direct connections possible"] + relay)]
    return relay


def format_tailnet_devices(result: dict) -> str:
    if not result.get("ok"):
        return "Tailscale read did not succeed."
    data = result["data"]
    if not data["devices"]:
        return "No device is on the tailnet."
    head = f"{data['count']} device(s) on the tailnet" + (" (list cut to the first 200)" if data["truncated"] else "")
    summary = data.get("summary")
    if summary:
        attention = summary["with_issues"]
        head += (f": {summary['online']} online, {summary['offline']} offline; "
                 + (f"{attention} need{'s' if attention == 1 else ''} attention." if attention else "none need attention."))
    else:
        head += ":"
    lines = [head]
    for device in data["devices"]:
        about = ", ".join(part for part in (device["os"], device["client_version"] and f"Tailscale {device['client_version']}") if part)
        state = ("online" if device["online"] else
                 "offline since " + device["last_seen"][:10] if device["last_seen"] else "offline")
        days = device["key_expires_in_days"]
        if device["key_expiry_disabled"]:
            key = "key never expires"
        elif days is None:
            key = "key expiry unknown"
        elif days < 0:
            key = f"key expired {-days} days ago"
        elif "key_expired" in device.get("issues", ()):
            # Lapsed within the last day: the whole-day count is still 0.
            key = "key expired"
        else:
            key = f"key expires in {days} days"
        parts = [state, key]
        if device["authorized"] is False:
            parts.append("waiting for approval")
        if device.get("tailnet_lock_error"):
            parts.append("tailnet lock reports a problem")
        if device.get("routes_waiting_for_approval"):
            parts.append(f"{device['routes_waiting_for_approval']} of {device['routes_advertised']} advertised routes "
                         "waiting for approval")
        if device.get("exit_node_offered"):
            parts.append("offers itself as an exit node")
        if device.get("blocks_incoming"):
            parts.append("blocks incoming connections")
        if device["online"]:
            parts += _reachability(device.get("connectivity"))
        if device["update_available"]:
            parts.append("update available")
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


def _endpoint(host: str, port: str) -> str:
    return host if port == "*" else f"{host}:{port}"


# Tailscale policy words a reading may use, defined once so neither a reader nor a model has to guess them
# (live, "check autogroup:member -> autogroup:self" was retold without check mode's fresh sign-in).
TAILNET_POLICY_TERMS = {
    "accept": "SSH without a fresh sign-in",
    "check": "SSH asks for a fresh sign-in unless the user signed in recently",
    "*": "any user, device or address",
    "autogroup:member": "every member of the tailnet",
    "autogroup:tagged": "every tagged device",
    "autogroup:self": "the user's own devices",
    "autogroup:nonroot": "any local user except root",
    "autogroup:admin": "the tailnet's admins",
    "autogroup:owner": "the tailnet's owner",
    "autogroup:internet": "the internet through an exit node",
    "autogroup:danger-all": "any source, including devices outside the tailnet",
    "a named user": "a local account other than root, name not shown",
}


def _policy_terms(data: dict) -> str | None:
    used = []
    for rule in data["rules"]:
        used += rule["sources"] + rule["destinations"]
    for rule in data["ssh"]:
        used += [rule["action"], *rule["sources"], *rule["destinations"], *rule["users"]]
    for row in data["tag_owners"]:
        used += row["owners"]
    defined = [term for term in dict.fromkeys(used) if term in TAILNET_POLICY_TERMS]
    if not defined:
        return None
    return "Words: " + "; ".join(f"{term} = {TAILNET_POLICY_TERMS[term]}" for term in defined) + "."


def format_tailnet_policy(result: dict) -> str:
    if not result.get("ok"):
        return "Tailscale read did not succeed."
    data = result["data"]
    summary = data["summary"]
    managed = data["externally_managed"]
    head = (f"Tailnet policy: {summary['rules']} access rule(s), {summary['ssh_rules']} SSH rule(s), "
            f"{summary['tags']} tag(s), {summary['groups']} group(s), {data['hosts']} host alias(es), "
            f"{data['tests']} policy test(s); "
            + ("managed outside the admin console (GitOps)." if managed else
               "edited in the Tailscale admin console." if managed is False else "where it is edited is unknown."))
    lines = [head]
    if summary["open_to_all_members"]:
        lines.append("Every member can reach every device on every port (a rule allows autogroup:member to * on "
                     "all ports).")
    if data["rules"]:
        parts = []
        for rule in data["rules"]:
            if rule["kind"] == "acl":
                targets = ", ".join(_endpoint(host, port) for host, port in zip(rule["destinations"], rule["ports"]))
            else:
                targets = ", ".join(rule["destinations"]) + (
                    " (all ports)" if rule["ports"] == ["*"] else " (" + ", ".join(rule["ports"]) + ")"
                    if rule["ports"] else "")
                if rule["app_capabilities"]:
                    targets += f", {rule['app_capabilities']} app capability(ies)"
            parts.append(f"{rule['kind']} {', '.join(rule['sources'])} -> {targets}")
        lines.append("Access: " + "; ".join(parts) + ("; list cut" if data["truncated"] else "") + ".")
    else:
        lines.append("Access: no rules, so no device can reach another.")
    if data["ssh"]:
        lines.append("SSH: " + "; ".join(f"{rule['action']} {', '.join(rule['sources'])} -> "
                                         f"{', '.join(rule['destinations'])} as {', '.join(rule['users'])}"
                                         for rule in data["ssh"]) + ".")
    if data["tag_owners"]:
        lines.append("Tag owners: " + "; ".join(f"{row['tag']} by {', '.join(row['owners']) or 'nobody'}"
                                                for row in data["tag_owners"]) + ".")
    if data["groups"]:
        lines.append("Groups: " + "; ".join(f"{row['name']} ({row['members']} members)" for row in data["groups"]) + ".")
    if data["other_sections"]:
        lines.append("Also set: " + ", ".join(data["other_sections"]) + ".")
    terms = _policy_terms(data)
    if terms:
        lines.append(terms)
    lines.append('Users, host aliases and addresses are shown as "a user", "a host alias" and "an address".')
    return "\n".join(lines)


def format_tailnet_posture(result: dict) -> str:
    if not result.get("ok"):
        return "Tailscale read did not succeed."
    data = result["data"]
    if not data["devices"]:
        return "No device is on the tailnet."
    summary = data["summary"]
    lines = [f"Posture of {data['count']} tailnet device(s)"
             + (f" (first {len(data['devices'])} read)" if data["truncated"] else "")
             + f": {summary['auto_update_off']} with Tailscale auto-update off, "
             f"{summary['state_not_encrypted']} storing Tailscale state unencrypted."]
    for row in data["devices"]:
        if row["other_attributes"] is None:
            lines.append(f"{row['name']}: posture could not be read.")
            continue
        system = " ".join(part for part in (row["os"], row["os_version"]) if part) or "OS unknown"
        version = f"Tailscale {row['tailscale_version']}" if row["tailscale_version"] else "Tailscale version unknown"
        if row["release_track"]:
            version += f" ({row['release_track']})"
        parts = [f"{system}, {version}",
                 {True: "auto-update on", False: "auto-update off", None: "auto-update unknown"}[row["auto_update"]],
                 {True: "Tailscale state stored encrypted", False: "Tailscale state stored unencrypted",
                  None: "state encryption unknown"}[row["state_encrypted"]]]
        if row["other_attributes"]:
            parts.append(f"{row['other_attributes']} other attribute(s)")
        lines.append(f"{row['name']}: " + "; ".join(parts) + ".")
    return "\n".join(lines)


def _online_word(value) -> str:
    return {True: "online", False: "offline"}.get(value, "online state unknown")


def _coverage_line(label: str, row: dict) -> str:
    devices = row["tailnet_devices"]
    if devices:
        text = ("tailnet device " if len(devices) == 1 else "tailnet devices ") + ", ".join(
            f"{device['name']} ({_online_word(device['online'])})" for device in devices)
    else:
        text = "no tailnet device named like it"
    if "device_offline" in row["issues"]:
        text += "; it runs, but its tailnet device is offline"
    return f"{label}: {text}."


def format_tailnet_coverage(result: dict) -> str:
    if not result.get("ok"):
        return "Tailnet coverage read did not succeed."
    data = result["data"]
    summary = data["summary"]
    if not data["nodes"] and not data["guests"]:
        return "Proxmox lists no nodes or guests, so there is nothing to match with the tailnet."
    head = (f"Tailnet coverage, matched by name: {summary['nodes_with_device']} of {summary['nodes']} Proxmox "
            f"node(s) and {summary['guests_with_device']} of {summary['guests']} guest(s) have a tailnet device "
            "named like them.")
    if summary["device_offline"]:
        head += f" {summary['device_offline']} running machine(s) have their tailnet device offline."
    if not data["complete"]:
        head += " A listing was cut, so no machine is called missing from the tailnet."
    lines = [head]
    lines += [_coverage_line(f"{row['node']} (node, {row['status']})", row) for row in data["nodes"]]
    lines += [_coverage_line(f"{row['name'] or 'unnamed'} ({'VM' if row['type'] == 'qemu' else 'LXC'} "
                             f"{row['vmid']} on {row['node']}, {row['status']})", row) for row in data["guests"]]
    if data["devices_outside_proxmox"]:
        lines.append("Tailnet devices named like no Proxmox node or guest: "
                     + ", ".join(data["devices_outside_proxmox"]) + ".")
    lines.append("A machine may be on the tailnet under another name; names are the only link Janus reads.")
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
    "network_dns_config": _read(
        "Read the OpenWrt router's saved DNS resolver settings: which upstream servers it asks, and whether "
        "rebind protection and DNSSEC are on. Use this to check how the network resolves names. Read only; "
        "it does not perform lookups or change settings.",
        "network.dns", "network.dns.read", format_dns_config),
}

# Tailscale network (tailnet) reads, through the deployment's own trust credential (read scopes plus
# policy_file, which these reads use only to read the policy).
TAILNET_TOOLS = {
    "network_tailnet_devices": _read(
        "List the devices on the Tailscale network (tailnet) with their operating system, Tailscale version, "
        "whether each is online, when its key expires, pending updates and tags, and each device's health: "
        "approval, routes waiting for approval, blocked incoming connections, tailnet lock problems and how it "
        "reaches peers (UDP, NAT, nearest relay). Use this for remote-access devices, VPN reachability and "
        "tailnet health. Read only; it does not approve, expire or remove devices.",
        "network.tailnet.device", "network.tailnet.device.read", format_tailnet_devices),
    "network_tailnet_dns": _read(
        "Read the Tailscale network's DNS settings: MagicDNS, which resolvers devices use and whether they "
        "are encrypted (DNS-over-HTTPS), split DNS domains and search domains. Use this to check whether "
        "remote devices are protected from DNS tampering. Read only.",
        "network.tailnet.dns", "network.tailnet.dns.read", format_tailnet_dns),
    "network_tailnet_policy": {
        **_read(
            "Read the Tailscale network's access policy (ACL): who may reach which devices and ports (grants and "
            "ACL rules), SSH rules, tag owners, group sizes, host aliases and whether the policy file is managed "
            "outside the admin console. Users, host aliases and addresses are masked. Use this for tailnet "
            "policy questions (access control, whether every member can reach every device). Read only; it "
            "never changes the policy.",
            "network.tailnet.policy", "network.tailnet.policy.read", format_tailnet_policy),
        "synthesis_evidence": "answer",
    },
    "network_tailnet_posture": {
        **_read(
            "Read each tailnet device's posture attributes: operating system and version, Tailscale version and "
            "release track, whether Tailscale auto-update is on and whether the device stores its Tailscale "
            "state encrypted. Use this for device posture and compliance questions; for online state, key "
            "expiry and reachability use network_tailnet_devices. Read only.",
            "network.tailnet.posture", "network.tailnet.posture.read", format_tailnet_posture),
        "synthesis_evidence": "answer",
    },
    "network_tailnet_coverage": {
        **_read(
            "Check which Proxmox nodes and guests (VMs and containers) have a device on the Tailscale network, "
            "matched by name, which running machines have their tailnet device offline, and which tailnet devices "
            "are named like no Proxmox machine. Use this for tailnet coverage of the Proxmox estate; for device "
            "health use network_tailnet_devices. Read only.",
            "network.tailnet.coverage", "network.tailnet.coverage.read", format_tailnet_coverage),
        "synthesis_evidence": "answer",
    },
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


# Governed tailnet policy changes (direction note D2, owner decision 5): one typed grant added or
# removed, never model-written policy text. Trusted policy validates and freezes the change first.
GRANT_TRUSTED = ["expected_etag", "policy_sha256", "new_policy_sha256", "grant"]
GRANT_TAIL = ("Tailscale validated the changed policy before this approval; Janus writes it only if the policy "
              "is still the version it read (a concurrent edit is refused, never overwritten) and reads the "
              "policy back.")
GRANT_PARAMETERS = {
    "sources": {"type": "list[str]", "description": "Named tag:, group: or autogroup: selectors."},
    "destinations": {"type": "list[str]", "description": "Named tag:, group:, autogroup: selectors or *."},
    "ports": {"type": "list[str]", "description": "Named ports: *, 22, 8000-8100 or tcp:443."},
}


def _policy_tailnet():
    """The proposing account's Tailscale (its own link, else the machine default) for trusted
    pre-approval reads, or None."""
    from config import Settings
    from services.provider_links import owner_tailnet_service

    return owner_tailnet_service(Settings())


def _grant_policy(operation: str):
    def evaluate(sources: list[str], destinations: list[str], ports: list[str]) -> dict:
        """Read-only: make the typed edit, have Tailscale validate it and freeze the version and digests."""
        from services.tailnet_policy_change import prepare_grant_change
        from services.tailnet_policy_edit import GrantSpec
        from tools.provider_reads import provider_read

        risk = "high" if operation == "remove" else "medium"

        def refused(status: str, error: str) -> dict:
            return {"ok": False, "status": status, "risk": risk, "requires_approval": True, "error": error}

        service = _policy_tailnet()
        if service is None:
            return refused("error", "The Tailscale network is not connected for this account. Add a Tailscale "
                                    "link in Connections.")
        try:
            spec = GrantSpec(tuple(sources or ()), tuple(destinations or ()), tuple(ports or ()))
            outcome = provider_read("tailscale", "The Tailscale network",
                                    lambda: {"ok": True, "data": prepare_grant_change(service, operation, spec)})
        finally:
            service.close()
        if not outcome.get("ok"):
            return refused("error", outcome.get("error") or "The Tailscale network could not check the change.")
        prepared = outcome["data"]
        if not prepared["ok"]:
            return refused(prepared["status"], prepared["error"])
        return {"ok": True, "status": "ready", "risk": risk, "requires_approval": True, "error": None,
                "execution_arguments": {"sources": sources, "destinations": destinations, "ports": ports,
                                        "grant": prepared["grant"], **prepared["frozen"]}}

    return evaluate


def _grant_approval(verb: str, consequence: str):
    def format_approval(arguments: dict) -> str:
        return f"{verb} the tailnet grant {arguments.get('grant')}. {consequence}{GRANT_TAIL}"
    return format_approval


def _grant_result(done: str):
    def format_result(result: dict) -> str:
        grant = result.get("grant") or "the grant"
        if result.get("ok") and result.get("verification_ok"):
            return f"{done} {grant}. Janus read the policy back and confirmed it."
        if result.get("status") == "outcome_unknown":
            return (f"The tailnet policy change for {grant} has an unknown outcome: {result.get('error')} "
                    "Do not repeat it before reading the policy.")
        return f"The tailnet policy change for {grant} did not happen: {result.get('error')}"
    return format_result


def _grant_tool(operation: str, description: str, approval, done: str) -> dict:
    tool = {
        "description": description, "resource_type": "network.tailnet.policy",
        "operation_kind": "delete" if operation == "remove" else "create",
        "effect": "mutation", "permission": f"network.tailnet.policy.grant.{operation}",
        "risk": "high" if operation == "remove" else "medium", "requires_approval": True,
        "policy_owns_preconditions": True, "policy_resolver": _grant_policy(operation),
        "trusted_policy_arguments": list(GRANT_TRUSTED),
        "grounded_arguments": ["sources", "destinations", "ports"], "derived_arguments": [],
        "required_arguments": ["sources", "destinations", "ports"], "parameters": GRANT_PARAMETERS,
        "condition_fields": ["ok", "status", "verification_ok"],
        "approval_formatter": approval, "result_formatter": _grant_result(done),
    }
    if operation == "remove":
        # Removing access is destructive: the owner types the grant exactly as the approval shows it.
        tool["confirmation_argument"] = "grant"
    return tool


TAILNET_CHANGE_TOOLS = {
    "network_tailnet_grant_add": _grant_tool(
        "add",
        "Add one tailnet access grant: named sources may reach named destinations on named ports. Approved "
        "first; never users or addresses.",
        _grant_approval("Add", "Devices it covers can start these connections at once. "), "Added the tailnet grant"),
    "network_tailnet_grant_remove": _grant_tool(
        "remove",
        "Remove one existing tailnet access grant, named by its sources, destinations and ports. Approved "
        "with the grant typed.",
        _grant_approval("Remove", "Connections that only this grant allowed stop at once and devices lose that "
                                  "access. "),
        "Removed the tailnet grant"),
}


# Governed tailnet device, DNS and key changes (owner decision 20 Nov 2026, ~14:10 WIB, item 1: Janus's
# client may write devices:core, devices:routes and dns). The trusted policy reads the target before
# approval and freezes its identity and the value being replaced; the tool re-reads, sends once and
# reads back (services/tailnet_changes.py).
DEVICE_TRUSTED = ["setting", "device_id", "device_name", "device_tags", "previous"]
DEVICE_SETTING_ARGUMENTS = {"routes": "routes", "authorized": "authorized", "tags": "tags",
                            "key_expiry_disabled": "key_expiry", "new_name": "name"}
DNS_SETTING_ARGUMENTS = {"nameservers": "nameservers", "magic_dns": "magic_dns", "search_paths": "search_paths",
                         "split_dns_domain": "split_dns"}
CHANGE_CHECK = ("Janus refuses the change if the target changed since this approval, sends it once and reads it "
                "back.")


def _tailnet_change_policy(prepare, risk_of):
    def evaluate(**arguments) -> dict:
        from tools.provider_reads import provider_read

        arguments = {name: value for name, value in arguments.items() if value is not None}
        risk = risk_of(arguments)

        def refused(status: str, error: str) -> dict:
            return {"ok": False, "status": status, "risk": risk, "requires_approval": True, "error": error}

        service = _policy_tailnet()
        if service is None:
            return refused("error", "The Tailscale network is not connected for this account. Add a Tailscale "
                                    "link in Connections.")
        try:
            outcome = provider_read("tailscale", "The Tailscale network",
                                    lambda: {"ok": True, "data": prepare(service, arguments)})
        finally:
            service.close()
        if not outcome.get("ok"):
            return refused("error", outcome.get("error") or "The Tailscale network could not check the change.")
        prepared = outcome["data"]
        if not prepared["ok"]:
            return refused(prepared["status"], prepared["error"])
        return {"ok": True, "status": "ready", "risk": risk, "requires_approval": True, "error": None,
                "execution_arguments": {**arguments, **prepared["frozen"]}}

    return evaluate


def _one_setting(arguments: dict, names: dict) -> str:
    given = [name for name in names if name in arguments]
    if len(given) != 1:
        raise ValueError("Change one setting per approval: " + ", ".join(names) + ".")
    return given[0]


def _device_change(arguments: dict):
    from services.tailnet_changes import DeviceChange

    device = str(arguments.get("device") or "")
    if arguments.get("remove"):
        return DeviceChange(device, "remove", {})
    name = _one_setting(arguments, DEVICE_SETTING_ARGUMENTS)
    setting = DEVICE_SETTING_ARGUMENTS[name]
    key = {"key_expiry": "key_expiry_disabled", "name": "name"}.get(setting, setting)
    return DeviceChange(device, setting, {key: arguments[name]})


def _prepare_device(service, arguments: dict) -> dict:
    from services.tailnet_changes import TailnetChangeRefused, prepare_device_change

    try:
        change = _device_change(arguments)
    except (ValueError, TailnetChangeRefused) as error:
        return {"ok": False, "status": "denied", "error": str(error)}
    prepared = prepare_device_change(service, change)
    if prepared["ok"] and change.setting != "remove":
        prepared["frozen"] = {"setting": change.setting, **prepared["frozen"]}
    return prepared


def _prepare_dns(service, arguments: dict) -> dict:
    from services.tailnet_changes import prepare_dns_change

    try:
        name = _one_setting(arguments, DNS_SETTING_ARGUMENTS)
    except ValueError as error:
        return {"ok": False, "status": "denied", "error": str(error)}
    setting = DNS_SETTING_ARGUMENTS[name]
    value = ({"domain": arguments[name], "nameservers": arguments.get("split_dns_nameservers")}
             if setting == "split_dns" else {name: arguments[name]})
    prepared = prepare_dns_change(service, setting, value)
    if prepared["ok"]:
        prepared["frozen"] = {"setting": setting, **prepared["frozen"]}
    return prepared


def _prepare_key(service, arguments: dict) -> dict:
    from services.tailnet_changes import prepare_key_revoke

    return prepare_key_revoke(service, str(arguments.get("key_id") or ""))


def _values(value) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{key.replace('_', ' ')} {_values(item)}" for key, item in value.items()) or "nothing"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "none"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "none" if value is None else str(value)


def _device_set_approval(arguments: dict) -> str:
    setting = arguments.get("setting") or "setting"
    wanted = next((arguments[name] for name in DEVICE_SETTING_ARGUMENTS if name in arguments), None)
    text = (f"Set the {setting.replace('_', ' ')} of tailnet device {arguments.get('device_name') or arguments.get('device')} "
            f"to {_values(wanted)} (now {_values(arguments.get('previous'))}). ")
    if setting == "routes":
        text += "Every tailnet member the policy allows can then reach those subnets through this device. "
    if setting == "authorized" and wanted is False:
        text += "The device loses tailnet access at once. "
    return text + CHANGE_CHECK


def _device_remove_approval(arguments: dict) -> str:
    return (f"Remove tailnet device {arguments.get('device_name')} (tags {_values(arguments.get('device_tags'))}) "
            "from the tailnet. It loses access at once and must enrol again with a new key. " + CHANGE_CHECK)


def _dns_approval(arguments: dict) -> str:
    setting = arguments.get("setting") or "dns"
    wanted = (f"{arguments.get('split_dns_domain')} -> {_values(arguments.get('split_dns_nameservers'))}"
              if setting == "split_dns" else _values(next((arguments[name] for name in DNS_SETTING_ARGUMENTS
                                                           if name in arguments), None)))
    return (f"Set the tailnet DNS {setting.replace('_', ' ')} to {wanted} (now {_values(arguments.get('previous'))}). "
            "Every device that uses tailnet DNS follows it at once. " + CHANGE_CHECK)


def _key_approval(arguments: dict) -> str:
    return (f"Revoke tailnet key {arguments.get('key_id')} ({arguments.get('key_type')} key "
            f"'{arguments.get('key_description')}', tags {_values(arguments.get('key_tags'))}). A device not yet "
            "enrolled with it can no longer join. " + CHANGE_CHECK)


def _tailnet_change_result(subject: str):
    def format_result(result: dict) -> str:
        target = result.get("device") or result.get("key") or "the tailnet"
        if result.get("ok") and result.get("verification_ok"):
            observed = result.get("observed")
            return (f"{subject} for {target} done; Janus read it back"
                    + (f": {_values(observed)}." if observed else "."))
        if result.get("status") == "outcome_unknown":
            return (f"The {subject.lower()} for {target} has an unknown outcome: {result.get('error')} "
                    "Do not repeat it before reading the tailnet.")
        return f"The {subject.lower()} for {target} did not happen: {result.get('error')}"
    return format_result


def _tailnet_change_tool(*, description: str, resource: str, operation: str, permission: str, risk: str,
                         policy, trusted: list[str], grounded: list[str], required: list[str], parameters: dict,
                         approval, subject: str, confirmation: str | None = None) -> dict:
    tool = {
        "description": description, "resource_type": resource, "operation_kind": operation,
        "effect": "mutation", "permission": permission, "risk": risk, "requires_approval": True,
        "policy_owns_preconditions": True, "policy_resolver": policy, "trusted_policy_arguments": trusted,
        "grounded_arguments": grounded, "derived_arguments": [], "required_arguments": required,
        "parameters": parameters, "condition_fields": ["ok", "status", "verification_ok"],
        "approval_formatter": approval, "result_formatter": _tailnet_change_result(subject),
    }
    if confirmation:
        tool["confirmation_argument"] = confirmation
    return tool


DEVICE_PARAMETER = {"type": "str", "description": "Exact tailnet device name, as listed by network_tailnet_devices."}


def _device_set_policy(device: str, routes: list[str] | None = None, authorized: bool | None = None,
                       tags: list[str] | None = None, key_expiry_disabled: bool | None = None,
                       new_name: str | None = None):
    return _tailnet_change_policy(_prepare_device, lambda a: "high" if a.get("authorized") is False else "medium")(
        device=device, routes=routes, authorized=authorized, tags=tags, key_expiry_disabled=key_expiry_disabled,
        new_name=new_name)


def _device_remove_policy(device: str):
    return _tailnet_change_policy(_prepare_device, lambda _a: "high")(device=device, remove=True)


def _dns_set_policy(nameservers: list[str] | None = None, magic_dns: bool | None = None,
                    search_paths: list[str] | None = None, split_dns_domain: str | None = None,
                    split_dns_nameservers: list[str] | None = None):
    return _tailnet_change_policy(_prepare_dns, lambda _a: "high")(
        nameservers=nameservers, magic_dns=magic_dns, search_paths=search_paths,
        split_dns_domain=split_dns_domain, split_dns_nameservers=split_dns_nameservers)


def _key_revoke_policy(key_id: str):
    return _tailnet_change_policy(_prepare_key, lambda _a: "high")(key_id=key_id)


def _remove_policy_arguments(policy):
    """The removal policy marks its request; the mark is not an execution argument."""
    def evaluate(**arguments):
        result = policy(**arguments)
        if result.get("ok"):
            result["execution_arguments"].pop("remove", None)
        return result
    return evaluate


TAILNET_CHANGE_TOOLS.update({
    "network_tailnet_device_set": _tailnet_change_tool(
        description="Change one setting of one tailnet device: approve its advertised subnet routes, authorize "
                    "or deauthorize it, set its tags, its key expiry or its name. Approved first.",
        resource="network.tailnet.device", operation="update", permission="network.tailnet.device.update",
        risk="medium", policy=_device_set_policy, trusted=DEVICE_TRUSTED,
        grounded=["device", "routes", "tags", "new_name"], required=["device"],
        parameters={"device": DEVICE_PARAMETER,
                    "routes": {"type": "list[str]", "description": "Advertised subnet routes to enable, e.g. "
                                                                   "10.77.0.0/24."},
                    "authorized": {"type": "bool", "description": "Authorize (true) or deauthorize (false)."},
                    "tags": {"type": "list[str]", "description": "Tags to set, e.g. tag:janus-lab."},
                    "key_expiry_disabled": {"type": "bool", "description": "True: the key never expires."},
                    "new_name": {"type": "str", "description": "New device name."}},
        approval=_device_set_approval, subject="The device change"),
    "network_tailnet_device_remove": _tailnet_change_tool(
        description="Remove one device from the tailnet; it must enrol again to rejoin. Approved with the "
                    "device name typed.",
        resource="network.tailnet.device", operation="delete", permission="network.tailnet.device.remove",
        risk="high", policy=_remove_policy_arguments(_device_remove_policy),
        trusted=["device_id", "device_name", "device_tags", "previous"], grounded=["device"], required=["device"],
        parameters={"device": DEVICE_PARAMETER}, approval=_device_remove_approval,
        subject="The device removal", confirmation="device_name"),
    "network_tailnet_dns_set": _tailnet_change_tool(
        description="Change one tailnet DNS setting: the global nameservers, MagicDNS, the search domains or "
                    "one split-DNS domain. Approved first.",
        resource="network.tailnet.dns", operation="update", permission="network.tailnet.dns.update", risk="high",
        policy=_dns_set_policy, trusted=["setting", "previous"],
        grounded=["nameservers", "search_paths", "split_dns_domain", "split_dns_nameservers"], required=[],
        parameters={"nameservers": {"type": "list[str]", "description": "Nameserver addresses."},
                    "magic_dns": {"type": "bool", "description": "MagicDNS on (true) or off (false)."},
                    "search_paths": {"type": "list[str]", "description": "Search domains."},
                    "split_dns_domain": {"type": "str", "description": "One split-DNS domain."},
                    "split_dns_nameservers": {"type": "list[str]", "description": "That domain's nameservers; "
                                                                                  "omit to remove it."}},
        approval=_dns_approval, subject="The DNS change"),
    "network_tailnet_key_revoke": _tailnet_change_tool(
        description="Revoke one tailnet auth key by its id; never Janus's own client. Approved with the key id "
                    "typed.",
        resource="network.tailnet.key", operation="delete", permission="network.tailnet.key.revoke", risk="high",
        policy=_key_revoke_policy, trusted=["key_type", "key_description", "key_tags"], grounded=["key_id"],
        required=["key_id"], parameters={"key_id": {"type": "str", "description": "Exact key id."}},
        approval=_key_approval, subject="The key revocation", confirmation="key_id"),
})

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
NETWORKING_TOOLS.update(TAILNET_CHANGE_TOOLS)
NETWORKING_TOOLS.update(DNS_CHECK_TOOLS)
