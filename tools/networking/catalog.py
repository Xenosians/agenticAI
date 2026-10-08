import json


def format_network_result(result: dict) -> str:
    if not result.get("ok"):
        return "Network provider read did not succeed."
    return "Observed OpenWrt data: " + json.dumps(result.get("data"), ensure_ascii=False)


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
}
