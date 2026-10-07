from __future__ import annotations


def format_palo_alto_system_info_result(result: dict) -> str:
    hostname = result.get("hostname") or "unknown"
    model = result.get("model") or "unknown"
    version = result.get("sw_version") or "unknown"
    return f"Palo Alto firewall {hostname} is reachable; model={model}, PAN-OS={version}."


PALO_ALTO_TOOLS = {
    "palo_alto_system_info": {
        "description": (
            "Retrieve read-only PAN-OS system information from the configured "
            "Palo Alto firewall or Panorama management endpoint. This does not "
            "modify candidate or running configuration and does not commit changes."
        ),
        "resource_type": "network.firewall",
        "operation_kind": "read",
        "permission": "network.firewall.read",
        "risk": "read",
        "requires_approval": False,
        "required_arguments": [],
        "grounded_arguments": [],
        "derived_arguments": [],
        "parameters": {},
        "condition_fields": ["ok", "status"],
        "result_formatter": format_palo_alto_system_info_result,
    },
}
