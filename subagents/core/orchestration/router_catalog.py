"""Deduplicate registry metadata for routing without selecting or hiding tools."""
from copy import deepcopy
import json


def compact_router_catalog(specialists: list[dict]) -> dict:
    agents = []
    capabilities = {}
    for specialist in specialists:
        agent = {k: deepcopy(v) for k, v in specialist.items() if k != "capabilities"}
        agent["capabilities"] = []
        for original in specialist["capabilities"]:
            name = original["name"]
            capability = deepcopy(original)
            metadata = capability.get("intent_metadata", {})
            # These root fields are exact duplicates, not omitted constraints.
            for key, value in metadata.items():
                if key in capability and capability[key] == value:
                    del capability[key]
            if name in capabilities and capabilities[name] != capability:
                raise ValueError(f"Conflicting trusted capability definitions: {name}")
            capabilities[name] = capability
            agent["capabilities"].append(name)
        agents.append(agent)
    return {"specialists": agents, "capabilities": capabilities}


def render_router_catalog(specialists: list[dict]) -> str:
    return (
        "Catalog format: specialists[].capabilities lists exact names in the shared "
        "capabilities dictionary. A specialist may only receive capabilities in its own list. "
        "Shared semantic fields occur once in intent_metadata; they retain their full meaning.\n"
        + json.dumps(compact_router_catalog(specialists), ensure_ascii=False, separators=(",", ":"))
    )
