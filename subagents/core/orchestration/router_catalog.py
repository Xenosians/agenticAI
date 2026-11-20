"""Deduplicate registry metadata for routing without selecting or hiding tools."""
from copy import deepcopy
import json

# The intent_metadata values most capabilities share. The compact catalog may
# state them once (intent_metadata_defaults) so that a capability lists only
# the fields that differ; an absent field then means its default. A capability
# must carry every defaulted field, so absence never gives it a value it lacks.
METADATA_DEFAULTS = {
    "effect": "read",
    "grounded_arguments": [],
    "required_arguments": [],
    "derived_arguments": [],
    "condition_fields": [],
    "requires_approval": False,
    "policy_owns_preconditions": False,
}
DESCRIPTIONS = "grounded_argument_descriptions"


def _without_defaults(name: str, metadata: dict) -> dict:
    missing = sorted(set(METADATA_DEFAULTS) - set(metadata))
    if missing:
        raise ValueError(f"Capability {name} lacks intent_metadata fields that have a catalog default: "
                         + ", ".join(missing))
    return {key: value for key, value in metadata.items()
            if key not in METADATA_DEFAULTS or value != METADATA_DEFAULTS[key]}


def _shared_descriptions(capabilities: dict) -> dict:
    """One wording per argument name that several capabilities describe identically.

    The most used wording of a name is shared; ties go to the alphabetically
    first. A grounded argument a capability leaves undescribed would silently
    borrow the shared wording, so such a catalog is refused instead.
    """
    uses: dict[str, dict[str, int]] = {}
    for name, capability in capabilities.items():
        metadata = capability.get("intent_metadata", {})
        described = metadata.get(DESCRIPTIONS) or {}
        for argument in metadata.get("grounded_arguments") or []:
            if argument not in described:
                raise ValueError(f"Capability {name} has no description for its grounded argument {argument}")
            texts = uses.setdefault(argument, {})
            texts[described[argument]] = texts.get(described[argument], 0) + 1
    shared = {}
    for argument, texts in sorted(uses.items()):
        text, count = min(texts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1:
            shared[argument] = text
    return shared


def compact_router_catalog(specialists: list[dict], *, factor_defaults: bool = False) -> dict:
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
            if factor_defaults:
                capability["intent_metadata"] = _without_defaults(name, metadata)
            if name in capabilities and capabilities[name] != capability:
                raise ValueError(f"Conflicting trusted capability definitions: {name}")
            capabilities[name] = capability
            agent["capabilities"].append(name)
        agents.append(agent)
    if factor_defaults:
        # A wording several capabilities share is stated once; a capability keeps
        # only the descriptions that differ from it. Nothing is dropped: each
        # grounded argument has exactly one description, its own or the shared one.
        shared = _shared_descriptions(capabilities)
        for capability in capabilities.values():
            metadata = capability["intent_metadata"]
            own = {argument: text for argument, text in (metadata.get(DESCRIPTIONS) or {}).items()
                   if shared.get(argument) != text}
            if own:
                metadata[DESCRIPTIONS] = own
            else:
                metadata.pop(DESCRIPTIONS, None)
        return {"intent_metadata_defaults": deepcopy(METADATA_DEFAULTS), "argument_descriptions": shared,
                "specialists": agents, "capabilities": capabilities}
    return {"specialists": agents, "capabilities": capabilities}


def render_router_catalog(specialists: list[dict], *, factor_defaults: bool = False) -> str:
    defaults = (" intent_metadata_defaults gives the value of every intent_metadata field a capability "
                "leaves absent; an absent field has exactly that default value. argument_descriptions "
                "describes each grounded argument whose capability lists no description of its own."
                if factor_defaults else "")
    return (
        "Catalog format: specialists[].capabilities lists exact names in the shared "
        "capabilities dictionary. A specialist may only receive capabilities in its own list. "
        "Shared semantic fields occur once in intent_metadata; they retain their full meaning."
        + defaults + "\n"
        + json.dumps(compact_router_catalog(specialists, factor_defaults=factor_defaults),
                     ensure_ascii=False, separators=(",", ":"))
    )
