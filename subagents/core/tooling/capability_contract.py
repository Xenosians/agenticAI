from __future__ import annotations

from copy import deepcopy
from typing import Any

CAPABILITY_CONTRACT_SCHEMA = "model-capability.v2"
CAPABILITY_CONTRACT_VERSION = 2


def _string_list(
    *,
    tool_name: str,
    field_name: str,
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(
            f"Capability '{tool_name}' has invalid {field_name} metadata."
        )

    normalized: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                f"Capability '{tool_name}' contains a non-string "
                f"{field_name} entry."
            )

        candidate = item.strip()

        if not candidate:
            raise ValueError(
                f"Capability '{tool_name}' contains an empty "
                f"{field_name} entry."
            )

        if candidate in seen:
            continue

        seen.add(candidate)
        normalized.append(candidate)

    return normalized


def _effect_from_risk(
    *,
    tool_name: str,
    tool: dict[str, Any],
) -> tuple[str, str]:
    risk = tool.get("risk")

    if not isinstance(risk, str) or not risk.strip():
        raise ValueError(
            f"Capability '{tool_name}' is missing trusted risk metadata."
        )

    normalized = risk.strip().lower()
    effect = "read" if normalized == "read" else "mutation"

    return normalized, effect


def build_model_capability_contract(
    *,
    tool_name: str,
    tool: dict[str, Any],
    description: str,
    argument_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Construct the additive v2 model-facing capability contract.

    This is descriptive metadata only. It exposes enough semantic structure
    for routing/specialist models to understand a capability without moving
    authorization into model output.

    SemanticGuard, ToolGateway, provider policy and approval execution remain
    authoritative.
    """

    risk, effect = _effect_from_risk(
        tool_name=tool_name,
        tool=tool,
    )

    requires_approval = tool.get("requires_approval")

    if not isinstance(requires_approval, bool):
        raise ValueError(
            f"Capability '{tool_name}' has invalid requires_approval metadata."
        )

    grounded_arguments = _string_list(
        tool_name=tool_name,
        field_name="grounded_arguments",
        value=tool.get("grounded_arguments", []),
    )

    condition_fields = _string_list(
        tool_name=tool_name,
        field_name="condition_fields",
        value=tool.get("condition_fields", []),
    )

    required_arguments = _string_list(
        tool_name=tool_name,
        field_name="required_arguments",
        value=tool.get("required_arguments", []),
    )

    derived_arguments = _string_list(
        tool_name=tool_name,
        field_name="derived_arguments",
        value=tool.get("derived_arguments", []),
    )

    trusted_policy_arguments = _string_list(
        tool_name=tool_name,
        field_name="trusted_policy_arguments",
        value=tool.get("trusted_policy_arguments", []),
    )

    policy_owns_preconditions = tool.get(
        "policy_owns_preconditions",
        False,
    )

    if not isinstance(policy_owns_preconditions, bool):
        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "policy_owns_preconditions metadata."
        )

    schema = (
        deepcopy(argument_schema)
        if argument_schema is not None
        else None
    )

    if schema is not None:
        unknown_grounded = [
            name
            for name in grounded_arguments
            if name not in schema
        ]

        if unknown_grounded:
            raise ValueError(
                f"Capability '{tool_name}' grounds unknown arguments: "
                + ", ".join(unknown_grounded)
            )

        unknown_required = [
            name
            for name in required_arguments
            if name not in schema
        ]

        if unknown_required:
            raise ValueError(
                f"Capability '{tool_name}' requires unknown arguments: "
                + ", ".join(unknown_required)
            )

    semantic_role_overlap = sorted(
        set(grounded_arguments)
        & set(derived_arguments)
    )

    if semantic_role_overlap:
        raise ValueError(
            f"Capability '{tool_name}' cannot mark the same argument "
            "as both grounded and derived: "
            + ", ".join(semantic_role_overlap)
        )

    unclassified_required = sorted(
        set(required_arguments)
        - set(grounded_arguments)
        - set(derived_arguments)
    )

    if unclassified_required:
        raise ValueError(
            f"Capability '{tool_name}' has required arguments without "
            "a grounded or derived semantic role: "
            + ", ".join(unclassified_required)
        )

    if schema is not None:
        unknown_derived = [
            name
            for name in derived_arguments
            if name not in schema
        ]

        if unknown_derived:
            raise ValueError(
                f"Capability '{tool_name}' derives unknown arguments: "
                + ", ".join(unknown_derived)
            )

    contract: dict[str, Any] = {
        "schema": CAPABILITY_CONTRACT_SCHEMA,
        "contract_version": CAPABILITY_CONTRACT_VERSION,
        "name": tool_name,
        "description": description.strip(),
        "effect": effect,
        "risk": risk,
        "requires_approval": requires_approval,
        "grounded_arguments": grounded_arguments,
        "required_arguments": required_arguments,
        "derived_arguments": derived_arguments,
        "condition_fields": condition_fields,
        "policy_owns_preconditions": policy_owns_preconditions,
        "trusted_policy_arguments": trusted_policy_arguments,
        "result_contract": {
            "type": "object",
            "condition_fields": condition_fields,
        },
    }

    if schema is not None:
        contract["argument_schema"] = schema

    return contract
