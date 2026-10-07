from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tools.runtime_schema import (
    runtime_optional_arguments,
    runtime_required_arguments,
    runtime_tool_schema,
)


VALID_OPERATION_KINDS = {
    "create",
    "read",
    "update",
    "delete",
    "search",
    "action",
}

VALID_EFFECTS = {
    "read",
    "mutation",
}

MUTATION_RISKS = {
    "low",
    "medium",
    "high",
    "critical",
}


@dataclass(frozen=True)
class CapabilityContract:
    name: str
    resource_type: str
    operation_kind: str
    effect: str
    risk: str
    requires_approval: bool
    permission: str
    # Model-facing arguments: values an AI/tool caller may propose.
    parameters: tuple[str, ...]

    # Full executable Python signature, including trusted runtime-injected
    # policy values that must never become model-proposable arguments.
    execution_parameters: tuple[str, ...]

    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...]
    grounded_arguments: tuple[str, ...]
    derived_arguments: tuple[str, ...]
    trusted_policy_arguments: tuple[str, ...]
    condition_fields: tuple[str, ...]
    requiredness_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "resource_type": self.resource_type,
            "operation_kind": self.operation_kind,
            "effect": self.effect,
            "risk": self.risk,
            "requires_approval": self.requires_approval,
            "permission": self.permission,
            "parameters": list(self.parameters),
            "execution_parameters": list(self.execution_parameters),
            "required_arguments": list(self.required_arguments),
            "optional_arguments": list(self.optional_arguments),
            "grounded_arguments": list(self.grounded_arguments),
            "derived_arguments": list(self.derived_arguments),
            "trusted_policy_arguments": list(self.trusted_policy_arguments),
            "condition_fields": list(self.condition_fields),
            "requiredness_source": self.requiredness_source,
        }


def _string_list(tool_name: str, tool: Mapping[str, Any], field: str) -> list[str]:
    value = tool.get(field, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Capability '{tool_name}' has invalid {field} metadata.")

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"Capability '{tool_name}' has an invalid {field} entry.")
        normalized = item.strip()
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _normalize_effect(tool_name: str, tool: Mapping[str, Any], risk: str) -> str:
    explicit = tool.get("effect")
    if not isinstance(explicit, str) or not explicit.strip():
        raise ValueError(
            f"Capability '{tool_name}' requires explicit effect metadata."
        )
    effect = explicit.strip().lower()

    if effect not in VALID_EFFECTS:
        raise ValueError(f"Capability '{tool_name}' has unsupported effect={effect!r}.")

    if risk == "read" and effect != "read":
        raise ValueError(f"Capability '{tool_name}' risk=read must have effect=read.")

    if risk in MUTATION_RISKS and effect != "mutation":
        raise ValueError(f"Capability '{tool_name}' mutation risk must have effect=mutation.")

    return effect


def _runtime_authoritative_requiredness(
    tool_name: str,
    tool: Mapping[str, Any],
    parameter_names: tuple[str, ...],
    trusted_policy_arguments: tuple[str, ...],
) -> tuple[list[str], list[str], str]:
    """Resolve caller-visible requiredness from the executable signature.

    Runtime signatures may contain trusted policy arguments that are injected
    only after deterministic policy evaluation. Those arguments are part of
    the executable contract, but deliberately are NOT part of the model-facing
    parameter surface and therefore cannot become model required/optional args.
    """

    runtime_required = runtime_required_arguments(tool_name)
    runtime_optional = runtime_optional_arguments(tool_name)
    explicit_required = tool.get("required_arguments")

    model_parameter_set = set(parameter_names)
    trusted_policy_set = set(trusted_policy_arguments)

    if runtime_required is not None:
        runtime_required_set = set(runtime_required)
        required = [
            name
            for name in parameter_names
            if name in runtime_required_set
        ]
        requiredness_source = "executable_signature"

        if explicit_required is not None:
            declared_required = _string_list(tool_name, tool, "required_arguments")
            if set(declared_required) != set(required):
                raise ValueError(
                    f"Capability '{tool_name}' catalog required_arguments disagree with "
                    f"the executable model-facing signature; catalog={declared_required} "
                    f"runtime_model_required={required}"
                )
    elif explicit_required is not None:
        required = _string_list(tool_name, tool, "required_arguments")
        requiredness_source = "catalog"
    else:
        required = []
        requiredness_source = "unresolved"

    required_set = set(required)

    if runtime_optional is not None:
        runtime_optional_set = set(runtime_optional)
        optional = [
            name
            for name in parameter_names
            if name in runtime_optional_set
        ]
        expected_optional = [
            name
            for name in parameter_names
            if name not in required_set
        ]
        if set(optional) != set(expected_optional):
            raise ValueError(
                f"Capability '{tool_name}' runtime required/optional split does not cover "
                f"the model-facing parameters; optional={optional} expected={expected_optional}"
            )

        # Every trusted policy argument must itself exist in the executable
        # signature. Its required/optional runtime status is intentionally
        # irrelevant to the model-facing contract because trusted code owns it.
        runtime_argument_set = runtime_required_set | runtime_optional_set
        missing_policy_runtime = trusted_policy_set - runtime_argument_set
        if missing_policy_runtime:
            raise ValueError(
                f"Capability '{tool_name}' trusted_policy_arguments are not executable "
                f"parameters: {sorted(missing_policy_runtime)}"
            )
    else:
        optional = [name for name in parameter_names if name not in required_set]

    if set(required) - model_parameter_set:
        raise ValueError(
            f"Capability '{tool_name}' resolved caller required arguments outside the "
            "model-facing parameter surface."
        )

    return required, optional, requiredness_source


def build_capability_contract(tool_name: str, tool: Mapping[str, Any]) -> CapabilityContract:
    if not isinstance(tool, Mapping):
        raise ValueError(f"Capability '{tool_name}' metadata must be a mapping.")

    parameters = tool.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError(f"Capability '{tool_name}' parameters must be an object.")

    parameter_names = tuple(parameters.keys())
    parameter_set = set(parameter_names)

    for argument_name, schema in parameters.items():
        if not isinstance(argument_name, str) or not argument_name.strip():
            raise ValueError(f"Capability '{tool_name}' contains an invalid parameter name.")
        if not isinstance(schema, dict):
            raise ValueError(f"Capability '{tool_name}' parameter '{argument_name}' must be an object.")
        value_type = schema.get("type")
        if not isinstance(value_type, str) or not value_type.strip():
            raise ValueError(f"Capability '{tool_name}' parameter '{argument_name}' requires a type.")

    grounded = _string_list(tool_name, tool, "grounded_arguments")
    derived = _string_list(tool_name, tool, "derived_arguments")
    trusted_policy = _string_list(tool_name, tool, "trusted_policy_arguments")
    trusted_policy_tuple = tuple(trusted_policy)
    trusted_policy_set = set(trusted_policy)
    condition_fields = _string_list(tool_name, tool, "condition_fields")

    # Trusted policy values are execution-only. If they were also exposed in
    # `parameters`, the model could propose authority-bearing values that must
    # instead come from deterministic policy. Fail closed on that mistake.
    policy_model_overlap = trusted_policy_set & parameter_set
    if policy_model_overlap:
        raise ValueError(
            f"Capability '{tool_name}' exposes trusted policy arguments to the model: "
            f"{sorted(policy_model_overlap)}"
        )

    runtime_schema = runtime_tool_schema(tool_name)
    execution_parameters = parameter_names

    if runtime_schema is not None:
        runtime_parameters = set(runtime_schema.parameters)
        expected_runtime_parameters = parameter_set | trusted_policy_set

        if runtime_parameters != expected_runtime_parameters:
            runtime_only = runtime_parameters - expected_runtime_parameters
            catalog_only = expected_runtime_parameters - runtime_parameters
            raise ValueError(
                f"Capability '{tool_name}' catalog/runtime signature mismatch; "
                f"runtime_only={sorted(runtime_only)} catalog_only={sorted(catalog_only)}"
            )

        execution_parameters = tuple(runtime_schema.parameters)

    required, optional, requiredness_source = _runtime_authoritative_requiredness(
        tool_name,
        tool,
        parameter_names,
        trusted_policy_tuple,
    )

    for field_name, values in {
        "required_arguments": required,
        "optional_arguments": optional,
        "grounded_arguments": grounded,
        "derived_arguments": derived,
    }.items():
        unknown = set(values) - parameter_set
        if unknown:
            raise ValueError(
                f"Capability '{tool_name}' {field_name} references unknown parameters: "
                + ", ".join(sorted(unknown))
            )

    if set(required) & set(optional):
        raise ValueError(f"Capability '{tool_name}' has required/optional argument overlap.")

    if set(grounded) & set(derived):
        raise ValueError(f"Capability '{tool_name}' has grounded/derived argument overlap.")

    risk = tool.get("risk")
    if not isinstance(risk, str) or not risk.strip():
        raise ValueError(f"Capability '{tool_name}' requires trusted risk metadata.")
    risk = risk.strip().lower()

    effect = _normalize_effect(tool_name, tool, risk)

    requires_approval = tool.get("requires_approval")
    if not isinstance(requires_approval, bool):
        raise ValueError(f"Capability '{tool_name}' requires boolean requires_approval metadata.")

    if effect == "read" and requires_approval:
        # Read tools can still be audit logged, but the current approval system
        # is a mutation gate. Keeping read capabilities approval-free prevents
        # accidental deadlocks in no-op/status paths.
        raise ValueError(f"Capability '{tool_name}' read effect should not require approval.")

    if effect == "mutation" and risk == "read":
        raise ValueError(f"Capability '{tool_name}' mutation effect cannot use risk=read.")

    resource_type = tool.get("resource_type")
    if not isinstance(resource_type, str) or not resource_type.strip():
        raise ValueError(
            f"Capability '{tool_name}' requires explicit resource_type metadata."
        )
    resource_type = resource_type.strip().lower()

    operation_kind = tool.get("operation_kind")
    if not isinstance(operation_kind, str) or not operation_kind.strip():
        raise ValueError(
            f"Capability '{tool_name}' requires explicit operation_kind metadata."
        )
    operation_kind = operation_kind.strip().lower()

    if operation_kind not in VALID_OPERATION_KINDS:
        raise ValueError(
            f"Capability '{tool_name}' has unsupported operation_kind={operation_kind!r}."
        )

    if effect == "read" and operation_kind in {"create", "update", "delete"}:
        raise ValueError(
            f"Capability '{tool_name}' read effect has mutating operation_kind={operation_kind!r}."
        )

    if effect == "mutation" and operation_kind in {"read", "search"}:
        raise ValueError(
            f"Capability '{tool_name}' mutation effect has read operation_kind={operation_kind!r}."
        )

    permission = tool.get("permission")
    if not isinstance(permission, str) or not permission.strip():
        raise ValueError(
            f"Capability '{tool_name}' requires explicit permission metadata."
        )
    permission = permission.strip().lower()

    return CapabilityContract(
        name=tool_name,
        resource_type=resource_type,
        operation_kind=operation_kind,
        effect=effect,
        risk=risk,
        requires_approval=requires_approval,
        permission=permission,
        parameters=parameter_names,
        execution_parameters=execution_parameters,
        required_arguments=tuple(required),
        optional_arguments=tuple(optional),
        grounded_arguments=tuple(grounded),
        derived_arguments=tuple(derived),
        trusted_policy_arguments=tuple(trusted_policy),
        condition_fields=tuple(condition_fields),
        requiredness_source=requiredness_source,
    )


def required_arguments_for(tool_name: str, tool: Mapping[str, Any]) -> list[str]:
    return list(build_capability_contract(tool_name, tool).required_arguments)


def normalize_capability_catalog(
    tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        tool_name: build_capability_contract(tool_name, tool).as_dict()
        for tool_name, tool in sorted(tools.items())
    }


def validate_capability_registry(
    tools: Mapping[str, Mapping[str, Any]],
    *,
    require_runtime_schema: bool = False,
) -> list[CapabilityContract]:
    contracts: list[CapabilityContract] = []

    for tool_name in sorted(tools):
        contract = build_capability_contract(tool_name, tools[tool_name])
        if require_runtime_schema and runtime_tool_schema(tool_name) is None:
            raise ValueError(
                f"Capability '{tool_name}' has catalog metadata but no registered MCP executable signature."
            )
        if contract.requiredness_source == "unresolved":
            raise ValueError(
                f"Capability '{tool_name}' requiredness is unresolved; register the executable signature or declare required_arguments."
            )
        contracts.append(contract)

    return contracts
