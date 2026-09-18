from __future__ import annotations

from typing import (
    Any,
    Callable,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    SemanticIntent,
)

from subagents.core.tooling.capabilities import (
    build_router_agent_spec,
)

from tools.registry import (
    get_tool,
)


ToolLookup = Callable[
    [
        str,
    ],
    dict
    | None,
]


VALID_EFFECTS = {
    "read",
    "mutation",
    "unknown",
}


def trusted_tool_effect(
    tool: dict,
) -> str:
    """
    Derive a generic semantic effect from trusted capability
    metadata.

    The rule is intentionally domain-independent:

        risk == "read"
            -> read

        every governed non-read operation
            -> mutation

    Tool-specific names are never inspected.
    """

    risk = (
        tool.get(
            "risk"
        )
    )

    if not isinstance(
        risk,
        str,
    ):

        raise ValueError(
            "Capability is missing trusted risk metadata."
        )

    normalized = (
        risk
        .strip()
        .lower()
    )

    if not normalized:

        raise ValueError(
            "Capability contains empty trusted risk metadata."
        )

    if normalized == "read":

        return "read"

    return "mutation"


def trusted_grounded_arguments(
    *,
    tool_name: str,
    tool: dict,
) -> list[str]:
    """
    Read generic semantically-grounded argument names from the
    trusted tool registry.

    No domain-specific argument such as user_id, ticket_id,
    repository, path, or resource is hardcoded here.
    """

    value = (
        tool.get(
            "grounded_arguments",
            [],
        )
    )

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "grounded_arguments metadata."
        )

    normalized: list[str] = []

    seen: set[str] = set()

    for item in value:

        if not isinstance(
            item,
            str,
        ):

            raise ValueError(
                f"Capability '{tool_name}' contains a "
                "non-string grounded argument."
            )

        name = (
            item.strip()
        )

        if not name:

            raise ValueError(
                f"Capability '{tool_name}' contains an "
                "empty grounded argument."
            )

        if name in seen:

            continue

        seen.add(
            name
        )

        normalized.append(
            name
        )

    return normalized


def trusted_condition_fields(
    *,
    tool_name: str,
    tool: dict,
) -> list[str]:
    """
    Return trusted top-level structured-result fields that may be
    used for deterministic workflow branching.

    Capabilities opt in explicitly.

    Missing metadata means that capability may NOT authorize a
    result-dependent workflow branch.
    """

    value = (
        tool.get(
            "condition_fields",
            [],
        )
    )

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "condition_fields metadata."
        )

    normalized: list[str] = []

    seen: set[str] = set()

    for item in value:

        if not isinstance(
            item,
            str,
        ):

            raise ValueError(
                f"Capability '{tool_name}' contains a "
                "non-string condition field."
            )

        field_name = (
            item.strip()
        )

        if not field_name:

            raise ValueError(
                f"Capability '{tool_name}' contains an "
                "empty condition field."
            )

        if field_name in seen:

            continue

        seen.add(
            field_name
        )

        normalized.append(
            field_name
        )

    return normalized


def trusted_requires_approval(
    *,
    tool_name: str,
    tool: dict,
) -> bool:
    """
    Read the static trusted approval requirement.

    Conditional mutation v1 deliberately supports only capabilities
    whose trusted metadata guarantees approval and whose approval
    policy is not dynamically rewritten at execution time.
    """

    value = (
        tool.get(
            "requires_approval"
        )
    )

    if not isinstance(
        value,
        bool,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "requires_approval metadata."
        )

    return value


def trusted_condition_fields(
    *,
    tool_name: str,
    tool: dict,
) -> list[str]:
    """
    Return trusted top-level structured-result fields that may
    control deterministic workflow branching.

    Capabilities must explicitly opt in.

    Missing metadata means the capability cannot be used as a
    result-dependent workflow source.
    """

    value = (
        tool.get(
            "condition_fields",
            [],
        )
    )

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "condition_fields metadata."
        )

    normalized: list[str] = []
    seen: set[str] = set()

    for item in value:

        if not isinstance(
            item,
            str,
        ):

            raise ValueError(
                f"Capability '{tool_name}' contains a "
                "non-string condition field."
            )

        field_name = (
            item.strip()
        )

        if not field_name:

            raise ValueError(
                f"Capability '{tool_name}' contains an "
                "empty condition field."
            )

        if field_name in seen:
            continue

        seen.add(
            field_name
        )

        normalized.append(
            field_name
        )

    return normalized


def trusted_requires_approval(
    *,
    tool_name: str,
    tool: dict,
) -> bool:
    """
    Return the capability's static trusted approval requirement.
    """

    value = (
        tool.get(
            "requires_approval"
        )
    )

    if not isinstance(
        value,
        bool,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "requires_approval metadata."
        )

    return value


def build_router_semantic_agent_spec(
    agent: AgentDefinition,
    *,
    tool_lookup: ToolLookup = get_tool,
) -> dict[
    str,
    Any,
]:
    """
    Build Hub-facing specialist metadata from the trusted runtime
    registry.

    There is no parallel hardcoded workflow table.
    """

    spec = (
        build_router_agent_spec(
            agent,
            tool_lookup=(
                tool_lookup
            ),
        )
    )

    capabilities = (
        spec.get(
            "capabilities"
        )
    )

    if not isinstance(
        capabilities,
        list,
    ):

        raise ValueError(
            f"Agent '{agent.name}' has invalid "
            "router capability metadata."
        )

    for capability in capabilities:

        if not isinstance(
            capability,
            dict,
        ):

            raise ValueError(
                f"Agent '{agent.name}' has invalid "
                "router capability metadata."
            )

        tool_name = (
            capability.get(
                "name"
            )
        )

        if (
            not isinstance(
                tool_name,
                str,
            )
            or not tool_name.strip()
        ):

            raise ValueError(
                f"Agent '{agent.name}' has invalid "
                "capability identity."
            )

        tool_name = (
            tool_name.strip()
        )

        tool = (
            tool_lookup(
                tool_name
            )
        )

        if tool is None:

            raise ValueError(
                "Unknown trusted capability: "
                f"{tool_name}"
            )

        capability[
            "intent_metadata"
        ] = {
            "effect":
                trusted_tool_effect(
                    tool
                ),

            "grounded_arguments":
                trusted_grounded_arguments(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                ),

            "condition_fields":
                trusted_condition_fields(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                ),

            "requires_approval":
                trusted_requires_approval(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                ),
        }

    return spec


def _normalize_string_list(
    value: Any,
    *,
    field_name: str,
) -> list[str]:

    if not isinstance(
        value,
        list,
    ):

        raise ValueError(
            f"{field_name} must be a list."
        )

    normalized: list[str] = []

    seen: set[str] = set()

    for item in value:

        if not isinstance(
            item,
            str,
        ):

            raise ValueError(
                f"{field_name} must contain only strings."
            )

        item = (
            item.strip()
        )

        if not item:

            raise ValueError(
                f"{field_name} must not contain "
                "empty values."
            )

        if item in seen:

            continue

        seen.add(
            item
        )

        normalized.append(
            item
        )

    return normalized


def _normalize_argument_map(
    value: Any,
    *,
    field_name: str,
) -> dict[
    str,
    list[str],
]:

    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            f"{field_name} must be an object."
        )

    normalized: dict[
        str,
        list[str],
    ] = {}

    for (
        argument_name,
        values,
    ) in value.items():

        if not isinstance(
            argument_name,
            str,
        ):

            raise ValueError(
                f"{field_name} contains an invalid "
                "argument name."
            )

        argument_name = (
            argument_name.strip()
        )

        if not argument_name:

            raise ValueError(
                f"{field_name} contains an empty "
                "argument name."
            )

        normalized[
            argument_name
        ] = (
            _normalize_string_list(
                values,
                field_name=(
                    f"{field_name}.{argument_name}"
                ),
            )
        )

    return normalized


def parse_semantic_intent(
    value: Any,
    *,
    agent: AgentDefinition,
    tool_lookup: ToolLookup = get_tool,
) -> (
    SemanticIntent
    | None
):
    """
    Validate one Hub-produced semantic intent contract against the
    trusted agent/tool registry.

    None remains supported temporarily for legacy callers and
    historical unit fixtures.

    When a semantic contract IS present, malformed or invented
    capability metadata fails closed.
    """

    if value is None:

        return None

    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            "intent must be an object."
        )

    supported_fields = {
        "summary",
        "effect",
        "allowed_tools",
        "forbidden_tools",
        "allowed_arguments",
        "forbidden_arguments",
        "max_tool_calls",
        "clarification_required",
    }

    unexpected_fields = (
        set(
            value.keys()
        )
        - supported_fields
    )

    if unexpected_fields:

        raise ValueError(
            "intent contains unsupported fields: "
            + ", ".join(
                sorted(
                    unexpected_fields
                )
            )
        )

    summary = (
        value.get(
            "summary"
        )
    )

    effect = (
        value.get(
            "effect"
        )
    )

    max_tool_calls = (
        value.get(
            "max_tool_calls"
        )
    )

    clarification_required = (
        value.get(
            "clarification_required"
        )
    )

    if not isinstance(
        summary,
        str,
    ) or not summary.strip():

        raise ValueError(
            "intent.summary must be a non-empty string."
        )

    summary = (
        summary.strip()
    )

    if not isinstance(
        effect,
        str,
    ):

        raise ValueError(
            "intent.effect must be a string."
        )

    effect = (
        effect
        .strip()
        .lower()
    )

    if effect not in VALID_EFFECTS:

        raise ValueError(
            "intent.effect must be read, mutation, "
            "or unknown."
        )

    if (
        not isinstance(
            max_tool_calls,
            int,
        )
        or isinstance(
            max_tool_calls,
            bool,
        )
        or max_tool_calls != 1
    ):

        raise ValueError(
            "intent.max_tool_calls must be exactly 1 "
            "for the current specialist runtime."
        )

    if not isinstance(
        clarification_required,
        bool,
    ):

        raise ValueError(
            "intent.clarification_required must "
            "be boolean."
        )

    allowed_tools = (
        _normalize_string_list(
            value.get(
                "allowed_tools",
                [],
            ),
            field_name=(
                "intent.allowed_tools"
            ),
        )
    )

    forbidden_tools = (
        _normalize_string_list(
            value.get(
                "forbidden_tools",
                [],
            ),
            field_name=(
                "intent.forbidden_tools"
            ),
        )
    )

    allowed_arguments = (
        _normalize_argument_map(
            value.get(
                "allowed_arguments",
                {},
            ),
            field_name=(
                "intent.allowed_arguments"
            ),
        )
    )

    forbidden_arguments = (
        _normalize_argument_map(
            value.get(
                "forbidden_arguments",
                {},
            ),
            field_name=(
                "intent.forbidden_arguments"
            ),
        )
    )

    if (
        not clarification_required
        and not allowed_tools
    ):

        raise ValueError(
            "A non-clarification intent must contain "
            "at least one allowed tool."
        )

    trusted_agent_tools = {
        tool_name.strip()

        for tool_name
        in agent.tools

        if tool_name.strip()
    }

    for tool_name in (
        allowed_tools
        + forbidden_tools
    ):

        if tool_name not in trusted_agent_tools:

            raise ValueError(
                f"Semantic intent references capability "
                f"'{tool_name}' that is not available to "
                f"agent '{agent.name}'."
            )

    overlap = (
        set(
            allowed_tools
        )
        & set(
            forbidden_tools
        )
    )

    if overlap:

        raise ValueError(
            "Semantic intent cannot both allow and forbid "
            "the same capability."
        )

    allowed_tool_effects: set[str] = set()

    grounded_argument_names: set[str] = set()

    for tool_name in allowed_tools:

        tool = (
            tool_lookup(
                tool_name
            )
        )

        if tool is None:

            raise ValueError(
                f"Unknown trusted capability: {tool_name}"
            )

        allowed_tool_effects.add(
            trusted_tool_effect(
                tool
            )
        )

        grounded_argument_names.update(
            trusted_grounded_arguments(
                tool_name=(
                    tool_name
                ),

                tool=(
                    tool
                ),
            )
        )

    if allowed_tool_effects:

        if len(
            allowed_tool_effects
        ) != 1:

            raise ValueError(
                "One semantic intent may not mix read "
                "and mutation capabilities."
            )

        trusted_effect = (
            next(
                iter(
                    allowed_tool_effects
                )
            )
        )

        if (
            effect
            != trusted_effect
        ):

            raise ValueError(
                "Semantic intent effect does not match "
                "trusted capability metadata."
            )

    elif (
        effect
        != "unknown"
    ):

        raise ValueError(
            "An intent without an allowed capability "
            "must use effect='unknown'."
        )

    for argument_name in (
        set(
            allowed_arguments
        )
        | set(
            forbidden_arguments
        )
    ):

        if argument_name not in grounded_argument_names:

            raise ValueError(
                "Semantic intent references untrusted "
                "or irrelevant grounded argument "
                f"'{argument_name}'."
            )

    for argument_name in (
        set(
            allowed_arguments
        )
        & set(
            forbidden_arguments
        )
    ):

        conflicting_values = (
            set(
                allowed_arguments[
                    argument_name
                ]
            )
            & set(
                forbidden_arguments[
                    argument_name
                ]
            )
        )

        if conflicting_values:

            raise ValueError(
                "Semantic intent cannot both allow and "
                "forbid the same argument value."
            )

    return (
        SemanticIntent(
            summary=(
                summary
            ),

            effect=(
                effect
            ),

            allowed_tools=(
                allowed_tools
            ),

            forbidden_tools=(
                forbidden_tools
            ),

            allowed_arguments=(
                allowed_arguments
            ),

            forbidden_arguments=(
                forbidden_arguments
            ),

            max_tool_calls=(
                max_tool_calls
            ),

            clarification_required=(
                clarification_required
            ),
        )
    )
