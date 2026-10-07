from __future__ import annotations

import re

from typing import (
    Any,
    Callable,
)

from subagents.core.definitions.types import (
    AgentDefinition,
    SemanticIntent,
)

from subagents.core.tooling.capabilities import (
    build_capability_spec,
    build_router_agent_spec,
)

from tools.registry import (
    get_tool,
)

from tools.capability_contract import (
    required_arguments_for,
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


def trusted_grounded_argument_descriptions(
    *,
    tool_name: str,
    tool: dict,
) -> dict[
    str,
    str,
]:
    """
    Return trusted descriptions for semantically grounded
    capability arguments.

    The Hub already receives grounded argument identities.

    Supplying their trusted parameter descriptions gives the
    routing model enough schema meaning to bind explicitly
    supplied current-turn values without introducing any
    domain-specific routing table.

    This metadata remains descriptive only. It does not authorize
    execution or weaken SemanticGuard exact binding.
    """

    grounded_arguments = (
        trusted_grounded_arguments(
            tool_name=(
                tool_name
            ),

            tool=(
                tool
            ),
        )
    )

    parameters = (
        tool.get(
            "parameters",
            {},
        )
    )

    if not isinstance(
        parameters,
        dict,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "parameters metadata."
        )

    descriptions: dict[
        str,
        str,
    ] = {}

    for argument_name in (
        grounded_arguments
    ):

        parameter = (
            parameters.get(
                argument_name
            )
        )

        if not isinstance(
            parameter,
            dict,
        ):

            continue

        description = (
            parameter.get(
                "description"
            )
        )

        if (
            isinstance(
                description,
                str,
            )
            and description.strip()
        ):

            descriptions[
                argument_name
            ] = (
                description.strip()
            )

    return (
        descriptions
    )


def _trusted_argument_role(
    *,
    tool_name: str,
    tool: dict,
    field_name: str,
) -> list[str]:
    # Generic trusted capability argument-role metadata.
    value = tool.get(field_name, [])

    if not isinstance(value, list):
        raise ValueError(
            f"Capability '{tool_name}' has invalid {field_name} metadata."
        )

    normalized: list[str] = []
    seen: set[str] = set()

    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Capability '{tool_name}' contains an invalid "
                f"{field_name} entry."
            )

        name = item.strip()
        if name in seen:
            continue

        seen.add(name)
        normalized.append(name)

    return normalized


def trusted_required_arguments(
    *,
    tool_name: str,
    tool: dict,
) -> list[str]:
    # Requiredness is an executable/API contract property, not a model
    # decision. Explicit catalog metadata wins; otherwise the registered
    # MCP Python signature supplies required-vs-optional semantics.
    return required_arguments_for(
        tool_name,
        tool,
    )


def trusted_derived_arguments(
    *,
    tool_name: str,
    tool: dict,
) -> list[str]:
    return _trusted_argument_role(
        tool_name=tool_name,
        tool=tool,
        field_name="derived_arguments",
    )


def _grounded_value_appears_in_request(
    value: str,
    user_request: str,
) -> bool:
    """
    Verify that one authority-bearing grounded value originates
    literally in the current user request.

    Important ordering:

        literal user value
            ↓
        request-origin validation
            ↓
        trusted bounded canonicalization

    Canonicalization is deliberately NOT performed here.

    This is generic. It does not know about Jira, accounts,
    repositories, projects, or any domain-specific identifier.
    """

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
        or not isinstance(
            user_request,
            str,
        )
        or not user_request.strip()
    ):
        return False

    identifier_chars = (
        r"A-Za-z0-9._/\\-"
    )

    pattern = (
        rf"(?<![{identifier_chars}])"
        rf"{re.escape(value)}"
        rf"(?="
        rf"$"
        rf"|[\s,;:!?()\[\]{{}}\"'`]"
        rf"|\.(?=\s|$)"
        rf")"
    )

    return (
        re.search(
            pattern,
            user_request,
        )
        is not None
    )


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


def trusted_policy_owns_preconditions(
    *,
    tool_name: str,
    tool: dict,
) -> bool:
    """
    Read generic model-facing metadata describing ownership of
    capability execution preconditions.

    This is descriptive routing metadata only.

    It does not authorize execution and does not bypass trusted
    policy evaluation.
    """

    value = (
        tool.get(
            "policy_owns_preconditions",
            False,
        )
    )

    if not isinstance(
        value,
        bool,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "policy_owns_preconditions metadata."
        )

    return value


def trusted_bounded_argument_values(
    *,
    tool_name: str,
    tool_lookup: ToolLookup = get_tool,
) -> dict[
    str,
    list[str],
]:
    """
    Return trusted bounded model-facing argument values for one
    capability.

    These values come only from trusted capability metadata via the
    existing argument_values_resolver path.

    Example:

        configured capability
            target -> ["alpha", "beta", "gamma"]

    This information may be used to canonicalize a Hub semantic
    contract.

    It does NOT:
        - authorize the capability;
        - bypass grounding;
        - widen the user's requested scope;
        - infer aliases not present in trusted metadata.
    """

    capability = (
        build_capability_spec(
            tool_name,

            tool_lookup=(
                tool_lookup
            ),

            include_arguments=True,
        )
    )

    argument_schema = (
        capability.get(
            "argument_schema"
        )
    )

    if not isinstance(
        argument_schema,
        dict,
    ):

        raise ValueError(
            f"Capability '{tool_name}' has invalid "
            "model-facing argument schema."
        )

    bounded: dict[
        str,
        list[str],
    ] = {}

    for (
        argument_name,
        field_schema,
    ) in argument_schema.items():

        if not isinstance(
            field_schema,
            dict,
        ):

            continue

        enum = (
            field_schema.get(
                "enum"
            )
        )

        if enum is None:

            continue

        if not isinstance(
            enum,
            list,
        ):

            raise ValueError(
                f"Capability '{tool_name}' contains invalid "
                f"bounded values for '{argument_name}'."
            )

        values: list[str] = []

        seen: set[str] = set()

        for item in enum:

            if not isinstance(
                item,
                str,
            ):

                raise ValueError(
                    f"Capability '{tool_name}' contains a "
                    "non-string bounded argument value."
                )

            normalized = (
                item.strip()
            )

            if not normalized:

                raise ValueError(
                    f"Capability '{tool_name}' contains an "
                    "empty bounded argument value."
                )

            if normalized in seen:

                continue

            seen.add(
                normalized
            )

            values.append(
                normalized
            )

        if values:

            bounded[
                argument_name
            ] = (
                values
            )

    return bounded


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

    Bounded argument values are included when trusted runtime
    metadata provides them.

    They are descriptive model-facing metadata only.
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

            "required_arguments":
                trusted_required_arguments(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                ),

            "derived_arguments":
                trusted_derived_arguments(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                ),

            "grounded_argument_descriptions":
                trusted_grounded_argument_descriptions(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),
                ),

            "bounded_argument_values":
                trusted_bounded_argument_values(
                    tool_name=(
                        tool_name
                    ),

                    tool_lookup=(
                        tool_lookup
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

            "policy_owns_preconditions":
                trusted_policy_owns_preconditions(
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

        # Hub models sometimes emit JSON null for an optional argument
        # they did not actually ground in the current request. Omitting
        # that entry is strictly narrower than accepting a fabricated
        # value and cannot widen execution authority.
        if values is None:
            continue

        if (
            isinstance(
                values,
                list,
            )
            and not values
        ):
            # Empty model-side semantic bindings are absence, not a
            # requirement that the worker must emit an empty value.
            continue

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


def _collect_trusted_bounded_values(
    *,
    tool_names: list[str],
    tool_lookup: ToolLookup,
) -> dict[
    str,
    list[str],
]:
    """
    Build a union of trusted bounded values exposed by the
    capabilities participating in one semantic intent.

    Values remain exact trusted canonical identifiers.
    """

    result: dict[
        str,
        list[str],
    ] = {}

    seen: dict[
        str,
        set[str],
    ] = {}

    for tool_name in tool_names:

        values_by_argument = (
            trusted_bounded_argument_values(
                tool_name=(
                    tool_name
                ),

                tool_lookup=(
                    tool_lookup
                ),
            )
        )

        for (
            argument_name,
            values,
        ) in values_by_argument.items():

            output_values = (
                result.setdefault(
                    argument_name,
                    [],
                )
            )

            output_seen = (
                seen.setdefault(
                    argument_name,
                    set(),
                )
            )

            for item in values:

                if item in output_seen:

                    continue

                output_seen.add(
                    item
                )

                output_values.append(
                    item
                )

    return result


def _canonicalize_bounded_value(
    value: str,
    *,
    trusted_values: list[str],
    field_name: str,
) -> str:
    """
    Canonicalize one Hub-produced value only when trusted bounded
    metadata makes the mapping deterministic.

    Rules:

        exact trusted value
            -> preserve

        exactly one case-insensitive trusted match
            -> canonical trusted spelling

        no trusted match
            -> reject

        multiple case-insensitive trusted matches
            -> reject as ambiguous

    No substring, synonym, abbreviation, fuzzy, semantic, or
    model-based normalization is permitted.
    """

    if value in trusted_values:

        return value

    folded = (
        value.casefold()
    )

    matches = [
        candidate

        for candidate
        in trusted_values

        if (
            candidate.casefold()
            == folded
        )
    ]

    if len(
        matches
    ) == 1:

        return (
            matches[
                0
            ]
        )

    if not matches:

        raise ValueError(
            f"{field_name} contains value "
            f"'{value}' that does not match any "
            "trusted bounded value."
        )

    raise ValueError(
        f"{field_name} contains value "
        f"'{value}' whose trusted bounded mapping "
        "is ambiguous."
    )


def _canonicalize_bounded_argument_map(
    value: dict[
        str,
        list[str],
    ],
    *,
    bounded_values: dict[
        str,
        list[str],
    ],
    field_name: str,
) -> dict[
    str,
    list[str],
]:
    """
    Canonicalize only argument values backed by trusted bounded
    metadata.

    Arguments without trusted bounded values remain exact.

    This distinction is critical.

    Example:

        repository:
            "AI" -> "ai"

        resource:
            "resource admin" remains "resource admin"

    because repository may have trusted configured enum values while
    resource may not.
    """

    normalized: dict[
        str,
        list[str],
    ] = {}

    for (
        argument_name,
        values,
    ) in value.items():

        trusted_values = (
            bounded_values.get(
                argument_name
            )
        )

        if not trusted_values:

            normalized[
                argument_name
            ] = (
                list(
                    values
                )
            )

            continue

        canonical_values: list[str] = []

        seen: set[str] = set()

        for item in values:

            canonical = (
                _canonicalize_bounded_value(
                    item,

                    trusted_values=(
                        trusted_values
                    ),

                    field_name=(
                        f"{field_name}.{argument_name}"
                    ),
                )
            )

            if canonical in seen:

                continue

            seen.add(
                canonical
            )

            canonical_values.append(
                canonical
            )

        normalized[
            argument_name
        ] = (
            canonical_values
        )

    return normalized


def parse_semantic_intent(
    value: Any,
    *,
    agent: AgentDefinition,
    tool_lookup: ToolLookup = get_tool,
    user_request: (
        str
        | None
    ) = None,
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

    Trusted bounded values may be canonicalized deterministically.

    Unbounded grounded values remain exact.
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

    if (
        not isinstance(
            summary,
            str,
        )
        or not summary.strip()
    ):

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
        and len(
            allowed_tools
        )
        != 1
    ):

        raise ValueError(
            "A non-clarification intent must contain "
            "exactly one allowed tool for the current "
            "exactly-one-call specialist runtime."
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
    required_argument_names: set[str] = set()
    derived_argument_names: set[str] = set()

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

        required_argument_names.update(
            trusted_required_arguments(
                tool_name=(
                    tool_name
                ),

                tool=(
                    tool
                ),
            )
        )

        derived_argument_names.update(
            trusted_derived_arguments(
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


    # Known derived arguments are non-authoritative proposal content.
    #
    # A Hub may incorrectly place one into an authority-bearing
    # semantic argument map. Because the argument identity itself is
    # trusted capability metadata, narrowing that mistake to ABSENCE is
    # safe.
    #
    # Unknown argument names are deliberately NOT removed here. They
    # continue into the grounded-argument validator and fail closed.
    for argument_map in (
        allowed_arguments,
        forbidden_arguments,
    ):

        for argument_name in list(
            argument_map
        ):

            if (
                argument_name
                in derived_argument_names
            ):

                argument_map.pop(
                    argument_name,
                    None,
                )

    for argument_name in (
        set(
            allowed_arguments
        )
        | set(
            forbidden_arguments
        )
    ):

        if (
            argument_name
            not in grounded_argument_names
        ):

            raise ValueError(
                "Semantic intent references untrusted "
                "or irrelevant grounded argument "
                f"'{argument_name}'."
            )

    # ========================================================
    # TRUSTED ARGUMENT ROLES
    # ========================================================

    semantic_role_overlap = (
        grounded_argument_names
        & derived_argument_names
    )

    if semantic_role_overlap:
        raise ValueError(
            "Capability metadata marks arguments as both grounded "
            "and derived: "
            + ", ".join(
                sorted(
                    semantic_role_overlap
                )
            )
        )

    unclassified_required = (
        required_argument_names
        - grounded_argument_names
        - derived_argument_names
    )

    if unclassified_required:
        raise ValueError(
            "Required capability arguments are missing a trusted "
            "grounded or derived semantic role: "
            + ", ".join(
                sorted(
                    unclassified_required
                )
            )
        )


    # ========================================================
    # CURRENT-REQUEST ORIGIN VALIDATION
    #
    # Do this BEFORE bounded-value canonicalization.
    #
    # This prevents:
    #
    #     user says:
    #         internal meeting
    #
    #     Hub invents:
    #         project_key = INTERNAL
    #
    # from becoming trusted merely because a configured enum happens
    # to contain INTERNAL.
    #
    # This remains domain-independent.
    # ========================================================

    raw_allowed_arguments = {
        argument_name:
            list(
                values
            )

        for (
            argument_name,
            values,
        )
        in allowed_arguments.items()
    }

    raw_forbidden_arguments = {
        argument_name:
            list(
                values
            )

        for (
            argument_name,
            values,
        )
        in forbidden_arguments.items()
    }


    if user_request is not None:

        if not isinstance(
            user_request,
            str,
        ):
            raise ValueError(
                "user_request must be a string when semantic "
                "grounding validation is enabled."
            )


        def current_request_bindings(
            argument_map: dict[
                str,
                list[str],
            ],
            *,
            allow_casefold_match: bool,
        ) -> dict[
            str,
            list[str],
        ]:

            narrowed: dict[
                str,
                list[str],
            ] = {}

            request_for_match = (
                user_request.casefold()
                if allow_casefold_match
                else user_request
            )

            for (
                argument_name,
                values,
            ) in argument_map.items():

                if (
                    argument_name
                    not in grounded_argument_names
                ):
                    continue

                current_values: list[str] = []

                for item in values:

                    candidate = (
                        item.casefold()
                        if allow_casefold_match
                        else item
                    )

                    if (
                        _grounded_value_appears_in_request(
                            candidate,
                            request_for_match,
                        )
                    ):
                        current_values.append(
                            item
                        )

                if current_values:
                    narrowed[
                        argument_name
                    ] = (
                        current_values
                    )

            return narrowed


        # Allowed bindings can authorize concrete execution scope.
        # Require strict literal request origin.
        allowed_arguments = (
            current_request_bindings(
                raw_allowed_arguments,
                allow_casefold_match=False,
            )
        )

        # Forbidden bindings only reduce authority.
        # Case-only normalization is therefore safe and avoids losing
        # explicit exclusions such as "Alice" -> "alice".
        forbidden_arguments = (
            current_request_bindings(
                raw_forbidden_arguments,
                allow_casefold_match=True,
            )
        )


    # ========================================================
    # TRUSTED BOUNDED-VALUE CANONICALIZATION
    #
    # Only values that already survived exact current-request
    # origin validation reach this point.
    #
    # Example:
    #
    #     user literal: "kan"
    #         ↓
    #     proven present
    #         ↓
    #     trusted enum: ["KAN"]
    #         ↓
    #     canonical: "KAN"
    #
    # No fuzzy/synonym/model normalization occurs.
    # ========================================================

    bounded_values = (
        _collect_trusted_bounded_values(
            tool_names=(
                allowed_tools
            ),

            tool_lookup=(
                tool_lookup
            ),
        )
    )

    allowed_arguments = (
        _canonicalize_bounded_argument_map(
            allowed_arguments,

            bounded_values=(
                bounded_values
            ),

            field_name=(
                "intent.allowed_arguments"
            ),
        )
    )

    forbidden_arguments = (
        _canonicalize_bounded_argument_map(
            forbidden_arguments,

            bounded_values=(
                bounded_values
            ),

            field_name=(
                "intent.forbidden_arguments"
            ),
        )
    )


    # ========================================================
    # REQUIRED EXACT VALUES
    #
    # Required + grounded means:
    #
    #     execution cannot safely continue without an exact
    #     authority-bearing value.
    #
    # Missing values cause clarification.
    #
    # Required + derived values are different:
    #
    #     they may be composed later by the specialist.
    # ========================================================

    missing_required_arguments = sorted(
        (
            required_argument_names
            & grounded_argument_names
        )
        - set(
            allowed_arguments
        )
    )

    if missing_required_arguments:
        clarification_required = (
            True
        )


    # ========================================================
    # CONFLICT CHECK AFTER CANONICALIZATION
    #
    # Example:
    #
    #     allow "AI"
    #     forbid "ai"
    #
    # both become canonical "ai" and therefore conflict.
    # ========================================================

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

            missing_required_arguments=(
                missing_required_arguments
            ),
        )
    )