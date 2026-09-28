from __future__ import annotations

from copy import (
    deepcopy,
)

from typing import (
    Any,
    Callable,
)

from subagents.core.definitions.types import (
    AgentDefinition,
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


def _resolve_argument_values(
    *,
    tool_name: str,
    tool: dict,
) -> dict[
    str,
    list[str],
]:
    """
    Resolve optional trusted bounded argument values.

    Tool implementations may expose:

        argument_values_resolver

    returning:

        {
            "argument_name": [
                "allowed-value-1",
                "allowed-value-2",
            ]
        }

    These values are model-facing metadata only.

    They do not replace ToolGateway authorization, grounding,
    policy evaluation, or trusted execution.
    """

    resolver = (
        tool.get(
            "argument_values_resolver"
        )
    )

    if resolver is None:

        return {}

    if not callable(
        resolver
    ):

        raise ValueError(
            f"Capability '{tool_name}' "
            "has an invalid "
            "argument_values_resolver."
        )

    resolved = (
        resolver()
    )

    if not isinstance(
        resolved,
        dict,
    ):

        raise ValueError(
            f"Capability '{tool_name}' "
            "argument value resolver "
            "returned an invalid result."
        )

    normalized: dict[
        str,
        list[str],
    ] = {}

    for (
        field_name,
        values,
    ) in resolved.items():

        if not isinstance(
            field_name,
            str,
        ) or not field_name.strip():

            raise ValueError(
                f"Capability '{tool_name}' "
                "argument value resolver "
                "returned an invalid field."
            )

        if not isinstance(
            values,
            list,
        ):

            raise ValueError(
                f"Capability '{tool_name}' "
                f"argument values for "
                f"'{field_name}' must be a list."
            )

        normalized_values: list[
            str
        ] = []

        seen: set[
            str
        ] = set()

        for value in values:

            if not isinstance(
                value,
                str,
            ):

                raise ValueError(
                    f"Capability '{tool_name}' "
                    f"argument value for "
                    f"'{field_name}' must "
                    "be a string."
                )

            normalized_value = (
                value.strip()
            )

            if not normalized_value:

                raise ValueError(
                    f"Capability '{tool_name}' "
                    f"contains an empty "
                    f"argument value for "
                    f"'{field_name}'."
                )

            if normalized_value in seen:

                continue

            seen.add(
                normalized_value
            )

            normalized_values.append(
                normalized_value
            )

        if not normalized_values:

            raise ValueError(
                f"Capability '{tool_name}' "
                f"argument values for "
                f"'{field_name}' are empty."
            )

        normalized[
            field_name.strip()
        ] = (
            normalized_values
        )

    return normalized


def _build_argument_schema(
    *,
    tool_name: str,
    tool: dict,
) -> dict[
    str,
    Any,
]:
    """
    Build a model-facing argument schema without mutating the
    trusted global tool registry.

    Runtime-resolved bounded values are represented using
    conventional JSON-schema-style `enum` metadata.
    """

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
            f"Capability '{tool_name}' "
            "has an invalid argument schema."
        )

    argument_schema = (
        deepcopy(
            parameters
        )
    )

    bounded_values = (
        _resolve_argument_values(
            tool_name=(
                tool_name
            ),

            tool=(
                tool
            ),
        )
    )

    for (
        field_name,
        values,
    ) in bounded_values.items():

        field_schema = (
            argument_schema.get(
                field_name
            )
        )

        if field_schema is None:

            raise ValueError(
                f"Capability '{tool_name}' "
                "provides bounded values for "
                f"unknown argument "
                f"'{field_name}'."
            )

        if not isinstance(
            field_schema,
            dict,
        ):

            raise ValueError(
                f"Capability '{tool_name}' "
                f"argument '{field_name}' "
                "has an invalid schema."
            )

        field_schema[
            "enum"
        ] = (
            list(
                values
            )
        )

    return argument_schema


def build_capability_spec(
    tool_name: str,
    *,
    tool_lookup: ToolLookup = get_tool,
    include_arguments: bool = True,
) -> dict[
    str,
    Any,
]:
    """
    Build the model-facing representation of one trusted
    capability.

    Internal implementation details are intentionally excluded.

    The model may understand:

        capability name
        capability purpose
        structured argument schema
        bounded argument values

    Trusted application code still owns:

        authorization
        grounding
        risk
        approvals
        execution policy
        provider implementation
    """

    tool = (
        tool_lookup(
            tool_name
        )
    )

    if tool is None:

        raise ValueError(
            "Unknown capability referenced "
            f"by agent: {tool_name}"
        )

    description = (
        tool.get(
            "description"
        )
    )

    if not isinstance(
        description,
        str,
    ) or not description.strip():

        raise ValueError(
            f"Capability '{tool_name}' "
            "has an invalid description."
        )

    capability: dict[
        str,
        Any,
    ] = {
        "name":
            tool_name,

        "description":
            description.strip(),
    }

    if include_arguments:

        capability[
            "argument_schema"
        ] = (
            _build_argument_schema(
                tool_name=(
                    tool_name
                ),

                tool=(
                    tool
                ),
            )
        )

    return capability


def build_agent_capability_catalog(
    agent: AgentDefinition,
    *,
    tool_lookup: ToolLookup = get_tool,
    include_arguments: bool = True,
    allowed_tools: (
        list[str]
        | None
    ) = None,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Build the trusted model-facing capability catalog for one
    specialist.

    AgentDefinition.tools owns the specialist's maximum trusted
    capability boundary.

    When allowed_tools is supplied, it may only NARROW that
    boundary for the current semantic task.

    It can never add a capability that the specialist does not own.

    This filtering is model-facing guidance only.

    SemanticGuard and ToolGateway remain independently authoritative
    for semantic validation, grounding, authorization, policy,
    approval, and execution.
    """

    # ============================================================
    # TRUSTED AGENT CAPABILITY BOUNDARY
    # ============================================================

    trusted_tools: list[
        str
    ] = []

    trusted_seen: set[
        str
    ] = set()

    for tool_name in (
        agent.tools
    ):

        if not isinstance(
            tool_name,
            str,
        ):

            raise ValueError(
                f"Agent '{agent.name}' "
                "contains a non-string capability."
            )

        normalized = (
            tool_name
            .strip()
        )

        if not normalized:

            raise ValueError(
                f"Agent '{agent.name}' "
                "contains an empty capability name."
            )

        if normalized in trusted_seen:

            raise ValueError(
                f"Agent '{agent.name}' "
                "contains duplicate capability "
                f"'{normalized}'."
            )

        trusted_seen.add(
            normalized
        )

        trusted_tools.append(
            normalized
        )

    # ============================================================
    # CURRENT-TURN SEMANTIC NARROWING
    # ============================================================

    if allowed_tools is None:

        selected_tools = (
            trusted_tools
        )

    else:

        if not isinstance(
            allowed_tools,
            list,
        ):

            raise ValueError(
                "allowed_tools must be a list."
            )

        requested_tools: list[
            str
        ] = []

        requested_seen: set[
            str
        ] = set()

        for tool_name in (
            allowed_tools
        ):

            if not isinstance(
                tool_name,
                str,
            ):

                raise ValueError(
                    "allowed_tools must contain "
                    "only strings."
                )

            normalized = (
                tool_name
                .strip()
            )

            if not normalized:

                raise ValueError(
                    "allowed_tools must not contain "
                    "empty capability names."
                )

            if normalized in requested_seen:

                continue

            requested_seen.add(
                normalized
            )

            requested_tools.append(
                normalized
            )

        unknown = [
            tool_name

            for tool_name
            in requested_tools

            if tool_name
            not in trusted_seen
        ]

        if unknown:

            raise ValueError(
                "Semantic capability narrowing "
                "contains capability outside agent "
                f"'{agent.name}' boundary: "
                + ", ".join(
                    sorted(
                        unknown
                    )
                )
            )

        requested_set = (
            set(
                requested_tools
            )
        )

        # Preserve trusted AgentDefinition ordering.
        #
        # Hub ordering is descriptive metadata and must not redefine
        # the specialist's canonical capability ordering.
        selected_tools = [
            tool_name

            for tool_name
            in trusted_tools

            if tool_name
            in requested_set
        ]

    # ============================================================
    # MODEL-FACING CATALOG
    # ============================================================

    return [
        build_capability_spec(
            tool_name,

            tool_lookup=(
                tool_lookup
            ),

            include_arguments=(
                include_arguments
            ),
        )

        for tool_name
        in selected_tools
    ]


def build_router_agent_spec(
    agent: AgentDefinition,
    *,
    tool_lookup: ToolLookup = get_tool,
) -> dict[
    str,
    Any,
]:
    """
    Build the Hub-facing representation of one specialist.

    The Hub receives capability names and descriptions.

    Detailed argument construction, including bounded enum values,
    remains specialist responsibility.
    """

    return {
        "name":
            agent.name,

        "description":
            agent.description,

        "capabilities":
            build_agent_capability_catalog(
                agent,

                tool_lookup=(
                    tool_lookup
                ),

                include_arguments=False,
            ),
    }