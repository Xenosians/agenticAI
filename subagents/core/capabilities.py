from __future__ import annotations

from copy import (
    deepcopy,
)

from typing import (
    Any,
    Callable,
)

from subagents.core.types import (
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
) -> list[
    dict[
        str,
        Any,
    ]
]:
    """
    Build the trusted model-facing capability catalog for one
    specialist.

    Only capabilities explicitly present in AgentDefinition.tools
    are exposed.
    """

    capabilities: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen: set[
        str
    ] = set()

    for tool_name in (
        agent.tools
    ):

        normalized = (
            tool_name.strip()
        )

        if not normalized:

            raise ValueError(
                f"Agent '{agent.name}' "
                "contains an empty "
                "capability name."
            )

        if normalized in seen:

            raise ValueError(
                f"Agent '{agent.name}' "
                "contains duplicate capability "
                f"'{normalized}'."
            )

        seen.add(
            normalized
        )

        capabilities.append(
            build_capability_spec(
                normalized,

                tool_lookup=(
                    tool_lookup
                ),

                include_arguments=(
                    include_arguments
                ),
            )
        )

    return capabilities


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