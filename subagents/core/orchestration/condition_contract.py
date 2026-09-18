from __future__ import annotations

import math

from typing import (
    Any,
    Callable,
)

from subagents.core.definitions.types import (
    ResultCondition,
    SemanticIntent,
    SpecialistRequest,
)

from subagents.core.orchestration.intent_contract import (
    trusted_condition_fields,
    trusted_requires_approval,
    trusted_tool_effect,
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


def _valid_scalar(
    value: Any,
) -> bool:

    if value is None:
        return True

    if isinstance(
        value,
        bool,
    ):
        return True

    if isinstance(
        value,
        str,
    ):
        return True

    if (
        isinstance(
            value,
            int,
        )
        and not isinstance(
            value,
            bool,
        )
    ):
        return True

    if isinstance(
        value,
        float,
    ):
        return (
            math.isfinite(
                value
            )
        )

    return False


def parse_result_condition(
    value: Any,
    *,
    prior_delegations: list[
        SpecialistRequest
    ],
    target_intent: (
        SemanticIntent
        | None
    ),
    tool_lookup: ToolLookup = get_tool,
) -> (
    ResultCondition
    | None
):
    """
    Validate one optional result-dependent workflow condition.

    Current v1 contract:

        earlier unconditional trusted read
            ↓
        explicitly trusted scalar result field
            ↓
        one statically approval-gated mutation

    No expression language.
    No condition chains.
    No runbook interpretation.
    No dynamic-policy mutations.
    """

    if value is None:
        return None

    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            "when must be an object."
        )

    expected_keys = {
        "source_agent",
        "result_field",
        "equals",
    }

    if (
        set(
            value.keys()
        )
        != expected_keys
    ):

        raise ValueError(
            "when must contain exactly "
            "source_agent, result_field, and equals."
        )

    source_agent = (
        value.get(
            "source_agent"
        )
    )

    result_field = (
        value.get(
            "result_field"
        )
    )

    expected_value = (
        value.get(
            "equals"
        )
    )

    if (
        not isinstance(
            source_agent,
            str,
        )
        or not source_agent.strip()
    ):

        raise ValueError(
            "when.source_agent must be a "
            "non-empty string."
        )

    source_agent = (
        source_agent.strip()
    )

    if (
        not isinstance(
            result_field,
            str,
        )
        or not result_field.strip()
    ):

        raise ValueError(
            "when.result_field must be a "
            "non-empty string."
        )

    result_field = (
        result_field.strip()
    )

    if not _valid_scalar(
        expected_value
    ):

        raise ValueError(
            "when.equals must be a finite JSON scalar."
        )

    # --------------------------------------------------------
    # SOURCE
    #
    # Only an EARLIER unconditional delegation may become a
    # branch source.
    #
    # This prevents condition chains and loops.
    # --------------------------------------------------------

    source = next(
        (
            delegation

            for delegation
            in reversed(
                prior_delegations
            )

            if (
                delegation.agent_name
                == source_agent
                and delegation.condition
                is None
            )
        ),
        None,
    )

    if source is None:

        raise ValueError(
            "when.source_agent must reference an "
            "earlier unconditional delegation."
        )

    source_intent = (
        source.semantic_intent
    )

    if source_intent is None:

        raise ValueError(
            "Conditional workflow source is missing "
            "semantic intent."
        )

    if (
        source_intent.effect
        != "read"
    ):

        raise ValueError(
            "Conditional workflow source must be read-only."
        )

    if (
        len(
            source_intent.allowed_tools
        )
        != 1
    ):

        raise ValueError(
            "Conditional workflow source must resolve "
            "to exactly one read capability."
        )

    source_tool_name = (
        source_intent
        .allowed_tools[
            0
        ]
    )

    source_tool = (
        tool_lookup(
            source_tool_name
        )
    )

    if source_tool is None:

        raise ValueError(
            "Conditional workflow source capability "
            "has no trusted metadata."
        )

    if (
        trusted_tool_effect(
            source_tool
        )
        != "read"
    ):

        raise ValueError(
            "Conditional workflow source capability "
            "is not trusted read-only."
        )

    allowed_fields = (
        trusted_condition_fields(
            tool_name=(
                source_tool_name
            ),

            tool=(
                source_tool
            ),
        )
    )

    if (
        result_field
        not in allowed_fields
    ):

        raise ValueError(
            "Conditional workflow references a result "
            "field that is not trusted for branching."
        )

    # --------------------------------------------------------
    # TARGET
    # --------------------------------------------------------

    if target_intent is None:

        raise ValueError(
            "Conditional workflow target is missing "
            "semantic intent."
        )

    if (
        target_intent.effect
        != "mutation"
    ):

        raise ValueError(
            "Conditional workflow target must be "
            "a mutation."
        )

    if (
        len(
            target_intent.allowed_tools
        )
        != 1
    ):

        raise ValueError(
            "Conditional workflow target must resolve "
            "to exactly one mutation capability."
        )

    target_tool_name = (
        target_intent
        .allowed_tools[
            0
        ]
    )

    target_tool = (
        tool_lookup(
            target_tool_name
        )
    )

    if target_tool is None:

        raise ValueError(
            "Conditional workflow target capability "
            "has no trusted metadata."
        )

    if (
        trusted_tool_effect(
            target_tool
        )
        != "mutation"
    ):

        raise ValueError(
            "Conditional workflow target capability "
            "is not a mutation."
        )

    if (
        target_tool.get(
            "policy_resolver"
        )
        is not None
    ):

        raise ValueError(
            "Conditional workflow v1 does not permit "
            "dynamic-policy mutation capabilities."
        )

    if not (
        trusted_requires_approval(
            tool_name=(
                target_tool_name
            ),

            tool=(
                target_tool
            ),
        )
    ):

        raise ValueError(
            "Conditional workflow mutation must be "
            "statically approval-gated."
        )

    return (
        ResultCondition(
            source_agent=(
                source_agent
            ),

            result_field=(
                result_field
            ),

            equals=(
                expected_value
            ),
        )
    )
