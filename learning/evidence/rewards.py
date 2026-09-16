from __future__ import annotations

from subagents.core.types import (
    HubResult,
)

from learning.evidence.types import (
    ExecutionReward,
)


def derive_execution_reward(
    result: HubResult,
) -> ExecutionReward:
    """
    Derive deterministic execution-level signals.

    This reward measures operational outcome only.

    It MUST NOT by itself be used as proof that the selected
    specialist, capability, arguments, or final answer were
    semantically correct.
    """

    components: dict[
        str,
        float,
    ] = {}

    if (
        result.status
        == "success"
    ):
        components[
            "hub_success"
        ] = 1.0

    elif (
        result.status
        == "partial_error"
    ):
        components[
            "hub_error"
        ] = -1.0

    elif (
        result.status
        == "approval_required"
    ):
        components[
            "approval_wait"
        ] = 0.0

    successful_specialists = sum(
        1

        for item
        in result.results

        if (
            item.status
            == "success"
        )
    )

    failed_specialists = sum(
        1

        for item
        in result.results

        if (
            item.status
            == "error"
        )
    )

    successful_tools = sum(
        1

        for item
        in result.results

        if (
            item.status
            == "success"
            and item.proposed_tool
            is not None
            and isinstance(
                item.tool_result,
                dict,
            )
        )
    )

    if successful_specialists:
        components[
            "specialist_success"
        ] = (
            0.5
            * successful_specialists
        )

    if successful_tools:
        components[
            "tool_execution_success"
        ] = (
            0.5
            * successful_tools
        )

    if failed_specialists:
        components[
            "specialist_error"
        ] = (
            -1.0
            * failed_specialists
        )

    total = (
        sum(
            components.values()
        )
    )

    return (
        ExecutionReward(
            components=(
                components
            ),

            total=(
                total
            ),

            # We have execution evidence, not semantic truth.
            quality_eligible=False,
        )
    )