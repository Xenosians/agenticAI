import re

from typing import (
    Any,
    Callable,
)

from tools.registry import (
    get_tool,
)

from subagents.core.definitions.types import (
    AgentDefinition,
)


VALID_RISKS = {
    "read",
    "low",
    "medium",
    "high",
}


def gateway_result(
    *,
    ok: bool,
    status: str,
    decision_code: str,
    **extra: Any,
) -> dict[
    str,
    Any,
]:
    """
    Build one stable ToolGateway result.

    decision_code is machine-readable and must remain independent
    from human-facing error prose.
    """

    return {
        "ok":
            ok,

        "status":
            status,

        "decision_code":
            decision_code,

        **extra,
    }


def identifier_appears_in_request(
    identifier: str,
    user_input: str,
) -> bool:
    """
    Check whether an identifier appears literally in the
    original user request without accepting partial identifiers.
    """

    identifier_chars = (
        r"A-Za-z0-9._/\\-"
    )

    pattern = (
        rf"(?<![{identifier_chars}])"
        rf"{re.escape(identifier)}"
        rf"(?="
        rf"$"
        rf"|[\s,;:!?()\[\]{{}}\"'`]"
        rf"|\.(?=\s|$)"
        rf")"
    )

    return (
        re.search(
            pattern,
            user_input,
            re.IGNORECASE,
        )
        is not None
    )


def validate_grounded_arguments(
    user_input: str,
    arguments: dict[
        str,
        Any,
    ],
    grounded_arguments: list[
        str
    ],
) -> tuple[
    bool,
    str | None,
]:
    """
    Verify model-produced identifiers that trusted tool policy
    declares must originate literally from the user request.
    """

    for field_name in (
        grounded_arguments
    ):
        value = arguments.get(
            field_name
        )

        if value is None:
            continue

        if not isinstance(
            value,
            str,
        ):
            return (
                False,
                (
                    f"{field_name} "
                    "must be a string."
                ),
            )

        if not identifier_appears_in_request(
            value,
            user_input,
        ):
            return (
                False,
                (
                    "The model produced "
                    f"{field_name} '{value}', "
                    "but that identifier does "
                    "not appear exactly in the "
                    "original request."
                ),
            )

    return (
        True,
        None,
    )


class ToolGateway:
    """
    Deterministic security boundary between model proposals and
    executable capabilities.

    Stable machine-readable decision codes are emitted alongside
    human-readable errors so learning infrastructure can classify
    outcomes without parsing prose.
    """

    def __init__(
        self,
        *,
        approval_creator: Callable,
        mcp,
        tool_lookup: Callable = (
            get_tool
        ),
    ) -> None:

        self.tool_lookup = (
            tool_lookup
        )

        self.approval_creator = (
            approval_creator
        )

        self.mcp = (
            mcp
        )

    async def execute(
        self,
        agent: AgentDefinition,
        user_input: str,
        tool_name: str,
        arguments: dict[
            str,
            Any,
        ],
    ) -> dict[
        str,
        Any,
    ]:

        # ========================================================
        # AGENT CAPABILITY PERMISSION
        # ========================================================

        if (
            tool_name
            not in agent.tools
        ):
            return gateway_result(
                ok=False,
                status="denied",
                decision_code=(
                    "agent_tool_not_allowed"
                ),
                error=(
                    f"Agent '{agent.name}' "
                    "is not allowed to use "
                    f"tool '{tool_name}'."
                ),
            )

        # ========================================================
        # TOOL EXISTENCE
        # ========================================================

        tool = (
            self.tool_lookup(
                tool_name
            )
        )

        if tool is None:
            return gateway_result(
                ok=False,
                status="error",
                decision_code=(
                    "unknown_tool"
                ),
                error=(
                    "Unknown tool requested: "
                    f"{tool_name}"
                ),
            )

        # ========================================================
        # USER-REQUEST GROUNDING
        # ========================================================

        grounded_arguments = (
            tool.get(
                "grounded_arguments",
                [],
            )
        )

        if not isinstance(
            grounded_arguments,
            list,
        ):
            return gateway_result(
                ok=False,
                status="error",
                decision_code=(
                    "grounding_policy_invalid"
                ),
                error=(
                    "Tool grounding policy "
                    "is invalid."
                ),
            )

        if not all(
            isinstance(
                field_name,
                str,
            )

            for field_name
            in grounded_arguments
        ):
            return gateway_result(
                ok=False,
                status="error",
                decision_code=(
                    "grounding_field_invalid"
                ),
                error=(
                    "Tool grounding fields "
                    "are invalid."
                ),
            )

        (
            valid,
            validation_error,
        ) = (
            validate_grounded_arguments(
                user_input=(
                    user_input
                ),

                arguments=(
                    arguments
                ),

                grounded_arguments=(
                    grounded_arguments
                ),
            )
        )

        if not valid:
            return gateway_result(
                ok=False,
                status="denied",
                decision_code=(
                    "grounding_failed"
                ),
                error=(
                    validation_error
                ),
            )

        # ========================================================
        # EFFECTIVE TRUSTED POLICY
        # ========================================================

        effective_risk = (
            tool.get(
                "risk"
            )
        )

        requires_approval = (
            tool.get(
                "requires_approval"
            )
        )

        policy_resolver = (
            tool.get(
                "policy_resolver"
            )
        )

        if (
            policy_resolver
            is not None
        ):
            try:
                policy_result = (
                    policy_resolver(
                        **arguments
                    )
                )

            except TypeError as exc:
                return gateway_result(
                    ok=False,
                    status="denied",
                    decision_code=(
                        "policy_arguments_invalid"
                    ),
                    error=(
                        "Invalid arguments for "
                        f"'{tool_name}' policy: "
                        f"{exc}"
                    ),
                )

            except Exception as exc:
                return gateway_result(
                    ok=False,
                    status="error",
                    decision_code=(
                        "policy_evaluation_error"
                    ),
                    error=(
                        "Tool policy evaluation "
                        f"failed: {exc}"
                    ),
                )

            if not isinstance(
                policy_result,
                dict,
            ):
                return gateway_result(
                    ok=False,
                    status="error",
                    decision_code=(
                        "policy_result_invalid"
                    ),
                    error=(
                        "Tool policy returned an "
                        "invalid structured result."
                    ),
                )

            if not policy_result.get(
                "ok",
                False,
            ):
                return gateway_result(
                    ok=False,
                    status=(
                        policy_result.get(
                            "status",
                            "denied",
                        )
                    ),
                    decision_code=(
                        "policy_denied"
                    ),
                    tool=(
                        tool_name
                    ),
                    error=(
                        policy_result.get(
                            "error",
                            (
                                "Tool policy denied "
                                "the operation."
                            ),
                        )
                    ),
                )

            effective_risk = (
                policy_result.get(
                    "risk"
                )
            )

            requires_approval = (
                policy_result.get(
                    "requires_approval"
                )
            )

        if (
            effective_risk
            not in VALID_RISKS
        ):
            return gateway_result(
                ok=False,
                status="error",
                decision_code=(
                    "risk_invalid"
                ),
                error=(
                    "Tool policy returned an "
                    "invalid risk classification."
                ),
            )

        if not isinstance(
            requires_approval,
            bool,
        ):
            return gateway_result(
                ok=False,
                status="error",
                decision_code=(
                    "approval_decision_invalid"
                ),
                error=(
                    "Tool policy returned an "
                    "invalid approval decision."
                ),
            )

        # ========================================================
        # APPROVAL REQUIRED
        # ========================================================

        if requires_approval:
            approval = (
                self.approval_creator(
                    tool_name,
                    arguments,
                    risk=(
                        effective_risk
                    ),
                )
            )

            return gateway_result(
                ok=True,
                status=(
                    "approval_required"
                ),
                decision_code=(
                    "approval_required"
                ),
                tool=(
                    tool_name
                ),
                risk=(
                    effective_risk
                ),
                approval_id=(
                    approval[
                        "id"
                    ]
                ),
            )

        # ========================================================
        # AUTO-APPROVED EXECUTION
        # ========================================================

        result = (
            await self.mcp.call_tool(
                tool_name,
                arguments,
            )
        )

        if not result.get(
            "ok",
            False,
        ):
            result_status = (
                result.get(
                    "status"
                )
            )

            if (
                result_status
                == "denied"
            ):
                failure_status = (
                    "denied"
                )

                decision_code = (
                    "tool_execution_denied"
                )

            else:
                failure_status = (
                    "error"
                )

                decision_code = (
                    "tool_execution_error"
                )

            return gateway_result(
                ok=False,
                status=(
                    failure_status
                ),
                decision_code=(
                    decision_code
                ),
                tool=(
                    tool_name
                ),
                error=(
                    result.get(
                        "error",
                        (
                            "MCP tool execution "
                            "failed."
                        ),
                    )
                ),
            )

        return gateway_result(
            ok=True,
            status="success",
            decision_code="success",
            tool=(
                tool_name
            ),
            result=(
                result
            ),
        )