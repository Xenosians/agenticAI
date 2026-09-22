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

    if not identifier.strip():

        return False

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


def _grounded_values(
    *,
    field_name: str,
    value: Any,
) -> tuple[
    list[str] | None,
    str | None,
]:
    """
    Normalize one grounded runtime argument.

    Supported forms:

        "jdoe"

        [
            "src/app.py",
            "tests/test_app.py",
        ]

    Lists are exact collections of grounded identifiers.

    Empty lists, non-string members, empty members, and duplicate
    members fail closed.
    """

    if isinstance(
        value,
        str,
    ):

        if not value.strip():

            return (
                None,
                f"{field_name} must not be empty.",
            )

        return (
            [
                value,
            ],
            None,
        )

    if isinstance(
        value,
        list,
    ):

        if not value:

            return (
                None,
                f"{field_name} must not be empty.",
            )

        normalized: list[str] = []

        seen: set[str] = set()

        for item in value:

            if not isinstance(
                item,
                str,
            ):

                return (
                    None,
                    (
                        f"{field_name} must contain "
                        "only strings."
                    ),
                )

            if not item.strip():

                return (
                    None,
                    (
                        f"{field_name} must not contain "
                        "empty values."
                    ),
                )

            if item in seen:

                return (
                    None,
                    (
                        f"{field_name} must not contain "
                        "duplicate values."
                    ),
                )

            seen.add(
                item
            )

            normalized.append(
                item
            )

        return (
            normalized,
            None,
        )

    return (
        None,
        (
            f"{field_name} must be a string "
            "or a list of strings."
        ),
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
    Verify model-produced semantic identifiers against the
    ORIGINAL USER REQUEST.

    Grounded values may be:

        exact scalar string

        exact list of strings

    For list-valued arguments EVERY member must independently
    appear in the original user request.

    This prevents:

        user asks:
            stage a.py and b.py

        specialist proposes:
            ["a.py", "b.py", "secret.py"]

    Missing grounded fields remain allowed here because requiredness
    belongs to capability policy/schema.
    """

    for field_name in (
        grounded_arguments
    ):

        if (
            field_name
            not in arguments
        ):

            continue

        value = (
            arguments[
                field_name
            ]
        )

        (
            values,
            error,
        ) = (
            _grounded_values(
                field_name=(
                    field_name
                ),

                value=(
                    value
                ),
            )
        )

        if values is None:

            return (
                False,
                error,
            )

        for identifier in values:

            if not (
                identifier_appears_in_request(
                    identifier,
                    user_input,
                )
            ):

                return (
                    False,
                    (
                        "The model produced "
                        f"{field_name} value "
                        f"'{identifier}', but that "
                        "identifier does not appear "
                        "exactly in the original request."
                    ),
                )

    return (
        True,
        None,
    )


def resolve_policy_execution_arguments(
    *,
    tool_name: str,
    tool: dict[
        str,
        Any,
    ],
    original_arguments: dict[
        str,
        Any,
    ],
    policy_result: dict[
        str,
        Any,
    ],
) -> tuple[
    dict[
        str,
        Any,
    ] | None,
    str | None,
]:
    """
    Resolve optional trusted policy-derived execution arguments.

    Purpose:

        Model / user arguments
            remain exact and immutable.

        Trusted policy
            may bind additional execution-state facts needed to
            make an approval exact and replay-safe.

    Example classes of trusted bound state:

        expected version
        expected branch
        expected commit
        expected upstream

    Policy may NEVER:
        - remove a model argument;
        - change a model argument;
        - add undeclared hidden arguments.

    Additional trusted arguments must be explicitly declared by the
    capability through:

        trusted_policy_arguments = [...]

    These arguments are not part of the model-facing parameter
    schema unless separately declared there.
    """

    resolved = (
        policy_result.get(
            "execution_arguments"
        )
    )

    if resolved is None:

        return (
            dict(
                original_arguments
            ),
            None,
        )

    if not isinstance(
        resolved,
        dict,
    ):

        return (
            None,
            (
                f"Capability '{tool_name}' policy returned "
                "invalid execution_arguments."
            ),
        )

    trusted_policy_arguments = (
        tool.get(
            "trusted_policy_arguments",
            [],
        )
    )

    if not isinstance(
        trusted_policy_arguments,
        list,
    ):

        return (
            None,
            (
                f"Capability '{tool_name}' has invalid "
                "trusted_policy_arguments metadata."
            ),
        )

    trusted_names: set[str] = set()

    for item in trusted_policy_arguments:

        if (
            not isinstance(
                item,
                str,
            )
            or not item.strip()
        ):

            return (
                None,
                (
                    f"Capability '{tool_name}' has invalid "
                    "trusted policy argument metadata."
                ),
            )

        name = (
            item.strip()
        )

        if name in trusted_names:

            return (
                None,
                (
                    f"Capability '{tool_name}' contains duplicate "
                    f"trusted policy argument '{name}'."
                ),
            )

        trusted_names.add(
            name
        )

    # --------------------------------------------------------
    # All keys must themselves be valid structured names.
    # --------------------------------------------------------

    for key in resolved:

        if (
            not isinstance(
                key,
                str,
            )
            or not key.strip()
        ):

            return (
                None,
                (
                    f"Capability '{tool_name}' policy returned "
                    "an invalid execution argument name."
                ),
            )

    # --------------------------------------------------------
    # Model/user-controlled arguments are immutable.
    # --------------------------------------------------------

    for (
        argument_name,
        original_value,
    ) in original_arguments.items():

        if (
            argument_name
            not in resolved
        ):

            return (
                None,
                (
                    f"Capability '{tool_name}' policy execution "
                    f"arguments removed original argument "
                    f"'{argument_name}'."
                ),
            )

        if (
            resolved[
                argument_name
            ]
            != original_value
        ):

            return (
                None,
                (
                    f"Capability '{tool_name}' policy execution "
                    f"arguments changed original argument "
                    f"'{argument_name}'."
                ),
            )

    # --------------------------------------------------------
    # Only explicitly trusted hidden arguments may be added.
    # --------------------------------------------------------

    added_names = (
        set(
            resolved.keys()
        )
        - set(
            original_arguments.keys()
        )
    )

    unexpected_added = (
        added_names
        - trusted_names
    )

    if unexpected_added:

        return (
            None,
            (
                f"Capability '{tool_name}' policy added "
                "undeclared trusted execution arguments: "
                + ", ".join(
                    sorted(
                        unexpected_added
                    )
                )
            ),
        )

    return (
        dict(
            resolved
        ),
        None,
    )


class ToolGateway:

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

        execution_arguments = (
            dict(
                arguments
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

            (
                execution_arguments,
                execution_arguments_error,
            ) = (
                resolve_policy_execution_arguments(
                    tool_name=(
                        tool_name
                    ),

                    tool=(
                        tool
                    ),

                    original_arguments=(
                        arguments
                    ),

                    policy_result=(
                        policy_result
                    ),
                )
            )

            if execution_arguments is None:

                return gateway_result(
                    ok=False,

                    status="error",

                    decision_code=(
                        "policy_execution_arguments_invalid"
                    ),

                    error=(
                        execution_arguments_error
                        or (
                            "Trusted policy returned invalid "
                            "execution arguments."
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
                    execution_arguments,
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

                approval_arguments=(
                    execution_arguments
                ),
            )

        # ========================================================
        # AUTO-APPROVED EXECUTION
        # ========================================================

        result = (
            await self.mcp.call_tool(
                tool_name,
                execution_arguments,
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
