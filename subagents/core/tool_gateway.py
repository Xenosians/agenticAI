import re

from typing import (
    Any,
    Callable,
)

from agent.approvals import (
    create_approval,
)

from agent.mcp_client import (
    mcp_runtime,
)

from tools.registry import (
    get_tool,
)

from subagents.core.types import (
    AgentDefinition,
)


VALID_RISKS = {
    "read",
    "low",
    "medium",
    "high",
}


def identifier_appears_in_request(
    identifier: str,
    user_input: str,
) -> bool:
    """
    Check whether an identifier appears literally in the
    original user request without accepting partial identifiers.

    Examples that should match:
        Read tools/workspace.py.
        Read "README.md".
        Unlock jdoe.

    Examples that should not match:
        identifier: README.md
        request: Read backup/README.md

        identifier: README.md
        request: Read README.md.bak

        identifier: jdoe
        request: Check jdoe-admin
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
    arguments: dict[str, Any],
    grounded_arguments: list[str],
) -> tuple[
    bool,
    str | None,
]:
    """
    Verify model-produced identifiers that policy declares must
    originate literally from the user's request.

    Which arguments require grounding is tool metadata, not
    hardcoded gateway knowledge.
    """

    for field_name in grounded_arguments:
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
                    f"The model produced "
                    f"{field_name} '{value}', "
                    "but that identifier does "
                    "not appear exactly in the "
                    "original request."
                ),
            )

    return True, None


class ToolGateway:
    """
    Security boundary between sub-agents and executable tools.

    Workers may propose operations.

    This gateway decides whether those operations are allowed.

    Tool metadata may declare:
    - arguments that must be grounded in the user request
    - a deterministic policy resolver
    - static risk and approval requirements
    """

    def __init__(
        self,
        tool_lookup: Callable = get_tool,
        approval_creator: Callable = (
            create_approval
        ),
        mcp=mcp_runtime,
    ) -> None:
        self.tool_lookup = (
            tool_lookup
        )

        self.approval_creator = (
            approval_creator
        )

        self.mcp = mcp

    async def execute(
        self,
        agent: AgentDefinition,
        user_input: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:

        # -----------------------------------------
        # AGENT TOOL PERMISSION
        # -----------------------------------------

        if tool_name not in agent.tools:
            return {
                "ok": False,
                "status": "denied",

                "error": (
                    f"Agent '{agent.name}' "
                    "is not allowed to use "
                    f"tool '{tool_name}'."
                ),
            }

        # -----------------------------------------
        # TOOL EXISTENCE
        # -----------------------------------------

        tool = self.tool_lookup(
            tool_name
        )

        if tool is None:
            return {
                "ok": False,
                "status": "error",

                "error": (
                    "Unknown tool requested: "
                    f"{tool_name}"
                ),
            }

        # -----------------------------------------
        # USER-REQUEST GROUNDING
        # -----------------------------------------

        grounded_arguments = tool.get(
            "grounded_arguments",
            [],
        )

        if not isinstance(
            grounded_arguments,
            list,
        ):
            return {
                "ok": False,
                "status": "error",

                "error": (
                    "Tool grounding policy "
                    "is invalid."
                ),
            }

        if not all(
            isinstance(
                field_name,
                str,
            )
            for field_name
            in grounded_arguments
        ):
            return {
                "ok": False,
                "status": "error",

                "error": (
                    "Tool grounding fields "
                    "are invalid."
                ),
            }

        (
            valid,
            validation_error,
        ) = validate_grounded_arguments(
            user_input=user_input,
            arguments=arguments,
            grounded_arguments=(
                grounded_arguments
            ),
        )

        if not valid:
            return {
                "ok": False,
                "status": "denied",
                "error": validation_error,
            }

        # -----------------------------------------
        # EFFECTIVE TOOL POLICY
        # -----------------------------------------

        effective_risk = tool.get(
            "risk"
        )

        requires_approval = tool.get(
            "requires_approval"
        )

        policy_resolver = tool.get(
            "policy_resolver"
        )

        if policy_resolver is not None:
            try:
                policy_result = (
                    policy_resolver(
                        **arguments
                    )
                )

            except TypeError as exc:
                return {
                    "ok": False,
                    "status": "denied",

                    "error": (
                        "Invalid arguments for "
                        f"'{tool_name}' policy: "
                        f"{exc}"
                    ),
                }

            except Exception as exc:
                return {
                    "ok": False,
                    "status": "error",

                    "error": (
                        "Tool policy evaluation "
                        f"failed: {exc}"
                    ),
                }

            if not isinstance(
                policy_result,
                dict,
            ):
                return {
                    "ok": False,
                    "status": "error",

                    "error": (
                        "Tool policy returned an "
                        "invalid structured result."
                    ),
                }

            if not policy_result.get(
                "ok",
                False,
            ):
                return {
                    "ok": False,

                    "status": (
                        policy_result.get(
                            "status",
                            "denied",
                        )
                    ),

                    "tool": tool_name,

                    "error": (
                        policy_result.get(
                            "error",
                            "Tool policy denied "
                            "the operation.",
                        )
                    ),
                }

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
            return {
                "ok": False,
                "status": "error",

                "error": (
                    "Tool policy returned an "
                    "invalid risk classification."
                ),
            }

        if not isinstance(
            requires_approval,
            bool,
        ):
            return {
                "ok": False,
                "status": "error",

                "error": (
                    "Tool policy returned an "
                    "invalid approval decision."
                ),
            }

        # -----------------------------------------
        # APPROVAL REQUIRED
        # -----------------------------------------

        if requires_approval:
            approval = (
                self.approval_creator(
                    tool_name,
                    arguments,
                    risk=effective_risk,
                )
            )

            return {
                "ok": True,

                "status": (
                    "approval_required"
                ),

                "tool": tool_name,
                "risk": effective_risk,

                "approval_id": (
                    approval["id"]
                ),
            }

        # -----------------------------------------
        # AUTO-APPROVED OPERATION
        # -----------------------------------------

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

            if result_status == "denied":
                failure_status = "denied"
            else:
                failure_status = "error"

            return {
                "ok": False,
                "status": failure_status,

                "tool": tool_name,

                "error": (
                    result.get(
                        "error",
                        (
                            "MCP tool execution "
                            "failed."
                        ),
                    )
                ),
            }

        return {
            "ok": True,
            "status": "success",
            "tool": tool_name,
            "result": result,
        }