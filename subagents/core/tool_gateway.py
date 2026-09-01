import re
from typing import Any, Callable

from agent.approvals import create_approval
from agent.mcp_client import mcp_runtime
from tools.registry import get_tool

from subagents.core.types import AgentDefinition



def identifier_appears_in_request(
    identifier: str,
    user_input: str,
) -> bool:
    pattern = (
        rf"(?<![A-Za-z0-9._-])"
        rf"{re.escape(identifier)}"
        rf"(?![A-Za-z0-9._-])"
    )

    return (
        re.search(
            pattern,
            user_input,
            re.IGNORECASE,
        )
        is not None
    )
    
def validate_identifiers(
    user_input: str,
    arguments: dict[str, Any],
) -> tuple[bool, str | None]:
    user_id = arguments.get("user_id")

    if user_id is None:
        return True, None

    if not isinstance(user_id, str):
        return False, "user_id must be a string."

    if not identifier_appears_in_request(
        user_id,
        user_input,
    ):
        return (
            False,
            (
                f"The model produced user_id '{user_id}', "
                "but that identifier does not appear exactly "
                "in the original request."
            ),
        )

    return True, None

class ToolGateway:
    """
    Security boundary between sub-agents and executable tools.

    Workers may propose operations.
    This gateway decides whether those operations are allowed.
    """

    def __init__(
        self,
        tool_lookup: Callable = get_tool,
        approval_creator: Callable = create_approval,
        mcp=mcp_runtime,
    ) -> None:
        self.tool_lookup = tool_lookup
        self.approval_creator = approval_creator
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
                    f"Agent '{agent.name}' is not allowed "
                    f"to use tool '{tool_name}'."
                ),
            }

        # -----------------------------------------
        # TOOL EXISTENCE
        # -----------------------------------------

        tool = self.tool_lookup(tool_name)

        if tool is None:
            return {
                "ok": False,
                "status": "error",
                "error": (
                    f"Unknown tool requested: {tool_name}"
                ),
            }

        # -----------------------------------------
        # IDENTIFIER GUARD
        # -----------------------------------------

        valid, validation_error = validate_identifiers(
            user_input,
            arguments,
        )

        if not valid:
            return {
                "ok": False,
                "status": "denied",
                "error": validation_error,
            }

        # -----------------------------------------
        # MUTATING OPERATION
        # -----------------------------------------

        if tool["requires_approval"]:
            approval = self.approval_creator(
                tool_name,
                arguments,
            )

            return {
                "ok": True,
                "status": "approval_required",
                "tool": tool_name,
                "risk": approval["risk"],
                "approval_id": approval["id"],
            }

        # -----------------------------------------
        # READ OPERATION
        # -----------------------------------------

        result = await self.mcp.call_tool(
            tool_name,
            arguments,
        )

        if not result.get("ok", False):
            return {
                "ok": False,
                "status": "error",
                "tool": tool_name,
                "error": result.get(
                    "error",
                    "MCP tool execution failed.",
                ),
            }

        return {
            "ok": True,
            "status": "success",
            "tool": tool_name,
            "result": result,
        }
    