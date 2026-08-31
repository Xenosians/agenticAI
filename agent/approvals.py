import uuid

from .mcp_client import mcp_runtime
from tools.registry import get_tool


APPROVALS: dict[str, dict] = {}


def create_approval(
    tool_name: str,
    arguments: dict,
) -> dict:
    tool = get_tool(tool_name)

    if tool is None:
        raise ValueError(
            f"Unknown tool: {tool_name}"
        )

    if not tool["requires_approval"]:
        raise ValueError(
            f"Tool '{tool_name}' "
            "does not require approval."
        )

    approval_id = uuid.uuid4().hex[:8]

    approval = {
        "id": approval_id,
        "tool": tool_name,
        "arguments": arguments,
        "risk": tool["risk"],
        "status": "pending",
        "result": None,
    }

    APPROVALS[approval_id] = approval

    return approval


def get_approval(
    approval_id: str,
) -> dict | None:
    return APPROVALS.get(
        approval_id
    )


def list_pending_approvals() -> list[dict]:
    return [
        approval
        for approval in APPROVALS.values()
        if approval["status"] == "pending"
    ]


async def approve_approval(
    approval_id: str,
) -> dict:
    approval = get_approval(
        approval_id
    )

    if approval is None:
        return {
            "ok": False,
            "error": (
                f"Approval '{approval_id}' "
                "not found."
            ),
        }

    if approval["status"] != "pending":
        return {
            "ok": False,
            "error": (
                f"Approval '{approval_id}' "
                f"is already {approval['status']}."
            ),
        }

    print(
        f"\n[MCP Mutation] "
        f"{approval['tool']} "
        f"{approval['arguments']}"
    )

    result = await mcp_runtime.call_tool(
        approval["tool"],
        approval["arguments"],
    )

    approval["result"] = result

    if (
        result.get("ok") is True
        and result.get("status") == "executed"
    ):
        approval["status"] = "approved"

    else:
        approval["status"] = "failed"

    return {
        "ok": (
            approval["status"]
            == "approved"
        ),
        "approval": approval,
        "result": result,
    }