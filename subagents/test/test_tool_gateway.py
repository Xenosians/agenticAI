import asyncio
import re

from subagents.core.tool_gateway import (
    ToolGateway,
    identifier_appears_in_request,
)
from subagents.core.types import AgentDefinition



ACCOUNT_AGENT = AgentDefinition(
    name="account-specialist",
    description="Account worker",
    tools=[
        "account_status",
        "unlock_user",
        "reset_password",
    ],
    model="test-model",
)


TOOLS = {
    "account_status": {
        "risk": "low",
        "requires_approval": False,
    },
    "unlock_user": {
        "risk": "high",
        "requires_approval": True,
    },
}


def fake_get_tool(name):
    return TOOLS.get(name)


def fake_create_approval(
    tool_name,
    arguments,
):
    return {
        "id": "approval-001",
        "risk": "high",
        "tool": tool_name,
        "arguments": arguments,
    }


class FakeMCP:
    def __init__(self):
        self.calls = []

    async def call_tool(
        self,
        tool_name,
        arguments,
    ):
        self.calls.append(
            (tool_name, arguments)
        )

        return {
            "ok": True,
            "user_id": arguments.get("user_id"),
            "enabled": True,
            "locked": False,
        }


def build_gateway():
    mcp = FakeMCP()

    gateway = ToolGateway(
        tool_lookup=fake_get_tool,
        approval_creator=fake_create_approval,
        mcp=mcp,
    )

    return gateway, mcp


def test_identifier_guard_accepts_exact_identifier():
    assert identifier_appears_in_request(
        "jdoe",
        "Is jdoe locked?",
    )


def test_identifier_guard_rejects_invented_identifier():
    assert not identifier_appears_in_request(
        "jsmith",
        "Is jdoe locked?",
    )


def test_agent_cannot_use_unlisted_tool():
    gateway, mcp = build_gateway()

    result = asyncio.run(
        gateway.execute(
            agent=ACCOUNT_AGENT,
            user_input="Do something",
            tool_name="check_access",
            arguments={},
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"

    assert mcp.calls == []


def test_identifier_mismatch_blocks_execution():
    gateway, mcp = build_gateway()

    result = asyncio.run(
        gateway.execute(
            agent=ACCOUNT_AGENT,
            user_input="Is jdoe locked?",
            tool_name="account_status",
            arguments={
                "user_id": "jsmith",
            },
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"

    assert mcp.calls == []


def test_read_operation_reaches_mcp():
    gateway, mcp = build_gateway()

    result = asyncio.run(
        gateway.execute(
            agent=ACCOUNT_AGENT,
            user_input="Is jdoe locked?",
            tool_name="account_status",
            arguments={
                "user_id": "jdoe",
            },
        )
    )

    assert result["ok"] is True
    assert result["status"] == "success"

    assert mcp.calls == [
        (
            "account_status",
            {
                "user_id": "jdoe",
            },
        )
    ]


def test_mutation_requires_approval_without_mcp():
    gateway, mcp = build_gateway()

    result = asyncio.run(
        gateway.execute(
            agent=ACCOUNT_AGENT,
            user_input="Unlock jdoe",
            tool_name="unlock_user",
            arguments={
                "user_id": "jdoe",
            },
        )
    )

    assert result["ok"] is True
    assert result["status"] == "approval_required"
    assert result["approval_id"] == "approval-001"

    # Critical security assertion:
    # mutation must NOT execute yet.
    assert mcp.calls == []