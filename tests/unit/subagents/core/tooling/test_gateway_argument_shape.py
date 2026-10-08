import asyncio
import pytest
from subagents.core.definitions.types import AgentDefinition
from subagents.core.tooling.gateway import ToolGateway


@pytest.mark.parametrize("arguments", [
    {"device_id":"0x12345678", "firmware_version":"v1.99"},
    {"expected_revision":"invented"},
    [],
])
def test_invalid_no_argument_proposal_never_reaches_policy_or_provider(arguments):
    class MCP:
        async def call_tool(self, *args):
            pytest.fail("Invalid model proposal reached provider")
    def approval(*args, **kwargs):
        pytest.fail("Invalid model proposal created approval")
    gateway = ToolGateway(approval_creator=approval, mcp=MCP())
    agent = AgentDefinition(name="networking-specialist", description="Read networking", model="test", tools=["network_system_info"])
    result = asyncio.run(gateway.execute(agent, "Read router firmware.", "network_system_info", arguments))
    assert result["ok"] is False
    assert result["decision_code"] in {"unknown_arguments", "invalid_argument_shape"}
