import asyncio
import sys
from pathlib import Path

from mcp import (
    Client,
    StdioServerParameters,
)


# Current file:
# itsm-agent/test/test_mcp_studio.py
#
# parents[0] = itsm-agent/test
# parents[1] = itsm-agent
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server.py"


async def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"MCP server:   {MCP_SERVER_PATH}")
    print()

    if not MCP_SERVER_PATH.exists():
        raise FileNotFoundError(
            f"MCP server not found: {MCP_SERVER_PATH}"
        )

    server = StdioServerParameters(
        command=sys.executable,
        args=[
            str(MCP_SERVER_PATH),
        ],
    )

    async with Client(server) as client:

        # -------------------------------------------------
        # TOOL DISCOVERY
        # -------------------------------------------------

        tools = await client.list_tools()

        print("Available MCP tools:")

        for tool in tools.tools:
            print(f"- {tool.name}")

        print()

        # -------------------------------------------------
        # ACCOUNT STATUS
        # -------------------------------------------------

        result = await client.call_tool(
            "account_status",
            {
                "user_id": "asmith",
            },
        )

        print("account_status:")
        print(result.structured_content)

        print()

        # -------------------------------------------------
        # ACCESS CHECK
        # -------------------------------------------------

        result = await client.call_tool(
            "check_access",
            {
                "user_id": "jdoe",
                "resource": "VPN access",
            },
        )

        print("check_access:")
        print(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())