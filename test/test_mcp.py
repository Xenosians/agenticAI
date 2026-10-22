"""Manual configured read-only smoke check; printed output is not acceptance."""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp import Client

from mcp_server import create_mcp_server


async def main(user_id: str, resource: str):
    async with Client(
        create_mcp_server(),
        raise_exceptions=True,
    ) as client:

        tools = await client.list_tools()

        print("Available MCP tools:")

        for tool in tools.tools:
            print(f"- {tool.name}")

        print()

        # ---------------------------------------------
        # ACCOUNT STATUS
        # ---------------------------------------------

        result = await client.call_tool(
            "account_status",
            {
                "user_id": user_id,
            },
        )

        print("account_status:")
        print("is_error:", result.is_error)
        print(
            "structured_content:",
            result.structured_content,
        )
        print(
            "content:",
            result.content,
        )

        print()

        # ---------------------------------------------
        # ACCESS CHECK
        # ---------------------------------------------

        result = await client.call_tool(
            "check_access",
            {
                "user_id": user_id,
                "resource": resource,
            },
        )

        print("check_access:")
        print("is_error:", result.is_error)
        print(
            "structured_content:",
            result.structured_content,
        )
        print(
            "content:",
            result.content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True, help="Exact account to inspect; no mutation is performed")
    parser.add_argument("--resource", required=True, help="Exact configured resource whose access is checked")
    args = parser.parse_args()
    asyncio.run(main(args.user_id, args.resource))
