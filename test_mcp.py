import asyncio

from mcp import Client

from mcp_server import mcp


async def main():
    async with Client(
        mcp,
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
                "user_id": "asmith",
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
                "user_id": "jdoe",
                "resource": "vpn",
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
    asyncio.run(main())