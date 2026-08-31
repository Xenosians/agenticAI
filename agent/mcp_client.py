import asyncio

from mcp import Client

from mcp_server import mcp


async def _call_mcp_tool_async(
    tool_name: str,
    arguments: dict,
) -> dict:
    async with Client(
        mcp,
        raise_exceptions=True,
    ) as client:
        result = await client.call_tool(
            tool_name,
            arguments,
        )

        if result.is_error:
            return {
                "ok": False,
                "status": "error",
                "error": f"MCP tool '{tool_name}' returned an error.",
            }

        if result.structured_content is not None:
            return result.structured_content

        return {
            "ok": False,
            "status": "error",
            "error": (
                f"MCP tool '{tool_name}' returned "
                "no structured content."
            ),
        }


def call_mcp_tool(
    tool_name: str,
    arguments: dict,
) -> dict:
    """
    Synchronous wrapper for the CLI-based agent.

    Later, when the application becomes fully async
    with FastAPI, we can remove asyncio.run().
    """
    try:
        return asyncio.run(
            _call_mcp_tool_async(
                tool_name,
                arguments,
            )
        )

    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "error": f"MCP client failed: {exc}",
        }