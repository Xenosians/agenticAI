import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server.py"


class MCPRuntime:
    def __init__(self):
        self._context = None
        self._client = None

    async def start(self):
        if self._client is not None:
            return

        server = StdioServerParameters(
            command=sys.executable,
            args=[str(MCP_SERVER_PATH)],
        )

        self._context = Client(server)

        self._client = await self._context.__aenter__()

        print("[MCP] Persistent server started.")

    async def stop(self):
        if self._context is None:
            return

        await self._context.__aexit__(
            None,
            None,
            None,
        )

        self._context = None
        self._client = None

        print("[MCP] Server stopped.")

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        if self._client is None:
            raise RuntimeError(
                "MCP runtime has not been started."
            )

        try:
            result = await self._client.call_tool(
                tool_name,
                arguments,
            )

        except Exception as exc:
            return {
                "ok": False,
                "status": "error",
                "error": f"MCP call failed: {exc}",
            }

        if result.is_error:
            return {
                "ok": False,
                "status": "error",
                "error": (
                    f"MCP tool '{tool_name}' "
                    "returned an error."
                ),
            }

        if result.structured_content is None:
            return {
                "ok": False,
                "status": "error",
                "error": (
                    f"MCP tool '{tool_name}' returned "
                    "no structured content."
                ),
            }

        return result.structured_content


mcp_runtime = MCPRuntime()