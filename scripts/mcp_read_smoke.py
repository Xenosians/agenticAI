"""Manual read-only MCP smoke check against this machine's configured providers.

Lists the registered tools and calls two read tools for exact, operator-chosen
inputs, either in process (default) or through the stdio server the AI service
starts (--stdio). Nothing is changed. Printed output is a manual observation,
not acceptance evidence. Run from the repository root:

    python scripts/mcp_read_smoke.py --user-id <account> --resource <resource> [--stdio]
"""
import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp import Client, StdioServerParameters  # noqa: E402


def _server(stdio: bool):
    if stdio:
        return StdioServerParameters(command=sys.executable, args=[str(PROJECT_ROOT / "mcp_server.py")])
    from mcp_server import create_mcp_server
    return create_mcp_server()


async def main(user_id: str, resource: str, stdio: bool) -> None:
    async with Client(_server(stdio), raise_exceptions=not stdio) as client:
        tools = await client.list_tools()
        print("Registered MCP tools:", len(tools.tools))
        for tool in tools.tools:
            print(f"- {tool.name}")
        for name, arguments in (("account_status", {"user_id": user_id}),
                                ("check_access", {"user_id": user_id, "resource": resource})):
            result = await client.call_tool(name, arguments)
            print()
            print(f"{name}: is_error={result.is_error}")
            print("structured_content:", result.structured_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-id", required=True, help="Exact account to inspect; nothing is changed")
    parser.add_argument("--resource", required=True, help="Exact configured resource whose access is checked")
    parser.add_argument("--stdio", action="store_true", help="Use the stdio server process instead of in-process")
    args = parser.parse_args()
    asyncio.run(main(args.user_id, args.resource, args.stdio))
