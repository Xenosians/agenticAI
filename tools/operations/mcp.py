"""Platform operations tools."""
from mcp.server import MCPServer

from config import Settings


def register_operations_tools(server: MCPServer, settings: Settings) -> None:
    """Register platform operations tools."""
