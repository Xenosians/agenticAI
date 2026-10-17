"""Jira Service Management tools."""
from mcp.server import MCPServer

from config import Settings


def register_service_desk_tools(server: MCPServer, settings: Settings) -> None:
    """Register service desk tools."""
