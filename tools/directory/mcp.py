"""Directory read tools."""
from mcp.server import MCPServer

from services.directory import DirectoryService


def register_directory_tools(server: MCPServer, directory: DirectoryService) -> None:
    """Register directory read tools."""
