"""Register fixed Proxmox reads with the MCP server."""
from typing import Any

from pydantic import BaseModel

from services.proxmox import ProxmoxService


class ProxmoxReadResult(BaseModel):
    ok: bool
    status: str
    provider: str = "proxmox"
    observed_at: str | None = None
    data: dict[str, Any] | None = None


def register_proxmox_tools(server, service: ProxmoxService | None) -> None:
    def require_service() -> ProxmoxService:
        if service is None:
            raise RuntimeError("Proxmox VE integration is not configured.")
        return service

    @server.tool()
    def proxmox_cluster_status() -> ProxmoxReadResult:
        """Read Proxmox VE cluster membership and reported online state."""
        return ProxmoxReadResult(**require_service().cluster_status())

    @server.tool()
    def proxmox_nodes() -> ProxmoxReadResult:
        """List Proxmox VE nodes with reported resource counters."""
        return ProxmoxReadResult(**require_service().nodes())

    @server.tool()
    def proxmox_guests() -> ProxmoxReadResult:
        """List QEMU virtual machines and LXC containers with status and resource counters."""
        return ProxmoxReadResult(**require_service().guests())

    @server.tool()
    def proxmox_node_storage(node: str) -> ProxmoxReadResult:
        """Read storage pools for one exact Proxmox node name."""
        return ProxmoxReadResult(**require_service().node_storage(node))

    @server.tool()
    def proxmox_recent_tasks(node: str) -> ProxmoxReadResult:
        """Read recent tasks for one exact Proxmox node name."""
        return ProxmoxReadResult(**require_service().recent_tasks(node))
