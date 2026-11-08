"""Register fixed Proxmox reads with the MCP server."""
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from services.proxmox import ProxmoxService
from tools.provider_reads import not_configured, provider_read

PROVIDER = "proxmox"
LABEL = "Proxmox VE"


class ProxmoxReadResult(BaseModel):
    ok: bool
    status: str
    provider: str = PROVIDER
    observed_at: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None


def register_proxmox_tools(server, service: ProxmoxService | None, missing_settings: Iterable[str] = ()) -> None:
    """Each read returns observed data, or a typed failure naming the cause (see tools.provider_reads)."""
    missing = tuple(missing_settings)

    def proxmox(call) -> ProxmoxReadResult:
        if service is None:
            return ProxmoxReadResult(**not_configured(PROVIDER, LABEL, missing))
        return ProxmoxReadResult(**provider_read(PROVIDER, LABEL, call))

    @server.tool()
    def proxmox_cluster_status() -> ProxmoxReadResult:
        """Read Proxmox VE cluster membership and reported online state."""
        return proxmox(lambda: service.cluster_status())

    @server.tool()
    def proxmox_nodes() -> ProxmoxReadResult:
        """List Proxmox VE nodes with reported resource counters."""
        return proxmox(lambda: service.nodes())

    @server.tool()
    def proxmox_guests() -> ProxmoxReadResult:
        """List QEMU virtual machines and LXC containers with status and resource counters."""
        return proxmox(lambda: service.guests())

    @server.tool()
    def proxmox_node_storage(node: str) -> ProxmoxReadResult:
        """Read storage pools for one exact Proxmox node name."""
        return proxmox(lambda: service.node_storage(node))

    @server.tool()
    def proxmox_recent_tasks(node: str) -> ProxmoxReadResult:
        """Read recent tasks for one exact Proxmox node name."""
        return proxmox(lambda: service.recent_tasks(node))
