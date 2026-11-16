"""Register the fixed Proxmox reads and the four governed guest power actions with the MCP server."""
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


class ProxmoxPowerResult(BaseModel):
    """The receipt of one approved power action: its task and the guest read back."""
    ok: bool
    status: str
    provider: str = PROVIDER
    operation: str = "guest_power"
    action: str | None = None
    node: str | None = None
    vmid: int | None = None
    type: str | None = None
    name: str | None = None
    expected_status: str | None = None
    observed_status: str | None = None
    upid: str | None = None
    task_exit_status: str | None = None
    changed: bool = False
    verification_ok: bool = False
    retry_safe: bool = True
    observed_at: str | None = None
    error: str | None = None
    error_code: str | None = None


def register_proxmox_tools(server, service: ProxmoxService | None, missing_settings: Iterable[str] = ()) -> None:
    """Each read returns observed data, or a typed failure naming the cause (see tools.provider_reads).

    A power action runs only with the approved, policy-frozen target (expected
    type and name) and returns the provider's receipt; it is never retried here.
    """
    missing = tuple(missing_settings)

    def proxmox(call) -> ProxmoxReadResult:
        if service is None:
            return ProxmoxReadResult(**not_configured(PROVIDER, LABEL, missing))
        return ProxmoxReadResult(**provider_read(PROVIDER, LABEL, call))

    def power(node: str, vmid: int, action: str, expected_type: str, expected_name: str | None) -> ProxmoxPowerResult:
        if service is None:
            return ProxmoxPowerResult(**not_configured(PROVIDER, LABEL, missing), action=action, node=node,
                                      vmid=vmid)
        return ProxmoxPowerResult(**service.guest_power(node, vmid, action, expected_type=expected_type,
                                                        expected_name=expected_name))

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
    def proxmox_guest_status(node: str, vmid: int) -> ProxmoxReadResult:
        """Read observed status and resource counters for one exact node and VMID."""
        return proxmox(lambda: service.guest_status(node, vmid))

    @server.tool()
    def proxmox_guest_config(node: str, vmid: int) -> ProxmoxReadResult:
        """Read the allow-listed, non-secret configuration for one exact node and VMID."""
        return proxmox(lambda: service.guest_config(node, vmid))

    @server.tool()
    def proxmox_node_storage(node: str) -> ProxmoxReadResult:
        """Read storage pools for one exact Proxmox node name."""
        return proxmox(lambda: service.node_storage(node))

    @server.tool()
    def proxmox_recent_tasks(node: str) -> ProxmoxReadResult:
        """Read recent tasks for one exact Proxmox node name."""
        return proxmox(lambda: service.recent_tasks(node))

    @server.tool()
    def proxmox_database_inventory() -> ProxmoxReadResult:
        """List guests tagged as database hosts with their state, capacity and backup coverage."""
        return proxmox(lambda: service.database_inventory())

    @server.tool()
    def proxmox_backups() -> ProxmoxReadResult:
        """Show each guest's newest backup, the backup jobs and which guests are overdue."""
        return proxmox(lambda: service.backups())

    @server.tool()
    def proxmox_guest_snapshots(node: str, vmid: int) -> ProxmoxReadResult:
        """List one exact guest's snapshots: name, age, parent and whether RAM was saved."""
        return proxmox(lambda: service.guest_snapshots(node, vmid))

    @server.tool()
    def proxmox_node_network(node: str) -> ProxmoxReadResult:
        """Show one exact node's bridges, bonds and ports; never addresses."""
        return proxmox(lambda: service.node_network(node))

    @server.tool()
    def proxmox_node_disks(node: str) -> ProxmoxReadResult:
        """Show one exact node's physical disks with SMART health and SSD wear."""
        return proxmox(lambda: service.node_disks(node))

    # The expected_* values are trusted policy arguments frozen at approval;
    # they are never model-facing parameters.

    @server.tool()
    def proxmox_guest_start(node: str, vmid: int, expected_type: str = "",
                            expected_name: str | None = None) -> ProxmoxPowerResult:
        """Start one approved, exact Proxmox guest and read it back."""
        return power(node, vmid, "start", expected_type, expected_name)

    @server.tool()
    def proxmox_guest_shutdown(node: str, vmid: int, expected_type: str = "",
                               expected_name: str | None = None) -> ProxmoxPowerResult:
        """Cleanly shut down one approved, exact Proxmox guest and read it back."""
        return power(node, vmid, "shutdown", expected_type, expected_name)

    @server.tool()
    def proxmox_guest_reboot(node: str, vmid: int, expected_type: str = "",
                             expected_name: str | None = None) -> ProxmoxPowerResult:
        """Cleanly reboot one approved, exact Proxmox guest and read it back."""
        return power(node, vmid, "reboot", expected_type, expected_name)

    @server.tool()
    def proxmox_guest_force_stop(node: str, vmid: int, expected_type: str = "",
                                 expected_name: str | None = None) -> ProxmoxPowerResult:
        """Force stop one approved, exact Proxmox guest and read it back."""
        return power(node, vmid, "force_stop", expected_type, expected_name)
