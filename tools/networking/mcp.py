from typing import Any
from pydantic import BaseModel
from services.openwrt import OpenWrtService


class NetworkReadResult(BaseModel):
    ok: bool
    status: str
    provider: str = "openwrt"
    observed_at: str | None = None
    data: dict[str, Any] | None = None
    scope: str | None = None
    note: str | None = None


def register_networking_tools(server, service: OpenWrtService | None):
    def require_service():
        if service is None:
            raise RuntimeError("OpenWrt integration is not configured.")
        return service

    @server.tool()
    def network_system_info() -> NetworkReadResult:
        return NetworkReadResult(**require_service().system_info())

    @server.tool()
    def network_interface_status(interface: str) -> NetworkReadResult:
        return NetworkReadResult(**require_service().interface_status(interface))

    @server.tool()
    def network_firewall_config() -> NetworkReadResult:
        return NetworkReadResult(**require_service().firewall_config())

    @server.tool()
    def network_dhcp_leases() -> NetworkReadResult:
        return NetworkReadResult(**require_service().dhcp_leases())

    @server.tool()
    def network_wireless_clients() -> NetworkReadResult:
        return NetworkReadResult(**require_service().wireless_clients())
