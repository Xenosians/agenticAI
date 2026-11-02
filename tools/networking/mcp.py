from typing import Any
from pydantic import BaseModel
from services.dns_integrity import DnsIntegrityService
from services.openwrt import OpenWrtService
from services.tailnet import TailnetService


class NetworkReadResult(BaseModel):
    ok: bool
    status: str
    provider: str = "openwrt"
    observed_at: str | None = None
    data: dict[str, Any] | None = None
    scope: str | None = None
    note: str | None = None


def register_networking_tools(server, service: OpenWrtService | None, tailnet: TailnetService | None = None,
                              dns_check: DnsIntegrityService | None = None):
    def require_service():
        if service is None:
            raise RuntimeError("OpenWrt integration is not configured.")
        return service

    def require_tailnet():
        if tailnet is None:
            raise RuntimeError("Tailscale integration is not configured.")
        return tailnet

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

    @server.tool()
    def network_router_health() -> NetworkReadResult:
        return NetworkReadResult(**require_service().router_health())

    @server.tool()
    def network_interfaces() -> NetworkReadResult:
        return NetworkReadResult(**require_service().interfaces())

    @server.tool()
    def network_known_hosts() -> NetworkReadResult:
        return NetworkReadResult(**require_service().known_hosts())

    @server.tool()
    def network_dns_config() -> NetworkReadResult:
        return NetworkReadResult(**require_service().dns_config())

    @server.tool()
    def network_tailnet_devices() -> NetworkReadResult:
        return NetworkReadResult(**require_tailnet().devices())

    @server.tool()
    def network_tailnet_dns() -> NetworkReadResult:
        return NetworkReadResult(**require_tailnet().dns_settings())

    @server.tool()
    def network_dns_integrity_check(names: list[str] | None = None) -> NetworkReadResult:
        if dns_check is None:
            raise RuntimeError("The DNS integrity check is not configured.")
        return NetworkReadResult(**dns_check.check(names))
