from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from services.dns_integrity import DnsIntegrityService
from services.openwrt import OpenWrtService
from services.tailnet import TailnetService
from tools.provider_reads import not_configured, provider_read


class NetworkReadResult(BaseModel):
    ok: bool
    status: str
    provider: str = "openwrt"
    observed_at: str | None = None
    data: dict[str, Any] | None = None
    scope: str | None = None
    note: str | None = None
    error: str | None = None
    error_code: str | None = None


def register_networking_tools(server, service: OpenWrtService | None, tailnet: TailnetService | None = None,
                              dns_check: DnsIntegrityService | None = None,
                              missing_settings: Mapping[str, Sequence[str]] | None = None):
    """Each read returns observed data, or a typed failure naming the cause (see tools.provider_reads).

    ``missing_settings`` maps a provider family (openwrt, tailscale, dns_check) to
    the setting names it still needs, so an unconfigured read can name them.
    """
    missing = missing_settings or {}

    def read(source, family: str, provider: str, label: str, call) -> NetworkReadResult:
        if source is None:
            return NetworkReadResult(**not_configured(provider, label, missing.get(family, ())))
        return NetworkReadResult(**provider_read(provider, label, call))

    def router(call):
        return read(service, "openwrt", "openwrt", "The OpenWrt router", call)

    def tailnet_read(call):
        return read(tailnet, "tailscale", "tailscale", "The Tailscale network", call)

    @server.tool()
    def network_system_info() -> NetworkReadResult:
        return router(lambda: service.system_info())

    @server.tool()
    def network_interface_status(interface: str) -> NetworkReadResult:
        return router(lambda: service.interface_status(interface))

    @server.tool()
    def network_firewall_config() -> NetworkReadResult:
        return router(lambda: service.firewall_config())

    @server.tool()
    def network_dhcp_leases() -> NetworkReadResult:
        return router(lambda: service.dhcp_leases())

    @server.tool()
    def network_wireless_clients() -> NetworkReadResult:
        return router(lambda: service.wireless_clients())

    @server.tool()
    def network_router_health() -> NetworkReadResult:
        return router(lambda: service.router_health())

    @server.tool()
    def network_interfaces() -> NetworkReadResult:
        return router(lambda: service.interfaces())

    @server.tool()
    def network_known_hosts() -> NetworkReadResult:
        return router(lambda: service.known_hosts())

    @server.tool()
    def network_dns_config() -> NetworkReadResult:
        return router(lambda: service.dns_config())

    @server.tool()
    def network_tailnet_devices() -> NetworkReadResult:
        return tailnet_read(lambda: tailnet.devices())

    @server.tool()
    def network_tailnet_dns() -> NetworkReadResult:
        return tailnet_read(lambda: tailnet.dns_settings())

    @server.tool()
    def network_dns_integrity_check(names: list[str] | None = None) -> NetworkReadResult:
        return read(dns_check, "dns_check", "dns", "The DNS integrity check", lambda: dns_check.check(names))
