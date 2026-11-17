from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel

from services.dns_integrity import DnsIntegrityService
from services.openwrt import OpenWrtService
from services.tailnet import TailnetService
from services.tailnet_coverage import tailnet_coverage
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
                              missing_settings: Mapping[str, Sequence[str]] | None = None, proxmox=None, *,
                              tailnet_for_request: Callable[[], AbstractContextManager] | None = None,
                              proxmox_for_request: Callable[[], AbstractContextManager] | None = None):
    """Each read returns observed data, or a typed failure naming the cause (see tools.provider_reads).

    ``missing_settings`` maps a provider family (openwrt, tailscale, dns_check,
    proxmox) to the setting names it still needs, so an unconfigured read can
    name them. ``proxmox`` (a ProxmoxService) is read only by the tailnet
    coverage join. The ``*_for_request`` resolvers yield the Tailscale or
    Proxmox service of the account each call serves (owner decision, 17
    November); without them every call gets the fixed ``tailnet``/``proxmox``.
    """
    from services.provider_links import PROXMOX_LINK_MISSING, TAILSCALE_LINK_MISSING
    from tools.proxmox.mcp import fixed_service

    missing = dict(missing_settings or {})
    # Resolved per account, the fix is that account's own link, not a server setting.
    if tailnet_for_request is not None:
        missing["tailscale"] = TAILSCALE_LINK_MISSING
    if proxmox_for_request is not None:
        missing["proxmox"] = PROXMOX_LINK_MISSING
    tailnet_resolve = tailnet_for_request or fixed_service(tailnet)
    proxmox_resolve = proxmox_for_request or fixed_service(proxmox)

    def read(source, family: str, provider: str, label: str, call) -> NetworkReadResult:
        if source is None:
            return NetworkReadResult(**not_configured(provider, label, missing.get(family, ())))
        return NetworkReadResult(**provider_read(provider, label, call))

    def router(call):
        return read(service, "openwrt", "openwrt", "The OpenWrt router", call)

    def tailnet_read(call):
        with tailnet_resolve() as current:
            return read(current, "tailscale", "tailscale", "The Tailscale network", lambda: call(current))

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
        return tailnet_read(lambda current: current.devices())

    @server.tool()
    def network_tailnet_dns() -> NetworkReadResult:
        return tailnet_read(lambda current: current.dns_settings())

    @server.tool()
    def network_tailnet_policy() -> NetworkReadResult:
        """Who may reach what under the tailnet policy; users and addresses masked."""
        return tailnet_read(lambda current: current.policy())

    @server.tool()
    def network_tailnet_posture() -> NetworkReadResult:
        """Each tailnet device's fixed posture attributes."""
        return tailnet_read(lambda current: current.device_posture())

    @server.tool()
    def network_tailnet_coverage() -> NetworkReadResult:
        """Which Proxmox nodes and guests have a tailnet device, matched by name; each side's failure is named."""
        with tailnet_resolve() as tailnet_now, proxmox_resolve() as proxmox_now:
            if tailnet_now is None:
                return NetworkReadResult(**not_configured("tailscale", "The Tailscale network",
                                                          missing.get("tailscale", ())))
            if proxmox_now is None:
                return NetworkReadResult(**not_configured("proxmox", "Proxmox VE", missing.get("proxmox", ())))
            machines = provider_read("proxmox", "Proxmox VE", lambda: proxmox_now.machines())
            if not machines.get("ok"):
                return NetworkReadResult(**machines)
            devices = provider_read("tailscale", "The Tailscale network", lambda: tailnet_now.devices())
            if not devices.get("ok"):
                return NetworkReadResult(**devices)
            return NetworkReadResult(ok=True, status="success", provider="tailscale",
                                     observed_at=devices.get("observed_at"),
                                     data=tailnet_coverage(machines["data"], devices["data"]))

    @server.tool()
    def network_dns_integrity_check(names: list[str] | None = None) -> NetworkReadResult:
        return read(dns_check, "dns_check", "dns", "The DNS integrity check", lambda: dns_check.check(names))
