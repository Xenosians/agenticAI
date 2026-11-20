from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any

from pydantic import BaseModel, ConfigDict

from services.dns_integrity import DnsIntegrityService
from services.openwrt import OpenWrtService
from services.tailnet import TailnetService
from services.tailnet_coverage import tailnet_coverage
from tools.provider_reads import not_configured, provider_read


class TailnetChangeResult(BaseModel):
    """The receipt of one approved tailnet policy change: the grant, the write and the read-back."""
    model_config = ConfigDict(extra="allow")

    ok: bool
    status: str
    provider: str = "tailscale"
    operation: str | None = None
    grant: str | None = None
    changed: bool = False
    verification_ok: bool = False
    retry_safe: bool = True
    error: str | None = None
    error_code: str | None = None


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

    # Approved tailnet policy changes. grant/expected_etag/*_sha256 are trusted policy arguments
    # frozen at approval; they are never model-facing.

    def grant_change(operation: str, sources: list[str], destinations: list[str], ports: list[str],
                     expected_etag: str, policy_sha256: str, new_policy_sha256: str) -> TailnetChangeResult:
        from services.tailnet_policy_change import apply_grant_change
        from services.tailnet_policy_edit import GrantSpec

        with tailnet_resolve() as current:
            if current is None:
                return TailnetChangeResult(**not_configured("tailscale", "The Tailscale network",
                                                            missing.get("tailscale", ())))
            try:
                spec = GrantSpec(tuple(sources or ()), tuple(destinations or ()), tuple(ports or ()))
                return TailnetChangeResult(**apply_grant_change(
                    current, operation, spec, expected_etag=expected_etag, policy_sha256=policy_sha256,
                    new_policy_sha256=new_policy_sha256))
            except Exception:
                # Nothing is known about a write that may have been sent: never invite a resend.
                return TailnetChangeResult(ok=False, status="outcome_unknown", retry_safe=False,
                                           error="The tailnet policy change did not report its outcome. Read "
                                                 "the policy before trying again.")

    @server.tool()
    def network_tailnet_grant_add(sources: list[str], destinations: list[str], ports: list[str], grant: str = "",
                                  expected_etag: str = "", policy_sha256: str = "",
                                  new_policy_sha256: str = "") -> TailnetChangeResult:
        """Add one approved grant to the tailnet policy and read the policy back."""
        return grant_change("add", sources, destinations, ports, expected_etag, policy_sha256, new_policy_sha256)

    @server.tool()
    def network_tailnet_grant_remove(sources: list[str], destinations: list[str], ports: list[str],
                                     grant: str = "", expected_etag: str = "", policy_sha256: str = "",
                                     new_policy_sha256: str = "") -> TailnetChangeResult:
        """Remove one approved grant from the tailnet policy and read the policy back."""
        return grant_change("remove", sources, destinations, ports, expected_etag, policy_sha256,
                            new_policy_sha256)

    # Approved device, DNS and key changes (services/tailnet_changes.py). setting/device_*/previous and
    # key_* are trusted policy arguments frozen at approval; they are never model-facing.

    def tailnet_change(run) -> TailnetChangeResult:
        with tailnet_resolve() as current:
            if current is None:
                return TailnetChangeResult(**not_configured("tailscale", "The Tailscale network",
                                                            missing.get("tailscale", ())))
            try:
                return TailnetChangeResult(**run(current))
            except Exception:
                # Nothing is known about a write that may have been sent: never invite a resend.
                return TailnetChangeResult(ok=False, status="outcome_unknown", retry_safe=False,
                                           error="The tailnet change did not report its outcome. Read the "
                                                 "tailnet before trying again.")

    def device_change(device: str, setting: str, value: dict, device_id: str, device_name: str,
                      device_tags: list[str], previous: dict) -> TailnetChangeResult:
        from services.tailnet_changes import DeviceChange, apply_device_change

        return tailnet_change(lambda current: apply_device_change(
            current, DeviceChange(device, setting, value), device_id=device_id, device_name=device_name,
            device_tags=list(device_tags or []), previous=dict(previous or {})))

    @server.tool()
    def network_tailnet_device_set(device: str, routes: list[str] | None = None, authorized: bool | None = None,
                                   tags: list[str] | None = None, key_expiry_disabled: bool | None = None,
                                   new_name: str | None = None, setting: str = "", device_id: str = "",
                                   device_name: str = "", device_tags: list[str] | None = None,
                                   previous: dict | None = None) -> TailnetChangeResult:
        """Change one approved setting of one tailnet device and read the device back."""
        value = {"routes": {"routes": routes}, "authorized": {"authorized": authorized},
                 "tags": {"tags": tags}, "key_expiry": {"key_expiry_disabled": key_expiry_disabled},
                 "name": {"name": new_name}}.get(setting)
        if value is None:
            return TailnetChangeResult(ok=False, status="denied", error="The approval names no device setting.")
        return device_change(device, setting, value, device_id, device_name, device_tags or [], previous or {})

    @server.tool()
    def network_tailnet_device_remove(device: str, device_id: str = "", device_name: str = "",
                                      device_tags: list[str] | None = None,
                                      previous: dict | None = None) -> TailnetChangeResult:
        """Remove one approved device from the tailnet and confirm it is no longer listed."""
        return device_change(device, "remove", {}, device_id, device_name, device_tags or [], previous or {})

    @server.tool()
    def network_tailnet_dns_set(nameservers: list[str] | None = None, magic_dns: bool | None = None,
                                search_paths: list[str] | None = None, split_dns_domain: str | None = None,
                                split_dns_nameservers: list[str] | None = None, setting: str = "",
                                previous: dict | None = None) -> TailnetChangeResult:
        """Change one approved tailnet DNS setting and read it back."""
        from services.tailnet_changes import apply_dns_change

        value = {"nameservers": {"nameservers": nameservers}, "magic_dns": {"magic_dns": magic_dns},
                 "search_paths": {"search_paths": search_paths},
                 "split_dns": {"domain": split_dns_domain, "nameservers": split_dns_nameservers}}.get(setting)
        if value is None:
            return TailnetChangeResult(ok=False, status="denied", error="The approval names no DNS setting.")
        return tailnet_change(lambda current: apply_dns_change(current, setting, value, previous=dict(previous or {})))

    @server.tool()
    def network_tailnet_key_revoke(key_id: str, key_type: str = "", key_description: str = "",
                                   key_tags: list[str] | None = None) -> TailnetChangeResult:
        """Revoke one approved tailnet auth key and confirm it is gone."""
        from services.tailnet_changes import apply_key_revoke

        return tailnet_change(lambda current: apply_key_revoke(
            current, key_id, key_type=key_type, key_description=key_description, key_tags=list(key_tags or [])))

    @server.tool()
    def network_dns_integrity_check(names: list[str] | None = None) -> NetworkReadResult:
        return read(dns_check, "dns_check", "dns", "The DNS integrity check", lambda: dns_check.check(names))
