"""The Proxmox and Tailscale service of the account a request serves (owner decision, 17 November).

Every caller (the MCP tools, the dashboard reads and the pre-approval checks)
asks here instead of building a client from the machine's settings. The
account's own link (services/extensions/links.py) wins; an account without one
uses the machine profile's link as its default; with neither there is no
service, and the failure names the account's missing connection.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import services.proxmox as proxmox
import services.tailnet as tailnet

PROXMOX_LINK_MISSING = ("Proxmox link in Connections",)
TAILSCALE_LINK_MISSING = ("Tailscale link in Connections",)


def _owner_link(kind: str) -> dict | None:
    from services.extensions.context import current_owner

    owner = current_owner()
    if owner is None:
        return None
    from services.extensions.links import owner_links

    return owner_links().get(owner, kind)


def owner_proxmox_service(settings) -> proxmox.ProxmoxService | None:
    """The requesting account's Proxmox, else the machine default, else None."""
    link = _owner_link("proxmox")
    if link is None:
        return proxmox.build_proxmox_service(settings)
    return proxmox.ProxmoxService(proxmox.ProxmoxConfig(
        link["url"], link["token_id"], link["token_secret"], link["verify_tls"], None, link["server_name"],
        settings.proxmox_timeout_seconds, (), getattr(settings, "proxmox_read_attempts", 3),
        getattr(settings, "proxmox_retry_wait_seconds", 0.5)))


def owner_tailnet_service(settings) -> tailnet.TailnetService | None:
    """The requesting account's Tailscale, else the machine default, else None."""
    link = _owner_link("tailscale")
    if link is None:
        return tailnet.build_tailnet_service(settings)
    return tailnet.TailnetService(tailnet.TailnetConfig(
        link["client_id"], link["client_secret"], settings.tailscale_timeout_seconds,
        getattr(settings, "tailscale_read_attempts", 3),
        getattr(settings, "tailscale_retry_wait_seconds", 0.5)))


def proxmox_setup_missing(settings) -> tuple[str, ...]:
    """What the requesting account still has to connect for Proxmox; empty when it can be read."""
    if _owner_link("proxmox") is not None or proxmox.proxmox_integration_status(settings)["configured"]:
        return ()
    return PROXMOX_LINK_MISSING


def tailnet_setup_missing(settings) -> tuple[str, ...]:
    """What the requesting account still has to connect for Tailscale; empty when it can be read."""
    if _owner_link("tailscale") is not None or tailnet.tailnet_integration_status(settings)["configured"]:
        return ()
    return TAILSCALE_LINK_MISSING


@contextmanager
def as_owner(owner: str | None) -> Iterator[None]:
    """Run a block for ``owner`` (a dashboard read Phoenix authenticated), then restore."""
    from services.extensions.context import request_owner

    token = request_owner.set(owner)
    try:
        yield
    finally:
        request_owner.reset(token)


def service_for_request(build: Callable) -> Callable:
    """A resolver for the MCP tools: each call gets the current owner's service and closes it after."""
    @contextmanager
    def for_request():
        service = build()
        try:
            yield service
        finally:
            if service is not None:
                service.close()
    return for_request
