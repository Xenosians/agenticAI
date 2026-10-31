"""Which network addresses an outbound reader may connect to, checked when it connects.

An allowed host name is not enough to keep a reader where it belongs: its DNS can
point, or be rebound between two lookups, at loopback, a private network, a link-local
or cloud metadata address. ``AddressPolicy`` decides per address: public internet
addresses are reachable when the policy allows them, other addresses only inside the
private networks the machine's profile names. ``CheckedAddressBackend`` resolves the
host itself when a connection is opened, refuses it unless every resolved address is
allowed, and then connects to exactly the address it checked, so no second lookup can
change the answer. TLS still verifies the certificate for the host name.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import ipaddress
import socket

import httpcore
import httpx

from services.transport_tls import configured_context

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


class AddressRefused(httpcore.ConnectError):
    """The host resolved to an address the policy does not allow."""


def _unmapped(address: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def public_address(address: str) -> bool:
    ip = _unmapped(address)
    return ip.is_global and not ip.is_multicast


@dataclass(frozen=True)
class AddressPolicy:
    """Public internet addresses (when allowed) plus the named private networks."""

    public: bool = True
    private_networks: tuple[Network, ...] = ()

    def allows(self, address: str) -> bool:
        if self.public and public_address(address):
            return True
        ip = _unmapped(address)
        return any(ip in network for network in self.private_networks)


def refused_address(error: BaseException | None) -> bool:
    """Whether an httpx error was caused by an address the policy refused."""
    while error is not None:
        if isinstance(error, AddressRefused):
            return True
        error = error.__cause__ or error.__context__
    return False


class CheckedAddressBackend(httpcore.AsyncNetworkBackend):
    """Opens TCP connections only to addresses the policy allows, resolved once and then used."""

    def __init__(self, policy: AddressPolicy):
        self._policy = policy
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            raise httpcore.ConnectError(str(error)) from error
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
        if not addresses or not all(self._policy.allows(address) for address in addresses):
            raise AddressRefused(f"{host} resolves to an address this machine does not allow")
        return await self._backend.connect_tcp(addresses[0], port, timeout=timeout, local_address=local_address,
                                               socket_options=socket_options)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise AddressRefused("local sockets are not reachable")

    async def sleep(self, seconds):
        await self._backend.sleep(seconds)


class CheckedAddressTransport(httpx.AsyncHTTPTransport):
    """An httpx transport whose connections go only to addresses the policy allows.

    It verifies TLS with the deployment's configured CA bundle when one is set,
    exactly as the project's other HTTP clients do, and never reads proxy settings.
    """

    def __init__(self, policy: AddressPolicy):
        context = configured_context() or httpx.create_ssl_context(trust_env=False)
        super().__init__(verify=context, trust_env=False)
        self._pool = httpcore.AsyncConnectionPool(ssl_context=context, network_backend=CheckedAddressBackend(policy))
