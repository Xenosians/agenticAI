"""Connections that may reach only public internet addresses, checked when they are made.

An allowed host name is not enough to keep a reader on the public internet: its DNS
can point, or be rebound between two lookups, at loopback, a private network, a link-
local or cloud metadata address. ``PublicAddressBackend`` resolves the host itself
when a connection is opened, refuses it unless every resolved address is public, and
then connects to exactly the address it checked, so no second lookup can change the
answer. TLS still verifies the certificate for the host name.
"""
from __future__ import annotations

import asyncio
import ipaddress
import socket

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend

from services.transport_tls import configured_context


class PrivateAddressRefused(httpcore.ConnectError):
    """The host resolved to an address that is not on the public internet."""


def public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def refused_private_address(error: BaseException | None) -> bool:
    """Whether an httpx error was caused by a refused non-public address."""
    while error is not None:
        if isinstance(error, PrivateAddressRefused):
            return True
        error = error.__cause__ or error.__context__
    return False


class PublicAddressBackend(httpcore.AsyncNetworkBackend):
    """Opens TCP connections only to public addresses, resolved once and then used."""

    def __init__(self):
        self._backend = AutoBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        loop = asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as error:
            raise httpcore.ConnectError(str(error)) from error
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
        if not addresses or not all(public_address(address) for address in addresses):
            raise PrivateAddressRefused(f"{host} does not resolve to a public address")
        return await self._backend.connect_tcp(addresses[0], port, timeout=timeout, local_address=local_address,
                                               socket_options=socket_options)

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise PrivateAddressRefused("local sockets are not reachable")

    async def sleep(self, seconds):
        await self._backend.sleep(seconds)


class PublicAddressTransport(httpx.AsyncHTTPTransport):
    """An httpx transport whose connections go only to public addresses.

    It verifies TLS with the deployment's configured CA bundle when one is set,
    exactly as the project's other HTTP clients do, and never reads proxy settings.
    """

    def __init__(self):
        context = configured_context() or httpx.create_ssl_context(trust_env=False)
        super().__init__(verify=context, trust_env=False)
        self._pool = httpcore.AsyncConnectionPool(ssl_context=context, network_backend=PublicAddressBackend())
