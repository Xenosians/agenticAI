"""Check whether DNS answers are being rewritten on the way to this host.

Some networks (several Indonesian ISPs among them) intercept plain DNS on port
53 and answer with their own addresses, for blocking pages or redirection. This
service asks each name three ways from the AI host and compares the answers:

- plain DNS straight to a public resolver over UDP, which such a network can rewrite;
- the host's own resolver, which is what the services here actually use;
- DNS-over-HTTPS to the same provider, which an on-path network cannot rewrite
  without breaking the certificate.

Answers that share no address with the encrypted answer are reported as
rewritten, as is a plain "no such name" for a name that exists (and the
reverse). Only plain DNS and DNS-over-HTTPS decide that verdict: they ask the
same provider. The host's resolver may be another provider, whose geo-routed
answer can differ honestly, so it is reported beside the verdict: does it match
the encrypted answer, follow the rewrite, or give its own answer. A plain answer
that seems rewritten is asked over DNS-over-HTTPS once more first, because CDNs
hand out different addresses on every query. A path that fails is reported as
unknown, never as tampering. Every path crosses the Lua connection adapter first.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import ipaddress
import re
import secrets
import socket
import struct
from datetime import datetime, timezone
from typing import Callable

import httpx

from services.lua_adapters.host_boundaries import inspect_link
from services.lua_adapters.http_clients import client as lua_http_client

MAX_NAMES = 10
MAX_ADDRESSES = 16
MAX_ANSWER_BYTES = 4096
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


@dataclass(frozen=True)
class DnsIntegrityConfig:
    resolver: str
    doh_url: str
    default_names: tuple[str, ...]
    timeout_seconds: float = 3.0


def normalized_name(name) -> str:
    """A lower-case host name without the trailing dot; anything else is refused."""
    if not isinstance(name, str):
        raise ValueError("A host name is required.")
    value = name.strip().rstrip(".").lower()
    labels = value.split(".")
    try:
        ipaddress.ip_address(value)
        literal = True
    except ValueError:
        literal = False
    if (literal or not value or len(value) > 253 or len(labels) < 2
            or not all(_LABEL.fullmatch(label) for label in labels) or labels[-1].isdigit()):
        raise ValueError(f"'{name}' is not a host name.")
    return value


def query_message(name: str, ident: int) -> bytes:
    """A DNS query for the IPv4 addresses of ``name`` (RFC 1035 wire format)."""
    labels = normalized_name(name).split(".")
    question = b"".join(bytes([len(label)]) + label.encode("ascii") for label in labels) + b"\x00"
    return struct.pack(">HHHHHH", ident, 0x0100, 1, 0, 0, 0) + question + struct.pack(">HH", 1, 1)


def question_name(message: bytes) -> str:
    labels, offset = [], 12
    while message[offset]:
        length = message[offset]
        labels.append(message[offset + 1:offset + 1 + length].decode("ascii"))
        offset += 1 + length
    return ".".join(labels)


def _skip_name(data: bytes, offset: int) -> int:
    while True:
        if offset >= len(data):
            raise ValueError("truncated name")
        length = data[offset]
        if length & 0xC0 == 0xC0:
            return offset + 2
        if length == 0:
            return offset + 1
        offset += 1 + length


def answer_addresses(data: bytes, ident: int) -> list[str]:
    """The IPv4 addresses a DNS answer gives, sorted; an empty list for no such name."""
    if len(data) < 12:
        raise ValueError("The DNS answer is too short.")
    got, flags, questions, answers = struct.unpack(">HHHH", data[:8])
    if got != ident or not flags & 0x8000:
        raise ValueError("The DNS answer does not match the query.")
    if flags & 0x000F:
        return []
    offset = 12
    for _ in range(questions):
        offset = _skip_name(data, offset) + 4
    addresses = set()
    for _ in range(answers):
        offset = _skip_name(data, offset)
        if offset + 10 > len(data):
            raise ValueError("The DNS answer is truncated.")
        kind, cls, _ttl, length = struct.unpack(">HHIH", data[offset:offset + 10])
        offset += 10
        if kind == 1 and cls == 1 and length == 4:
            addresses.add(str(ipaddress.IPv4Address(data[offset:offset + 4])))
        offset += length
    return sorted(addresses)[:MAX_ADDRESSES]


def _udp_exchange(resolver: str, message: bytes, timeout: float) -> bytes:
    with socket.socket(socket.AF_INET6 if ":" in resolver else socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(message, (resolver, 53))
        return sock.recv(MAX_ANSWER_BYTES)


def _system_addresses(name: str) -> list[str]:
    rows = socket.getaddrinfo(name, 443, socket.AF_INET, socket.SOCK_STREAM)
    return sorted({str(row[4][0]) for row in rows})[:MAX_ADDRESSES]


def _failure(path: str, error: Exception) -> str:
    if isinstance(error, (TimeoutError, socket.timeout, httpx.TimeoutException)):
        return f"{path}: no answer in time"
    if isinstance(error, socket.gaierror):
        return f"{path}: the name does not resolve"
    if isinstance(error, ValueError):
        return f"{path}: invalid answer"
    return f"{path}: not reachable"


class DnsIntegrityService:
    def __init__(self, config: DnsIntegrityConfig, client: httpx.Client | None = None,
                 udp: Callable[[str, bytes, float], bytes] = _udp_exchange,
                 resolve: Callable[[str], list[str]] = _system_addresses):
        self.config = config
        self.client = client or lua_http_client(timeout=config.timeout_seconds)
        self.owns_client = client is None
        self.udp, self.resolve = udp, resolve

    def close(self):
        if self.owns_client:
            self.client.close()

    def _plain(self, name: str) -> list[str]:
        ident = secrets.randbelow(0x10000)
        inspect_link("dns", f"dns://{self.config.resolver}:53", operation="query")
        return answer_addresses(self.udp(self.config.resolver, query_message(name, ident),
                                         self.config.timeout_seconds), ident)

    def _system(self, name: str) -> list[str]:
        inspect_link("dns", "dns://system", operation="query")
        return self.resolve(name)

    def _encrypted(self, name: str) -> list[str]:
        # RFC 8484 GET with a zero id, so the answer can be cached by the resolver.
        message = query_message(name, 0)
        encoded = base64.urlsafe_b64encode(message).rstrip(b"=").decode("ascii")
        response = self.client.get(self.config.doh_url, params={"dns": encoded},
                                   headers={"Accept": "application/dns-message"})
        if response.status_code != 200 or response.headers.get("content-type", "").split(";")[0] != \
                "application/dns-message" or len(response.content) > MAX_ANSWER_BYTES:
            raise ValueError("DNS-over-HTTPS returned no DNS message")
        return answer_addresses(response.content, 0)

    def _row(self, name: str) -> dict:
        values, errors = {}, []
        for key, path, read in (("plain", "plain DNS", self._plain), ("system", "system resolver", self._system),
                                ("encrypted", "DNS-over-HTTPS", self._encrypted)):
            try:
                values[key] = read(name)
            except Exception as error:  # every path is reported, none hides another
                values[key] = None
                errors.append(_failure(path, error))
        plain, system, encrypted = values["plain"], values["system"], values["encrypted"]
        if plain is not None and encrypted is not None and not _agree(plain, encrypted):
            # A CDN may have rotated between the two queries: ask the encrypted path once more.
            try:
                encrypted = sorted(set(encrypted) | set(self._encrypted(name)))[:MAX_ADDRESSES]
            except Exception:
                pass
        if plain is None or encrypted is None:
            verdict = "unknown"
        elif _agree(plain, encrypted):
            verdict = "consistent"
        else:
            verdict = "plain_dns_rewritten"
        if system is None or encrypted is None:
            system_view = "unknown"
        elif _agree(system, encrypted):
            system_view = "matches_encrypted"
        elif plain is not None and verdict == "plain_dns_rewritten" and _agree(system, plain):
            system_view = "follows_rewrite"
        else:
            system_view = "different_answer"
        return {"name": name, "plain": plain, "system": system, "encrypted": encrypted,
                "verdict": verdict, "system_resolver": system_view, "errors": errors}

    def check(self, names: list[str] | None) -> dict:
        requested = list(self.config.default_names) if not names else names
        if len(requested) > MAX_NAMES:
            raise ValueError(f"Check at most {MAX_NAMES} names at a time.")
        unique = list(dict.fromkeys(normalized_name(name) for name in requested))
        rows = [self._row(name) for name in unique]
        rewritten = [row["name"] for row in rows if row["verdict"] == "plain_dns_rewritten"]
        follows = [row["name"] for row in rows if row["system_resolver"] == "follows_rewrite"]
        protected = None
        if rewritten:
            # Protected only when every rewritten name came back true from the host's resolver.
            protected = all(row["system_resolver"] == "matches_encrypted"
                            for row in rows if row["name"] in rewritten)
        return {"ok": True, "status": "success", "provider": "dns",
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "data": {"resolver": self.config.resolver, "names_checked": len(rows), "names": rows,
                         "rewritten": rewritten, "system_follows_rewrite": follows,
                         "system_resolver_protected": protected}}


def _agree(answer: list[str], reference: list[str]) -> bool:
    """Two answers agree when they share an address, or when both say the name does not exist."""
    return bool(set(answer) & set(reference)) or (not answer and not reference)


def build_dns_integrity_service(settings) -> DnsIntegrityService:
    return DnsIntegrityService(DnsIntegrityConfig(
        resolver=settings.dns_check_resolver, doh_url=settings.dns_check_doh_url,
        default_names=tuple(settings.dns_check_names), timeout_seconds=settings.dns_check_timeout_seconds))
