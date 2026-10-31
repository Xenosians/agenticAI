"""Which sites a crawl may read: the machine's sites plus those each user registers.

The machine's Lua profile (``sources``) lists sites every user may crawl and decides,
once per machine, whether users may register their own (``user_sites``) and which
private networks a crawl may reach (``private_networks``). A user registers a site
once, from the browser; it is remembered for that owner only, with its address
encrypted in the extension store.

Registering a site never widens where a crawl may connect: every connection is
checked against the address policy (public internet addresses plus the named private
networks). A user's plain http:// site is accepted only on a machine that names
private networks, and it then reaches only those networks; internet sites need https://.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import re
from urllib.parse import SplitResult, urlsplit

from services.knowledge.address_policy import AddressPolicy, Network
from services.knowledge.errors import CrawlError
from services.lua_adapters.profile import source_policy

MAX_SITES_PER_OWNER = 50
MAX_ORIGIN_LENGTH = 300
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST_NAME = re.compile(rf"(?=.{{1,253}}\Z){_LABEL}(?:\.{_LABEL})*")
_DEFAULT_PORTS = {"http": 80, "https": 443}


def origin_of(parts: SplitResult) -> str | None:
    """The canonical ``scheme://host[:port]`` of a split URL, or None when it is not a web origin."""
    try:
        port = parts.port
    except ValueError:
        return None
    scheme, host = parts.scheme.lower(), (parts.hostname or "").lower()
    if (scheme not in _DEFAULT_PORTS or not host or parts.username is not None or parts.password is not None
            or (port is not None and not 1 <= port <= 65535)):
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if not _HOST_NAME.fullmatch(host):
            return None
        shown = host
    else:
        shown = f"[{address.compressed}]" if address.version == 6 else address.compressed
    return f"{scheme}://{shown}" + (f":{port}" if port not in (None, _DEFAULT_PORTS[scheme]) else "")


def canonical_origin(value) -> str:
    """The exact site a user typed (scheme, host and optional port, no path), or CrawlError."""
    text = value.strip() if isinstance(value, str) else ""
    origin = None
    if 0 < len(text) <= MAX_ORIGIN_LENGTH and not any(mark in text for mark in ("?", "#", "\\", " ")):
        try:
            parts = urlsplit(text)
        except ValueError:
            parts = None
        if parts is not None and parts.path in {"", "/"}:
            origin = origin_of(parts)
    if origin is None:
        raise CrawlError("crawl_site_invalid",
                         "Enter a site address such as https://docs.example.org, without a path.")
    return origin


@dataclass(frozen=True)
class CrawlSite:
    origin: str
    managed: bool  # Declared in the machine profile, so the user cannot remove it.

    def public(self) -> dict:
        return {"origin": self.origin, "managed": self.managed}


@dataclass(frozen=True)
class CrawlAccess:
    """What one crawl may read, and where each of those sites may connect."""

    origins: frozenset[str]                     # Sites a crawl may start on.
    plain_http: frozenset[str] = frozenset()    # Those that may be read over plain http://.
    private_only: frozenset[str] = frozenset()  # Those that reach only the named private networks.
    private_networks: tuple[Network, ...] = ()

    def address_policy(self, origin: str) -> AddressPolicy:
        return AddressPolicy(public=origin not in self.private_only, private_networks=self.private_networks)


def _machine_origins(policy: dict) -> tuple[list[str], set[str]]:
    """The profile's declared sites and its plain-http exceptions, in canonical form."""
    canonical = {origin: origin_of(urlsplit(origin)) for origin in policy["allowed_origins"]}
    managed = list(dict.fromkeys(value for value in canonical.values() if value))
    plain = {canonical[origin] for origin in policy["insecure_http_origins"] if canonical.get(origin)}
    return managed, plain


def crawl_access(policy: dict, registered: list[str]) -> CrawlAccess:
    """The machine's declared sites plus one owner's registered (canonical) sites."""
    managed, plain = _machine_origins(policy)
    own = [origin for origin in registered if origin not in managed] if policy["user_sites"] else []
    own_plain = {origin for origin in own if origin.startswith("http://")}
    return CrawlAccess(origins=frozenset(managed) | frozenset(own), plain_http=frozenset(plain | own_plain),
                       private_only=frozenset(own_plain),
                       private_networks=tuple(ipaddress.ip_network(item) for item in policy["private_networks"]))


class CrawlSites:
    """One machine's crawl sites: those it declares, then those each owner registers."""

    def __init__(self, store):
        self.store = store
        self.policy = source_policy()

    @property
    def registration_open(self) -> bool:
        return self.policy["user_sites"]

    def list(self, owner: str) -> list[CrawlSite]:
        managed, _plain = _machine_origins(self.policy)
        own = self.store.crawl_sites(owner) if self.registration_open else []
        return ([CrawlSite(origin, True) for origin in managed]
                + [CrawlSite(origin, False) for origin in sorted(set(own) - set(managed))])

    def access(self, owner: str) -> CrawlAccess:
        return crawl_access(self.policy, [site.origin for site in self.list(owner) if not site.managed])

    def add(self, owner: str, value) -> CrawlSite:
        if not self.registration_open:
            raise CrawlError("crawl_site_registration_disabled", "This machine does not let users add sites.")
        origin = canonical_origin(value)
        managed, _plain = _machine_origins(self.policy)
        if origin in managed:
            return CrawlSite(origin, True)
        if origin.startswith("http://") and not self.policy["private_networks"]:
            raise CrawlError("crawl_site_needs_https", "Use https:// for a site on the internet.")
        if not self.store.add_crawl_site(owner, origin, limit=MAX_SITES_PER_OWNER):
            raise CrawlError("crawl_site_limit", f"You can keep up to {MAX_SITES_PER_OWNER} sites; "
                                                 "remove one you no longer need first.")
        return CrawlSite(origin, False)

    def remove(self, owner: str, value) -> bool:
        return self.store.remove_crawl_site(owner, canonical_origin(value))
