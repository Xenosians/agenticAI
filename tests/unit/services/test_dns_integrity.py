"""DNS integrity check: plain DNS vs the system resolver vs DNS-over-HTTPS, from the AI host."""
import struct

import httpx
import pytest

from services import dns_integrity
from services.dns_integrity import (
    DnsIntegrityConfig, DnsIntegrityService, answer_addresses, build_dns_integrity_service, query_message,
)


def wire_answer(query: bytes, addresses: list[str], *, rcode=0) -> bytes:
    ident = struct.unpack(">H", query[:2])[0]
    question = query[12:]
    header = struct.pack(">HHHHHH", ident, 0x8180 | rcode, 1, len(addresses), 0, 0)
    records = b"".join(b"\xc0\x0c" + struct.pack(">HHIH", 1, 1, 300, 4) + bytes(int(part) for part in a.split("."))
                       for a in addresses)
    return header + question + records


def test_queries_and_answers_use_the_dns_wire_format():
    message = query_message("vimeo.com", 0x1234)
    assert message[:2] == b"\x12\x34" and message[12:] == b"\x05vimeo\x03com\x00\x00\x01\x00\x01"
    assert answer_addresses(wire_answer(message, ["162.159.128.61", "162.159.138.60"]), 0x1234) == [
        "162.159.128.61", "162.159.138.60"]
    assert answer_addresses(wire_answer(message, [], rcode=3), 0x1234) == []
    with pytest.raises(ValueError):
        answer_addresses(wire_answer(message, ["1.2.3.4"]), 0x9999)
    with pytest.raises(ValueError):
        answer_addresses(b"\x00\x01", 1)


@pytest.mark.parametrize("name", ["", "-bad.example", "bad..example", "a" * 64 + ".com", "1.2.3.4",
                                  "exa mple.com", "example.com/path", "x" * 254])
def test_only_host_names_are_checked(name):
    with pytest.raises(ValueError):
        query_message(name, 1)


class FakeNetwork:
    """Answers per path; the Lua link checks are recorded in order. An encrypted entry may be a list of
    answers, returned one per query, to model a CDN rotating its addresses."""

    def __init__(self, plain, system, encrypted):
        self.plain, self.system, self.encrypted = plain, system, encrypted
        self.links, self.encrypted_requests = [], []

    def udp(self, resolver, message, timeout):
        name = dns_integrity.question_name(message)
        if isinstance(self.plain.get(name), Exception):
            raise self.plain[name]
        return wire_answer(message, self.plain.get(name, []))

    def resolve(self, name):
        if isinstance(self.system.get(name), Exception):
            raise self.system[name]
        return list(self.system.get(name, []))

    def transport(self, request: httpx.Request) -> httpx.Response:
        self.encrypted_requests.append(request)
        message = __import__("base64").urlsafe_b64decode(request.url.params["dns"] + "==")
        answers = self.encrypted.get(dns_integrity.question_name(message), [])
        if answers and isinstance(answers[0], list):
            answers = answers.pop(0) if len(answers) > 1 else answers[0]
        return httpx.Response(200, headers={"content-type": "application/dns-message"},
                              content=wire_answer(message, answers))


def service_for(network, names=("example.com",), monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setattr(dns_integrity, "inspect_link",
                            lambda transport, origin, **kwargs: network.links.append((transport, origin)))
    client = httpx.Client(transport=httpx.MockTransport(network.transport))
    return DnsIntegrityService(DnsIntegrityConfig(resolver="1.1.1.1", doh_url="https://cloudflare-dns.com/dns-query",
                                                  default_names=tuple(names), timeout_seconds=3.0),
                               client=client, udp=network.udp, resolve=network.resolve)


def test_a_rewritten_plain_answer_is_reported_and_the_protected_system_resolver_is_credited(monkeypatch):
    network = FakeNetwork(plain={"example.com": ["104.20.23.154"], "vimeo.com": ["36.86.63.185"]},
                          system={"example.com": ["104.20.23.154"], "vimeo.com": ["162.159.128.61"]},
                          encrypted={"example.com": ["104.20.23.154", "172.66.147.243"],
                                     "vimeo.com": ["162.159.128.61", "162.159.138.60"]})
    result = service_for(network, monkeypatch=monkeypatch).check(["example.com", "vimeo.com"])
    assert result["ok"] is True and result["status"] == "success" and result["provider"] == "dns"
    data = result["data"]
    assert data["resolver"] == "1.1.1.1" and data["names_checked"] == 2
    rows = {row["name"]: row for row in data["names"]}
    assert rows["example.com"]["verdict"] == "consistent"
    assert rows["vimeo.com"] == {
        "name": "vimeo.com", "plain": ["36.86.63.185"], "system": ["162.159.128.61"],
        "encrypted": ["162.159.128.61", "162.159.138.60"], "verdict": "plain_dns_rewritten",
        "system_resolver": "matches_encrypted", "errors": []}
    assert data["rewritten"] == ["vimeo.com"] and data["system_follows_rewrite"] == []
    assert data["system_resolver_protected"] is True
    # Every path crosses Lua before it connects: plain DNS and the system resolver as links, DoH as HTTP.
    assert ("dns", "dns://1.1.1.1:53") in network.links and ("dns", "dns://system") in network.links
    assert all(request.headers["accept"] == "application/dns-message" for request in network.encrypted_requests)


def test_a_system_resolver_that_follows_the_rewrite_is_not_called_protected(monkeypatch):
    network = FakeNetwork(plain={"vimeo.com": ["36.86.63.185"]}, system={"vimeo.com": ["36.86.63.185"]},
                          encrypted={"vimeo.com": ["162.159.128.61"]})
    data = service_for(network, monkeypatch=monkeypatch).check(["vimeo.com"])["data"]
    assert data["names"][0]["verdict"] == "plain_dns_rewritten"
    assert data["names"][0]["system_resolver"] == "follows_rewrite"
    assert data["rewritten"] == ["vimeo.com"] and data["system_follows_rewrite"] == ["vimeo.com"]
    assert data["system_resolver_protected"] is False


def test_another_resolvers_different_answer_is_its_own_view_not_tampering(monkeypatch):
    # Plain DNS and DNS-over-HTTPS ask the same provider and agree; the host's resolver is a
    # different one and gets a different geo-routed address, as observed live for github.com.
    network = FakeNetwork(plain={"github.com": ["20.205.243.166"]}, system={"github.com": ["140.82.121.3"]},
                          encrypted={"github.com": ["20.205.243.166"]})
    data = service_for(network, monkeypatch=monkeypatch).check(["github.com"])["data"]
    assert data["names"][0]["verdict"] == "consistent"
    assert data["names"][0]["system_resolver"] == "different_answer"
    assert data["rewritten"] == [] and data["system_resolver_protected"] is None


def test_a_blocked_or_invented_name_counts_as_a_rewrite(monkeypatch):
    network = FakeNetwork(plain={"blocked.example": [], "missing.example": ["36.86.63.185"]},
                          system={"blocked.example": ["93.184.215.14"], "missing.example": []},
                          encrypted={"blocked.example": ["93.184.215.14"], "missing.example": [], "gone.example": []})
    rows = {row["name"]: row for row in service_for(network, monkeypatch=monkeypatch).check(
        ["blocked.example", "missing.example", "gone.example"])["data"]["names"]}
    assert rows["blocked.example"]["verdict"] == "plain_dns_rewritten"
    assert rows["missing.example"]["verdict"] == "plain_dns_rewritten"
    assert rows["gone.example"]["verdict"] == "consistent"


def test_a_rotating_answer_is_asked_again_before_it_is_called_a_rewrite(monkeypatch):
    network = FakeNetwork(plain={"cdn.example": ["1.1.1.10"]}, system={"cdn.example": ["1.1.1.10"]},
                          encrypted={"cdn.example": [["1.1.1.11"], ["1.1.1.10"]]})
    row = service_for(network, monkeypatch=monkeypatch).check(["cdn.example"])["data"]["names"][0]
    assert row["verdict"] == "consistent" and row["encrypted"] == ["1.1.1.10", "1.1.1.11"]
    assert len(network.encrypted_requests) == 2


def test_overlapping_answers_are_consistent_because_cdns_rotate_addresses(monkeypatch):
    network = FakeNetwork(plain={"cdn.example": ["1.1.1.10", "1.1.1.11"]}, system={"cdn.example": ["1.1.1.11"]},
                          encrypted={"cdn.example": ["1.1.1.11", "1.1.1.12"]})
    data = service_for(network, monkeypatch=monkeypatch).check(["cdn.example"])["data"]
    assert data["names"][0]["verdict"] == "consistent" and data["system_resolver_protected"] is None


def test_a_path_that_fails_is_reported_as_unknown_never_as_tampering(monkeypatch):
    network = FakeNetwork(plain={"example.com": TimeoutError("timed out")}, system={"example.com": ["104.20.23.154"]},
                          encrypted={"example.com": ["104.20.23.154"]})
    row = service_for(network, monkeypatch=monkeypatch).check(["example.com"])["data"]["names"][0]
    assert row["verdict"] == "unknown" and row["plain"] is None
    assert row["errors"] == ["plain DNS: no answer in time"]


def test_without_names_the_configured_ones_are_checked_and_lists_are_bounded(monkeypatch):
    network = FakeNetwork(plain={}, system={}, encrypted={})
    service = service_for(network, names=("example.com", "wikipedia.org"), monkeypatch=monkeypatch)
    assert [row["name"] for row in service.check(None)["data"]["names"]] == ["example.com", "wikipedia.org"]
    with pytest.raises(ValueError, match="at most 10"):
        service.check([f"host{index}.example" for index in range(11)])
    assert [row["name"] for row in service.check(["Example.COM.", "example.com"])["data"]["names"]] == ["example.com"]


def test_a_doh_answer_that_is_not_a_dns_message_is_refused(monkeypatch):
    network = FakeNetwork(plain={}, system={}, encrypted={})
    network.transport = lambda request: httpx.Response(200, headers={"content-type": "text/html"}, content=b"<html>")
    row = service_for(network, monkeypatch=monkeypatch).check(["example.com"])["data"]["names"][0]
    assert row["verdict"] == "unknown" and row["encrypted"] is None
    assert row["errors"] == ["DNS-over-HTTPS: invalid answer"]


def test_the_resolver_pair_comes_from_settings():
    settings = type("Settings", (), {"dns_check_resolver": "9.9.9.9", "dns_check_doh_url": "https://dns.quad9.net/dns-query",
                                     "dns_check_names": ["example.com"], "dns_check_timeout_seconds": 2.0})()
    service = build_dns_integrity_service(settings)
    try:
        assert service.config.resolver == "9.9.9.9" and service.config.doh_url == "https://dns.quad9.net/dns-query"
        assert service.config.default_names == ("example.com",)
    finally:
        service.close()
