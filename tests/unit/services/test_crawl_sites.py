"""Sites a user registers for crawling: saved per owner, encrypted at rest.

The machine's Lua profile decides whether users may register sites (``user_sites``)
and which private networks a crawl may reach (``private_networks``, CIDR ranges), once
per machine. The user decides which sites, from the browser, once. A registered site
is remembered for its owner only, its address is encrypted in the store, and every
crawl still checks the address it actually connects to: plain http:// reaches only
the named private networks, https:// also reaches the public internet.
These tests run real HTTP against a local server.
"""
import asyncio
import hashlib
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

pytest.importorskip("lupa.lua54")

from services.extensions.service import command, extension_store  # noqa: E402
from services.knowledge.crawler import CrawlError  # noqa: E402
from services.knowledge.sites import MAX_SITES_PER_OWNER  # noqa: E402
from services.lua_adapters import profile  # noqa: E402

SETTINGS = SimpleNamespace(model_profiles={}, require_internal_job_token=lambda: "sites-test-token")
PAGES = {
    "/robots.txt": ("text/plain", ""),
    "/docs/": ("text/html", "<html><head><title>VPN runbook</title></head><body>"
                            "<p>Reset a VPN password from the self-service portal.</p></body></html>"),
}
CRAWL = {"name": "VPN runbooks", "license": "Internal use", "purpose": "rag", "reviewed": True}


class Site(BaseHTTPRequestHandler):
    requested: list = []

    def do_GET(self):  # noqa: N802 (http.server API)
        Site.requested.append(self.path)
        kind, body = PAGES.get(self.path, ("text/plain", None))
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002 (http.server API)
        pass


@pytest.fixture
def site():
    Site.requested = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """Install a machine profile whose ``sources`` section is the given Lua table."""
    def install(lua_sources: str):
        script = tmp_path / "machine.lua"
        script.write_text(f"return {{links = {{sources = {lua_sources}}}}}", encoding="utf-8")
        monkeypatch.setenv("ITSM_LUA_ADAPTER_FILE", str(script))
        monkeypatch.setenv("ITSM_LUA_ADAPTER_SHA256", hashlib.sha256(script.read_bytes()).hexdigest())
        monkeypatch.delenv("ITSM_LINK_SECRETS_FILE", raising=False)
    return install


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path / "extensions"))
    extension_store.cache_clear()
    try:
        yield extension_store()
    finally:
        extension_store.cache_clear()


def act(owner, action, payload):
    return asyncio.run(command(owner, action, payload, SETTINGS))


def sites_of(owner):
    return act(owner, "list", {})["crawl_sites"]


def test_the_profile_lets_users_register_sites_and_names_private_networks_once(machine):
    machine('{user_sites = true, private_networks = {"10.20.0.0/16", "192.168.8.0/24"}}')
    policy = profile.source_policy()
    assert policy["user_sites"] is True
    assert policy["private_networks"] == ["10.20.0.0/16", "192.168.8.0/24"]


@pytest.mark.parametrize("networks", ['{"0.0.0.0/0"}', '{"8.8.8.0/24"}', '{"not-a-network"}', '{"::/0"}',
                                      '{"169.254.0.0/16"}', '{"10.20.0.1/16"}'])
def test_a_private_network_entry_must_be_an_exact_private_range(machine, networks):
    # Never the whole internet, a public range, link-local (cloud metadata) or a host with a mask.
    machine(f"{{user_sites = true, private_networks = {networks}}}")
    with pytest.raises(Exception) as caught:
        profile.source_policy()
    assert getattr(caught.value, "code", "") == "lua_adapter_configuration_invalid"


def test_without_user_sites_a_user_cannot_register_a_site(machine, store):
    machine('{allowed_origins = {"https://docs.example.org"}}')
    with pytest.raises(CrawlError) as caught:
        act("alice", "crawl_site_add", {"origin": "https://wiki.example.net"})
    assert caught.value.code == "crawl_site_registration_disabled"
    assert sites_of("alice") == [{"origin": "https://docs.example.org", "managed": True}]
    assert act("alice", "list", {})["crawl_site_registration"] is False


def test_a_registered_site_is_remembered_for_its_owner_only(machine, store):
    machine('{allowed_origins = {"https://docs.example.org"}, user_sites = true}')
    added = act("alice", "crawl_site_add", {"origin": " https://Wiki.Example.NET/ "})
    assert added == {"origin": "https://wiki.example.net", "managed": False}
    act("alice", "crawl_site_add", {"origin": "https://wiki.example.net"})  # Registering again changes nothing.
    assert sites_of("alice") == [{"origin": "https://docs.example.org", "managed": True},
                                 {"origin": "https://wiki.example.net", "managed": False}]
    assert sites_of("bob") == [{"origin": "https://docs.example.org", "managed": True}]
    assert act("alice", "list", {})["crawl_site_registration"] is True


def test_the_site_address_is_encrypted_at_rest(machine, store):
    machine("{user_sites = true}")
    act("alice", "crawl_site_add", {"origin": "https://intranet-docs.example.net"})
    raw = b"".join(path.read_bytes() for path in store.root.glob("extensions.sqlite3*"))
    assert b"intranet-docs" not in raw
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM crawl_site").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM resource").fetchone()[0] == 0


@pytest.mark.parametrize("origin", [
    "ftp://docs.example.org", "https://user:pw@docs.example.org", "https://docs.example.org/path",
    "https://docs.example.org?q=1", "https://docs.example.org#top", "docs.example.org", "",
    "https://docs example.org", "https://docs.example.org:0", "https://docs.example.org:99999",
    "https://" + "a" * 300 + ".org", 42,
])
def test_only_an_exact_web_origin_can_be_registered(machine, store, origin):
    machine('{user_sites = true, private_networks = {"10.20.0.0/16"}}')
    with pytest.raises(CrawlError) as caught:
        act("alice", "crawl_site_add", {"origin": origin})
    assert caught.value.code == "crawl_site_invalid"
    assert sites_of("alice") == []


def test_plain_http_needs_a_private_network_the_machine_names(machine, store):
    machine("{user_sites = true}")
    with pytest.raises(CrawlError) as caught:
        act("alice", "crawl_site_add", {"origin": "http://wiki.lan:8090"})
    assert caught.value.code == "crawl_site_needs_https"
    machine('{user_sites = true, private_networks = {"10.20.0.0/16"}}')
    assert act("alice", "crawl_site_add", {"origin": "http://wiki.lan:8090"})["managed"] is False


def test_a_registered_site_inside_a_named_private_network_is_read_and_can_be_removed(machine, store, site):
    machine('{user_sites = true, private_networks = {"127.0.0.0/8"}}')
    act("alice", "crawl_site_add", {"origin": site})
    row = act("alice", "dataset_crawl", {"url": site + "/docs/", **CRAWL})
    assert (row["kind"], row["purpose"]) == ("dataset", "rag")
    assert row["crawl"]["pages"][0]["url"] == site + "/docs/"

    assert act("alice", "crawl_site_remove", {"origin": site}) == {"removed": True}
    assert sites_of("alice") == []
    with pytest.raises(CrawlError) as caught:
        act("alice", "dataset_crawl", {"url": site + "/docs/", **CRAWL, "name": "Again"})
    assert caught.value.code == "crawl_origin_not_declared"


def test_another_owner_cannot_crawl_a_site_registered_by_someone_else(machine, store, site):
    machine('{user_sites = true, private_networks = {"127.0.0.0/8"}}')
    act("alice", "crawl_site_add", {"origin": site})
    with pytest.raises(CrawlError) as caught:
        act("bob", "dataset_crawl", {"url": site + "/docs/", **CRAWL})
    assert caught.value.code == "crawl_origin_not_declared"
    assert Site.requested == []


def test_a_registered_site_outside_every_named_network_is_refused_when_connecting(machine, store, site):
    """Registering a site never widens where a crawl may connect: loopback that the
    machine did not name is refused, and nothing is read."""
    machine('{user_sites = true, private_networks = {"10.20.0.0/16"}}')
    act("alice", "crawl_site_add", {"origin": site})
    with pytest.raises(CrawlError) as caught:
        act("alice", "dataset_crawl", {"url": site + "/docs/", **CRAWL})
    assert caught.value.code == "crawl_site_unreachable"
    assert Site.requested == []


def test_registered_sites_are_bounded_per_owner(machine, store):
    machine("{user_sites = true}")
    for index in range(MAX_SITES_PER_OWNER):
        act("alice", "crawl_site_add", {"origin": f"https://docs{index}.example.org"})
    with pytest.raises(CrawlError) as caught:
        act("alice", "crawl_site_add", {"origin": "https://one-more.example.org"})
    assert caught.value.code == "crawl_site_limit"
    assert len(sites_of("alice")) == MAX_SITES_PER_OWNER
