"""Knowledge crawler: a machine-declared site becomes one owner-private retrieval dataset.

The machine's Lua profile declares which web origins may be crawled; nothing else
is reachable. The crawl stays inside the starting origin and path, honours
robots.txt, follows only redirects that stay inside both, refuses non-text
content, bounds pages, bytes and depth, and its
text then passes the same admission (secret/PII scan, provenance, versions) as
any upload. These tests run real HTTP against a local server.
"""
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("lupa.lua54")

from services.extensions.service import command, extension_store  # noqa: E402
from services.knowledge.crawler import MAX_REDIRECTS, CrawlError, CrawlLimits, crawl  # noqa: E402
from services.lua_adapters import profile  # noqa: E402

SETTINGS = SimpleNamespace(model_profiles={}, require_internal_job_token=lambda: "crawler-test-token")

PAGES = {
    "/robots.txt": ("text/plain", "User-agent: *\nDisallow: /docs/private/\n"),
    "/docs/": ("text/html", """<html><head><title>VPN runbook</title><style>.x{color:red}</style>
        <script>steal()</script></head><body><nav>Menu</nav>
        <h1>Connect to the VPN</h1><p>Open the client and choose the <b>office</b> profile.</p>
        <a href="reset.html">Reset</a> <a href="/docs/reset.html#top">Again</a>
        <a href="/docs/private/keys.html">Keys</a> <a href="https://elsewhere.example/x">Away</a>
        <a href="/docs/manual.pdf">Manual</a> <a href="/docs/moved">Moved</a> <a href="/admin/">Admin</a>
        <a href="mailto:helpdesk@example.org">Mail</a></body></html>"""),
    "/docs/reset.html": ("text/html", "<html><head><title>Password reset</title></head><body>"
                                       "<p>Reset a password from the self-service portal.</p>"
                                       "<a href='/docs/'>Home</a></body></html>"),
    "/docs/private/keys.html": ("text/html", "<p>robots.txt excludes this page</p>"),
    "/admin/": ("text/html", "<p>Outside the starting path</p>"),
    "/docs/manual.pdf": ("application/pdf", "%PDF-1.7 binary"),
}


class Site(BaseHTTPRequestHandler):
    requested: list[str] = []
    pages = PAGES
    redirects: dict[str, str] = {}

    def do_GET(self):  # noqa: N802 - http.server API
        self.requested.append(self.path)
        if self.path == "/docs/moved":
            self.send_response(302)
            self.send_header("Location", "https://elsewhere.example/landing")
            self.end_headers()
            return
        if self.path in self.redirects:
            self.send_response(302)
            self.send_header("Location", self.redirects[self.path])
            self.end_headers()
            return
        page = self.pages.get(self.path)
        if page is None:
            self.send_response(404)
            self.end_headers()
            return
        kind, body = page
        if isinstance(kind, int):  # A bare status, such as an unavailable robots.txt.
            self.send_response(kind)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def site():
    Site.requested = []
    Site.pages = PAGES
    Site.redirects = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """Install a machine profile that declares the given crawlable origins."""
    def install(*origins, insecure=None):
        # By default every plain-HTTP origin given is also a declared plain-HTTP exception.
        insecure = [origin for origin in origins if origin.startswith("http://")] if insecure is None else insecure
        quoted = lambda values: "{" + ",".join(f'"{value}"' for value in values) + "}"  # noqa: E731
        script = tmp_path / "machine.lua"
        script.write_text(f"return {{links = {{sources = {{allowed_origins = {quoted(origins)}, "
                          f"insecure_http_origins = {quoted(insecure)}}}}}}}", encoding="utf-8")
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


def run(url, **limits):
    return asyncio.run(crawl(url, CrawlLimits(**limits)))


def test_the_profile_declares_which_sources_may_be_crawled(machine, site):
    machine(site)
    assert profile.source_policy() == {"allowed_origins": [site], "insecure_http_origins": [site]}


def test_plain_http_sources_must_also_be_allowed(machine, site):
    machine(site, insecure=["http://198.51.100.7"])
    with pytest.raises(Exception) as caught:
        profile.source_policy()
    assert getattr(caught.value, "code", "") == "lua_adapter_configuration_invalid"


def test_a_machine_that_declares_no_sources_can_crawl_nothing(monkeypatch, site):
    monkeypatch.delenv("ITSM_LUA_ADAPTER_FILE", raising=False)
    monkeypatch.delenv("ITSM_LUA_ADAPTER_SHA256", raising=False)
    with pytest.raises(CrawlError) as caught:
        run(site + "/docs/")
    assert caught.value.code == "crawl_origin_not_declared"
    assert Site.requested == []


def test_an_origin_the_profile_does_not_declare_is_refused_before_any_request(machine, site):
    machine("https://docs.example.org")
    with pytest.raises(CrawlError) as caught:
        run(site + "/docs/")
    assert caught.value.code == "crawl_origin_not_declared"
    assert Site.requested == []


def test_the_crawl_reads_text_pages_of_the_declared_site_only(machine, site):
    machine(site)
    result = run(site + "/docs/")

    assert [page.url for page in result.pages] == [site + "/docs/", site + "/docs/reset.html"]
    assert result.pages[0].title == "VPN runbook"
    assert "Connect to the VPN" in result.pages[0].text
    assert "Open the client and choose the office profile." in result.pages[0].text
    # Markup, scripts, styles and navigation never become knowledge.
    assert not any(word in result.pages[0].text for word in ("steal()", "color:red", "Menu", "<"))
    # robots.txt, other origins, other paths, redirects and mail links are never followed.
    assert "/docs/private/keys.html" not in Site.requested and result.skipped["robots"] == 1
    assert "/admin/" not in Site.requested
    assert result.skipped["redirect"] == 1
    # A binary file is refused on its declared type; its body is never read as text.
    assert result.skipped["unsupported_content"] == 1
    assert Site.requested.count("/docs/reset.html") == 1


def test_every_request_crosses_the_lua_connection_boundary(machine, site, monkeypatch):
    from services.lua_adapters import http_clients

    machine(site)
    seen = []
    original = http_clients.inspect_data_async

    async def recording(kind, phase, payload, binding):
        seen.append((kind, phase, payload["origin"], payload["method"]))
        return await original(kind, phase, payload, binding)

    monkeypatch.setattr(http_clients, "inspect_data_async", recording)
    run(site + "/docs/")
    assert ("connection", "request", site, "GET") in seen
    assert len(seen) == len(Site.requested)


def test_page_count_depth_and_bytes_are_bounded(machine, site):
    machine(site)
    assert len(run(site + "/docs/", max_pages=1).pages) == 1
    assert len(run(site + "/docs/", max_depth=0).pages) == 1
    first = len(run(site + "/docs/").pages[0].text.encode())
    result = run(site + "/docs/", max_total_bytes=first + 1)
    assert [len(page.text.encode()) for page in result.pages] == [first] and result.truncated


def test_a_first_page_beyond_the_byte_budget_is_reported_as_too_large(machine, site):
    machine(site)
    with pytest.raises(CrawlError) as caught:
        run(site + "/docs/", max_total_bytes=10)
    assert caught.value.code == "crawl_too_large"


def test_a_redirect_within_the_crawled_path_is_followed_and_checked_like_a_link(machine, site):
    # Documentation sites commonly send a directory to its start page.
    Site.pages = {"/robots.txt": ("text/plain", "User-agent: *\nDisallow: /kb/private/\n"),
                  "/kb/start": ("text/html", "<title>Start</title><p>Begin here</p><a href='hop'></a>"
                                             "<a href='out'></a><a href='hidden'></a><a href='up'></a>"),
                  "/kb/private/page": ("text/html", "<p>Excluded by robots.txt</p>"),
                  "/admin/": ("text/html", "<p>Outside the starting path</p>")}
    Site.redirects = {"/kb/": "/kb/start", "/kb/hop": "/kb/start", "/kb/out": "https://elsewhere.example/",
                      "/kb/hidden": "/kb/private/page", "/kb/up": "/admin/"}
    machine(site)
    result = run(site + "/kb/")

    # The start page is stored once under the address that served it.
    assert [(page.url, page.title) for page in result.pages] == [(site + "/kb/start", "Start")]
    # Redirects that leave the site, the starting path or robots.txt's permission are never followed.
    assert not {"/kb/private/page", "/admin/"} & set(Site.requested)
    assert result.skipped["redirect"] == 3
    assert Site.requested.count("/kb/start") == 1


def test_redirect_chains_are_bounded(machine, site):
    Site.pages = {"/robots.txt": ("text/plain", "")}
    Site.redirects = {f"/kb/{index}": f"/kb/{index + 1}" for index in range(10)}
    machine(site)
    with pytest.raises(CrawlError) as caught:
        run(site + "/kb/0")
    assert caught.value.code == "crawl_found_no_text"
    assert len(Site.requested) <= 1 + 1 + MAX_REDIRECTS  # robots.txt, the first page, its bounded hops


def test_markdown_documentation_is_read_as_text(machine, site):
    Site.pages = {"/robots.txt": ("text/plain", ""),
                  "/kb/": ("text/html", "<h1>KB</h1><a href='vpn.md'>VPN</a>"),
                  "/kb/vpn.md": ("text/markdown", "# VPN\n\nUse the office profile.")}
    machine(site)
    result = run(site + "/kb/")
    assert [page.url for page in result.pages] == [site + "/kb/", site + "/kb/vpn.md"]
    assert result.pages[1].text == "# VPN\n\nUse the office profile."


def test_pages_that_ask_not_to_be_indexed_or_followed_are_respected(machine, site):
    Site.pages = {"/robots.txt": ("text/plain", ""),
                  "/kb/": ("text/html", "<p>Index</p><a href='hidden.html'>H</a><a href='closed.html'>C</a>"),
                  "/kb/hidden.html": ("text/html", "<meta name='robots' content='noindex'><p>Hidden</p>"
                                                   "<a href='via-hidden.html'>V</a>"),
                  "/kb/closed.html": ("text/html", "<meta name='robots' content='nofollow'><p>Closed</p>"
                                                   "<a href='via-closed.html'>V</a>"),
                  "/kb/via-hidden.html": ("text/html", "<p>Reached through a noindex page</p>"),
                  "/kb/via-closed.html": ("text/html", "<p>Never followed</p>")}
    machine(site)
    result = run(site + "/kb/")
    assert [page.url.rsplit("/", 1)[1] for page in result.pages] == ["", "closed.html", "via-hidden.html"]
    assert result.skipped["noindex"] == 1
    assert "/kb/via-closed.html" not in Site.requested


def test_a_page_already_read_under_another_address_is_stored_once(machine, site):
    # The same document served at two addresses (a directory index and its file name).
    page = "<p>Same text</p><a href='index.html'></a>"
    Site.pages = {"/robots.txt": ("text/plain", ""), "/kb/": ("text/html", page),
                  "/kb/index.html": ("text/html", page)}
    machine(site)
    result = run(site + "/kb/")
    assert len(result.pages) == 1 and result.skipped["duplicate"] == 1


@pytest.mark.parametrize("status", [500, 503])
def test_an_unreachable_robots_txt_means_nothing_may_be_crawled(machine, site, status):
    Site.pages = {"/robots.txt": (status, ""), "/kb/": ("text/html", "<p>Never read</p>")}
    machine(site)
    with pytest.raises(CrawlError) as caught:
        run(site + "/kb/")
    assert caught.value.code == "crawl_robots_unavailable"
    assert Site.requested == ["/robots.txt"]


def test_a_missing_robots_txt_allows_crawling(machine, site):
    Site.pages = {"/kb/": ("text/html", "<p>Readable</p>")}
    machine(site)
    assert [page.text for page in run(site + "/kb/").pages] == ["Readable"]


def test_the_sites_crawl_delay_spaces_requests(machine, site):
    Site.pages = {"/robots.txt": ("text/plain", "User-agent: *\nCrawl-delay: 1\n"),
                  "/kb/": ("text/html", "<p>One</p><a href='two.html'>2</a>"),
                  "/kb/two.html": ("text/html", "<p>Two</p>")}
    machine(site)
    started = time.monotonic()
    assert len(run(site + "/kb/").pages) == 2
    assert time.monotonic() - started >= 1.0


def test_a_crawl_that_reaches_its_time_budget_keeps_what_it_read(machine, site):
    # The site asks for 3 s between requests; the next request would end after the 2 s budget.
    Site.pages = {"/robots.txt": ("text/plain", "User-agent: *\nCrawl-delay: 3\n"),
                  "/kb/": ("text/html", "<p>One</p><a href='two.html'></a>"),
                  "/kb/two.html": ("text/html", "<p>Two</p>")}
    machine(site)
    started = time.monotonic()
    result = run(site + "/kb/", total_timeout=2, request_timeout=1)
    assert [page.text for page in result.pages] == ["One"] and result.truncated
    assert "/kb/two.html" not in Site.requested
    # It stops as soon as the next request cannot fit, rather than waiting out the budget.
    assert time.monotonic() - started < 1.5


def test_a_slow_site_keeps_the_pages_read_before_the_time_budget_ended(machine, site, monkeypatch):
    # No politeness delay: the budget ends while the second page is still being read.
    from services.knowledge import crawler

    Site.pages = {"/robots.txt": ("text/plain", ""),
                  "/kb/": ("text/html", "<p>One</p><a href='two.html'></a>"),
                  "/kb/two.html": ("text/html", "<p>Two</p>")}
    machine(site)
    original = crawler._read

    async def slow(response, limit):
        if response.url.path.endswith("two.html"):
            await asyncio.sleep(5)
        return await original(response, limit)

    monkeypatch.setattr(crawler, "_read", slow)
    result = run(site + "/kb/", total_timeout=1, request_timeout=1)
    assert [page.text for page in result.pages] == ["One"] and result.truncated


def test_a_start_url_with_credentials_or_another_scheme_is_refused(machine, site):
    machine(site)
    for url in (site.replace("http://", "http://user:pass@") + "/docs/", "file:///etc/passwd",
                site + "/docs/?token=1"):
        with pytest.raises(CrawlError):
            run(url)
    assert Site.requested == []


def test_a_crawl_with_no_readable_text_admits_nothing(machine, site):
    Site.pages = {"/robots.txt": ("text/plain", ""), "/empty/": ("text/html", "<html><body></body></html>")}
    machine(site)
    with pytest.raises(CrawlError) as caught:
        run(site + "/empty/")
    assert caught.value.code == "crawl_found_no_text"


def test_the_crawl_action_admits_one_versioned_retrieval_dataset_with_provenance(machine, site, store):
    machine(site)
    payload = {"url": site + "/docs/", "name": "VPN runbooks", "license": "Internal use",
               "purpose": "rag", "reviewed": True}
    row = asyncio.run(command("alice", "dataset_crawl", payload, SETTINGS))

    assert (row["kind"], row["purpose"], row["version"], row["source"]) == ("dataset", "rag", 1, site + "/docs/")
    crawl_record = store.get("alice", row["id"])["crawl"]
    assert crawl_record["schema"] == "knowledge-crawl.v1"
    assert [page["url"] for page in crawl_record["pages"]] == [site + "/docs/", site + "/docs/reset.html"]
    assert all(len(page["sha256"]) == 64 for page in crawl_record["pages"])
    hits = asyncio.run(command("alice", "search", {"id": row["id"], "query": "password reset portal"}, SETTINGS))
    assert "self-service portal" in hits["hits"][0]["content"]
    assert hits["hits"][0]["trust"] == "untrusted_retrieval"
    # Owner-private like every dataset.
    assert store.list("bob") == []


def test_crawled_text_still_passes_secret_and_personal_data_admission(machine, site, store):
    Site.pages = {"/robots.txt": ("text/plain", ""),
                  "/leak/": ("text/html", "<p>Service account password=hunter2-secret-value</p>")}
    machine(site)
    payload = {"url": site + "/leak/", "name": "Leak", "license": "Internal", "purpose": "rag", "reviewed": True}
    with pytest.raises(CrawlError) as caught:
        asyncio.run(command("alice", "dataset_crawl", payload, SETTINGS))
    assert caught.value.code == "crawl_all_pages_withheld"
    assert store.list("alice") == []


def test_pages_that_fail_admission_are_withheld_and_counted_while_the_rest_are_saved(machine, site, store):
    Site.pages = {"/robots.txt": ("text/plain", ""),
                  "/kb/": ("text/html", "<p>Start</p><a href='leak.html'></a><a href='titled.html'></a>"
                                        "<a href='ok.html'></a>"),
                  "/kb/leak.html": ("text/html", "<p>Set password=hunter2-secret-value first</p>"),
                  # The stored form includes the title, so a title is scanned like the text.
                  "/kb/titled.html": ("text/html", "<title>Mail admin@example.org</title><p>Clean body</p>"),
                  "/kb/ok.html": ("text/html", "<p>Reset a password from the self-service portal.</p>")}
    machine(site)
    payload = {"url": site + "/kb/", "name": "KB", "license": "Internal", "purpose": "rag", "reviewed": True}
    row = asyncio.run(command("alice", "dataset_crawl", payload, SETTINGS))

    stored = store.get("alice", row["id"])
    assert [page["url"] for page in stored["crawl"]["pages"]] == [site + "/kb/", site + "/kb/ok.html"]
    assert stored["crawl"]["skipped"]["sensitive"] == 2
    assert "hunter2" not in stored["content"]["text"] and "admin@" not in stored["content"]["text"]


def test_crawled_sites_are_reference_knowledge_never_training_data(machine, site, store):
    machine(site)
    payload = {"url": site + "/docs/", "name": "Docs", "license": "Internal", "purpose": "training",
               "reviewed": True}
    with pytest.raises(ValueError, match="reference"):
        asyncio.run(command("alice", "dataset_crawl", payload, SETTINGS))
    assert Site.requested == []


def test_the_catalog_lists_the_sites_this_machine_lets_users_crawl(machine, site, store):
    machine(site, "https://docs.example.org")
    catalog = asyncio.run(command("alice", "list", {}, SimpleNamespace(
        model_profiles={}, require_internal_job_token=lambda: "crawler-test-token")))
    assert catalog["crawl_sources"] == [site, "https://docs.example.org"]


def test_a_catalog_without_declared_sources_offers_no_crawling(monkeypatch, store):
    monkeypatch.delenv("ITSM_LUA_ADAPTER_FILE", raising=False)
    monkeypatch.delenv("ITSM_LUA_ADAPTER_SHA256", raising=False)
    catalog = asyncio.run(command("alice", "list", {}, SimpleNamespace(
        model_profiles={}, require_internal_job_token=lambda: "crawler-test-token")))
    assert catalog["crawl_sources"] == []


def test_the_public_boundary_carries_only_the_crawl_code_and_message(machine, site, store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import extensions

    machine("https://docs.example.org")
    monkeypatch.setattr(extensions, "get_settings", lambda: SETTINGS)
    app = FastAPI()
    app.include_router(extensions.router)
    body = {"owner": "alice", "action": "dataset_crawl",
            "payload": {"url": site + "/docs/", "name": "Docs", "license": "Internal", "reviewed": True}}

    reply = TestClient(app).post("/internal/v1/extensions", json=body,
                                 headers={"x-internal-token": "crawler-test-token"})

    assert reply.status_code == 422
    assert reply.json()["detail"] == {"code": "crawl_origin_not_declared",
                                      "message": "This machine's profile does not allow crawling that site."}
