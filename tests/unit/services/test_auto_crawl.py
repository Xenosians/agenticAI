"""Janus reads permitted sites into reference knowledge on its own (Stage 3).

When a knowledge search finds none of the owner's reference passages, the sites
the owner may crawl (the machine's sites plus the owner's registered ones) that
were not read in the last day are read in the background. The answer never
waits for it; the next question searches the new passages, and passage search
alone decides whether they answer it. Nothing outside the owner's permitted
sites is ever fetched, the crawled text passes the same admission as an upload,
it is reference knowledge only, and each site is kept as one dataset that is
replaced in place when its text changes.
These tests run real HTTP against a local server, like the crawler's own tests.
"""
import asyncio
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("lupa.lua54")

from services.extensions.context import owner_execution  # noqa: E402
from services.extensions.service import extension_store  # noqa: E402
from services.knowledge import auto_crawl  # noqa: E402
from services.knowledge.auto_crawl import AutoCrawler  # noqa: E402
from services.knowledge.sites import CrawlSites  # noqa: E402
from tools.knowledge.owner_search import owner_knowledge_search  # noqa: E402

PAGES = {
    "/robots.txt": ("text/plain", ""),
    "/": ("text/html", "<html><head><title>Lab network handbook</title></head><body>"
                       "<p>Guest Wi-Fi isolation keeps visitors off the office network.</p>"
                       "<a href='/guest-wifi.html'>Guest Wi-Fi</a></body></html>"),
    "/guest-wifi.html": ("text/html", "<html><head><title>Guest Wi-Fi</title></head><body>"
                                      "<p>Create the guest Wi-Fi network with client isolation enabled, "
                                      "then allow only DNS and DHCP from the guest zone.</p></body></html>"),
}


class Site(BaseHTTPRequestHandler):
    requested: list[str] = []

    def do_GET(self):  # noqa: N802 - http.server API
        Site.requested.append(self.path)
        kind, body = PAGES.get(self.path, ("text/plain", None))
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002 - http.server API
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
    """A machine profile declaring the given sites, on loopback, as the crawler tests do."""
    def install(*origins, user_sites=False):
        quoted = "{" + ",".join(f'"{origin}"' for origin in origins) + "}"
        script = tmp_path / "machine.lua"
        script.write_text(f"return {{links = {{sources = {{allowed_origins = {quoted}, insecure_http_origins = "
                          f"{quoted}, private_networks = {{\"127.0.0.0/8\"}}, "
                          f"user_sites = {'true' if user_sites else 'false'}}}}}}}", encoding="utf-8")
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


@pytest.fixture
def crawler(monkeypatch):
    """The process's background crawler, run to completion inside each test."""
    instance = AutoCrawler()
    monkeypatch.setattr(auto_crawl, "AUTO_CRAWLER", instance)
    yield instance
    instance.wait(timeout=60)


EMPTY = {"ok": True, "status": "success", "hits": [], "count": 0}
QUESTION = "How do I set up guest Wi-Fi isolation?"


def search(owner, question=QUESTION):
    with owner_execution(owner, question):
        return owner_knowledge_search(dict(EMPTY), {"query": question})


def test_a_question_nothing_answers_reads_the_matching_permitted_site_in_the_background(
    machine, site, store, crawler,
):
    machine(site)

    first = search("alice")
    assert "reference_passages" not in first      # The answer never waits for the crawl.
    assert crawler.wait(timeout=60)

    datasets = [row for row in store.list("alice") if row.get("kind") == "dataset"]
    assert len(datasets) == 1
    dataset = datasets[0]
    assert (dataset["purpose"], dataset["source"]) == ("rag", site + "/")
    assert dataset["crawl"]["schema"] == "knowledge-crawl.v1"
    assert dataset["crawl"]["automatic"] is True
    assert {page["url"] for page in dataset["crawl"]["pages"]} == {site + "/", site + "/guest-wifi.html"}

    second = search("alice")
    assert "client isolation" in second["reference_passages"][0]["text"]
    assert store.list("bob") == []                 # Owner-private like any dataset.


def test_passages_from_a_site_answer_only_the_questions_they_match(machine, site, store, crawler):
    machine(site)

    search("alice", "When does the payroll change freeze start?")
    crawler.wait(timeout=60)

    # The site was read, but nothing in it is about payroll, so no passage is offered.
    assert [row["kind"] for row in store.list("alice")] == ["dataset"]
    assert "reference_passages" not in search("alice", "When does the payroll change freeze start?")
    assert "client isolation" in search("alice")["reference_passages"][0]["text"]


def test_answered_questions_start_no_read(machine, site, store, crawler):
    machine(site)
    search("alice")
    crawler.wait(timeout=60)
    requests = len(Site.requested)

    assert search("alice")["reference_passages"]
    crawler.wait(timeout=60)
    assert len(Site.requested) == requests


def test_nothing_is_read_without_a_permitted_site(machine, site, store, crawler):
    machine()  # The machine declares no sites, and users may not add any.

    search("alice")
    crawler.wait(timeout=60)

    assert Site.requested == []
    assert store.list("alice") == []


def test_a_site_is_read_at_most_once_a_day_and_replaced_in_place(machine, site, store, crawler, monkeypatch):
    machine(site)
    now = [1_000_000.0]
    monkeypatch.setattr(auto_crawl, "_now", lambda: now[0])

    unanswered = "When does the payroll change freeze start?"
    search("alice", unanswered)
    crawler.wait(timeout=60)
    first = [row for row in store.list("alice") if row.get("kind") == "dataset"]
    requests = len(Site.requested)

    search("alice", unanswered)
    crawler.wait(timeout=60)
    assert len(Site.requested) == requests             # Within a day: not read again.

    now[0] += 24 * 3600 + 1
    search("alice", unanswered)
    crawler.wait(timeout=60)
    assert len(Site.requested) > requests              # A day later: read again,
    assert [row["id"] for row in store.list("alice")] == [first[0]["id"]]  # unchanged text: no new version.

    PAGES["/guest-wifi.html"] = ("text/html", PAGES["/guest-wifi.html"][1].replace("DNS and DHCP", "DHCP only"))
    try:
        now[0] += 24 * 3600 + 1
        search("alice", unanswered)
        crawler.wait(timeout=60)
    finally:
        PAGES["/guest-wifi.html"] = ("text/html", PAGES["/guest-wifi.html"][1].replace("DHCP only", "DNS and DHCP"))
    after = [row for row in store.list("alice") if row.get("kind") == "dataset"]
    latest = max(after, key=lambda row: row["version"])
    assert latest["lineage_id"] == first[0]["lineage_id"] and latest["version"] == 2


def test_a_site_removed_and_added_again_is_read_again(machine, site, store, crawler):
    # Removing a site forgets when Janus last read it: registering it again (after
    # removing its dataset too) is a new site, read on the next unanswered question
    # instead of a day after the first read.
    machine(user_sites=True)
    sites = CrawlSites(store)
    sites.add("alice", site)
    search("alice")
    crawler.wait(timeout=60)
    first = [row for row in store.list("alice") if row.get("kind") == "dataset"]
    assert len(first) == 1
    requests = len(Site.requested)

    assert sites.remove("alice", site)
    store.delete("alice", first[0]["id"])
    sites.add("alice", site)
    search("alice")
    crawler.wait(timeout=60)
    assert len(Site.requested) > requests
    assert [row["crawl"]["automatic"] for row in store.list("alice") if row.get("kind") == "dataset"] == [True]


def test_one_owner_has_at_most_one_crawl_running(machine, site, store, crawler):
    machine(site)
    crawler.pause()
    try:
        search("alice")
        search("alice", "Where is guest Wi-Fi isolation configured?")
        assert crawler.pending("alice") == 1
    finally:
        crawler.resume()
    crawler.wait(timeout=60)


def test_crawled_pages_that_fail_admission_are_withheld(machine, site, store, crawler):
    machine(site)
    original = PAGES["/guest-wifi.html"]
    PAGES["/guest-wifi.html"] = ("text/html", "<html><head><title>Guest Wi-Fi</title></head><body>"
                                              "<p>Guest Wi-Fi isolation: password = hunter2secret</p></body></html>")
    try:
        search("alice")
        crawler.wait(timeout=60)
    finally:
        PAGES["/guest-wifi.html"] = original
    dataset = next(row for row in store.list("alice") if row.get("kind") == "dataset")
    assert dataset["crawl"]["skipped"].get("sensitive") == 1
    assert [page["url"] for page in dataset["crawl"]["pages"]] == [site + "/"]
