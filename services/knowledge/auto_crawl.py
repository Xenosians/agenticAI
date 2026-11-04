"""Janus keeps the owner's permitted sites read into reference knowledge on its own.

When a knowledge search finds none of the owner's reference passages, the sites
the owner may crawl (the machine's declared sites plus the ones the owner
registered) that were not read in the last day are read again, in the
background, a few per request. The answer never waits: the next question
searches the new passages. Which passage answers a question is decided by the
same passage search as for every other dataset, never by guessing a site from
the question; no address ever comes from a model or a request.

Each site is kept as one reference dataset, replaced in place when its text
changes and left alone when it does not. Crawled text passes the same
admission as an upload, and it is reference knowledge only, never training
data. One site is read at a time per process, and each owner is queued at
most once. Every request crosses the Lua connection boundary like a manual
crawl. An owner stops a site being read by removing it from their crawl sites.
"""
from __future__ import annotations

import asyncio
from collections import deque
import hashlib
import threading
import time

from services.knowledge.crawler import CrawlError, crawl
from services.knowledge.sites import CrawlAccess, CrawlSites

REFRESH_SECONDS = 24 * 3600
SITES_PER_PASS = 3
NAME_SUFFIX = " (read automatically)"
LICENSE = "Reference only: read from a site this account may crawl, within the site's robots.txt"


def _now() -> float:
    return time.time()


def stale_sites(store, owner: str) -> list[str]:
    """The owner's permitted sites not read in the last day, least recently read first."""
    now = _now()
    read_at = {site.origin: store.auto_crawl_read_at(owner, site.origin) for site in CrawlSites(store).list(owner)}
    stale = [origin for origin, read in read_at.items() if read is None or now - read >= REFRESH_SECONDS]
    return sorted(stale, key=lambda origin: read_at[origin] or 0.0)


def _dataset_name(origin: str) -> str:
    site = origin.split("://", 1)[1]
    return site[:120 - len(NAME_SUFFIX)] + NAME_SUFFIX


async def _read_site(store, owner: str, origin: str, access: CrawlAccess) -> str:
    """Read one site into its dataset; the outcome is "saved" or "unchanged"."""
    from services.extensions.store import sensitive_content
    from services.lua_adapters.host_boundaries import inspect_data_async

    start = origin + "/"
    # The same Lua admission a manual crawl crosses before any page is fetched.
    await inspect_data_async("source", "admission", {"url": start, "purpose": "rag"},
                             {"owner": owner, "action": "dataset_crawl"})
    crawled = await crawl(start, access)
    crawled.withhold(lambda page: sensitive_content(page) is None)
    if not crawled.pages:
        raise CrawlError("crawl_all_pages_withheld", "Every page read contained possible credentials or "
                         "personal identifiers, so nothing was saved.")
    text = crawled.corpus()
    previous = store.auto_crawl_dataset(owner, origin)
    if previous is not None and previous["sha256"] == hashlib.sha256(text.encode()).hexdigest():
        return "unchanged"
    # The owner registered this site for crawling (or the machine declares it):
    # that is the permission an upload's review confirms.
    store.dataset(owner, name=_dataset_name(origin), text=text, source=start, license=LICENSE,
                  purpose="rag", reviewed=True, replaces=previous["id"] if previous else None,
                  crawl={**crawled.provenance(), "automatic": True, "site": origin})
    return "saved"


async def refresh_sites(owner: str) -> None:
    """Read up to SITES_PER_PASS of the owner's stale sites, one after another."""
    from services.extensions.service import extension_store

    store = extension_store()
    access = CrawlSites(store).access(owner)
    for origin in stale_sites(store, owner)[:SITES_PER_PASS]:
        # Recorded first, so a site that fails is also left alone for a day.
        store.record_auto_crawl(owner, origin, _now())
        try:
            outcome = await _read_site(store, owner, origin, access)
        except Exception as exc:  # One site failing never stops the others.
            outcome = getattr(exc, "code", None) or type(exc).__name__
        print(f"[KNOWLEDGE] Automatic site read outcome={outcome}")


class AutoCrawler:
    """One background reader per process: one pass at a time, each owner queued at most once."""

    def __init__(self):
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._running: str | None = None
        self._paused = False
        self._thread: threading.Thread | None = None

    def submit(self, owner: str) -> bool:
        """Queue a pass for the owner; False when one is already queued or running."""
        with self._condition:
            if owner == self._running or owner in self._queue:
                return False
            self._queue.append(owner)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="auto-crawl", daemon=True)
                self._thread.start()
            self._condition.notify_all()
            return True

    def pending(self, owner: str) -> int:
        """1 while a pass for the owner is queued or running, else 0."""
        with self._condition:
            return int(owner == self._running or owner in self._queue)

    def pause(self) -> None:
        """Start no new pass until resume(); a running pass finishes."""
        with self._condition:
            self._paused = True

    def resume(self) -> None:
        with self._condition:
            self._paused = False
            self._condition.notify_all()

    def wait(self, timeout: float) -> bool:
        """Wait until nothing is queued or running; False when the timeout ends first."""
        with self._condition:
            return self._condition.wait_for(lambda: not self._queue and self._running is None, timeout)

    def _work(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue and not self._paused)
                owner = self._running = self._queue.popleft()
            try:
                asyncio.run(refresh_sites(owner))
            except Exception as exc:
                print(f"[KNOWLEDGE] Automatic site reading failed error={type(exc).__name__}")
            finally:
                with self._condition:
                    self._running = None
                    self._condition.notify_all()


AUTO_CRAWLER = AutoCrawler()


def request_update(owner: str) -> bool:
    """Read the owner's stale permitted sites in the background; True when a read is queued or running."""
    from services.extensions.service import extension_store

    if AUTO_CRAWLER.pending(owner):
        return True
    if not stale_sites(extension_store(), owner):
        return False
    AUTO_CRAWLER.submit(owner)
    return True
