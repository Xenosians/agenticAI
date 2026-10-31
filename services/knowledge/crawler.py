"""Bounded knowledge crawler: one machine-declared site becomes reference text.

What may be crawled is the machine's decision, not the user's: the Lua machine
profile's ``sources`` section lists the allowed web origins, and nothing else is
reachable. Every request goes through the project's Lua-wrapped HTTP client, so
the connection boundary sees each fetch before it is made.

A crawl stays on its starting origin and beneath the starting path, honours the
site's robots.txt (including Crawl-delay) and robots meta tags, follows a redirect
only when its target passes the same checks as a link (at most three hops), reads
only HTML, Markdown and plain text, stores identical pages once,
and is bounded in pages, depth, bytes and time; when the time budget ends, the
pages already read are kept. It returns text with per-page provenance; admission
(secret/PII scanning, licence, review, versioning) is the dataset store's job,
exactly as for an uploaded file.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
from html.parser import HTMLParser
import posixpath
import re
from urllib.parse import urldefrag, urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from services.lua_adapters.http_clients import async_client
from services.lua_adapters.profile import source_policy

USER_AGENT = "JanusKnowledgeCrawler/1.0"
PLAIN_TYPES = {"text/plain", "text/markdown", "text/x-markdown"}
TEXT_TYPES = {"text/html", "application/xhtml+xml"} | PLAIN_TYPES
MAX_ROBOTS_BYTES = 512_000
MAX_CRAWL_DELAY = 10.0  # A site asking for longer simply gets fewer pages within the time budget.
MAX_REDIRECTS = 3
SKIPPED_ELEMENTS = {"script", "style", "noscript", "template", "svg", "nav", "header", "footer", "aside", "form"}
BLOCK_ELEMENTS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "table", "tr", "br", "pre",
                  "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "hr"}


class CrawlError(ValueError):
    """A crawl refused before or during reading; the code is safe to show the user."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def public_detail(self) -> dict:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class CrawlLimits:
    max_pages: int = 25
    max_depth: int = 3
    max_page_bytes: int = 1_000_000
    max_total_bytes: int = 1_500_000  # Leaves room beneath the 2 MB dataset admission limit.
    request_timeout: float = 15.0
    total_timeout: float = 110.0  # Below the backend's 130 s extension request timeout.

    def __post_init__(self):
        if not (1 <= self.max_pages <= 100 and 0 <= self.max_depth <= 5 and 1 <= self.max_page_bytes <= 5_000_000
                and 1 <= self.max_total_bytes <= 1_900_000 and 0 < self.request_timeout <= 60
                and 0 < self.total_timeout <= 120):
            raise CrawlError("crawl_limits_invalid", "Crawl limits are outside the supported range.")


@dataclass(frozen=True)
class CrawledPage:
    url: str
    title: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def stored(self) -> str:
        """The page as it is saved in the dataset: title, address, then the text."""
        return f"# {self.title}\nSource: {self.url}\n\n{self.text}"


@dataclass
class CrawlResult:
    start_url: str
    pages: list[CrawledPage] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    truncated: bool = False

    def skip(self, reason: str):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def corpus(self) -> str:
        """One reference text; each page is introduced by its title and address."""
        return "\n\n".join(page.stored() for page in self.pages)

    def withhold(self, admissible) -> None:
        """Drop pages admission would refuse, counting them, so one page cannot block a site.

        ``admissible`` decides on a page's stored form (title, address and text together).
        """
        kept = [page for page in self.pages if admissible(page.stored())]
        for _page in range(len(self.pages) - len(kept)):
            self.skip("sensitive")
        self.pages = kept

    def provenance(self) -> dict:
        return {"schema": "knowledge-crawl.v1", "start_url": self.start_url,
                "pages": [{"url": page.url, "title": page.title, "sha256": page.sha256} for page in self.pages],
                "skipped": dict(sorted(self.skipped.items())), "truncated": self.truncated}


class _Readable(HTMLParser):
    """Visible text, the title, outgoing links and robots directives; markup never becomes text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.links, self.robots, self._parts, self._skipping, self._in_title = "", [], [], [], 0, False

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            values = dict(attrs)
            if (values.get("name") or "").lower() in {"robots", "januscrawler"} and values.get("content"):
                self.robots.append(values["content"])
        if tag in SKIPPED_ELEMENTS:
            self._skipping += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIPPED_ELEMENTS and self._skipping:
            self._skipping -= 1
        elif tag == "title":
            self._in_title = False
        if tag in BLOCK_ELEMENTS:
            self._parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skipping:
            self._parts.append(data)

    def text(self) -> str:
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in "".join(self._parts).split("\n"))
        return "\n".join(line for line in lines if line)


def _origin(parts) -> str:
    return f"{parts.scheme}://{parts.netloc}".lower()


def _start(url: str) -> tuple[str, str, str]:
    """The canonical start URL, its origin and the path prefix the crawl stays beneath."""
    if not isinstance(url, str) or len(url) > 2000:
        raise CrawlError("crawl_url_invalid", "Enter the address of the page to start from.")
    parts = urlsplit(url.strip())
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment):
        raise CrawlError("crawl_url_invalid",
                         "Use an http(s) address without credentials, query parameters or fragments.")
    origin = _origin(parts)
    policy = source_policy()
    if origin not in policy["allowed_origins"]:
        raise CrawlError("crawl_origin_not_declared",
                         "This machine's profile does not allow crawling that site.")
    if parts.scheme == "http" and origin not in policy["insecure_http_origins"]:
        raise CrawlError("crawl_origin_not_declared", "This machine's profile allows that site only over HTTPS.")
    path = parts.path or "/"
    prefix = path if path.endswith("/") else posixpath.dirname(path).rstrip("/") + "/"
    return origin + path, origin, prefix


def _normalized(base: str, href: str, origin: str, prefix: str) -> str | None:
    """A same-site page beneath the starting path, or None."""
    url, _ = urldefrag(urljoin(base, href.strip()))
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or _origin(parts) != origin or parts.query or parts.username:
        return None
    path = posixpath.normpath(parts.path or "/") + ("/" if (parts.path or "/").endswith("/") else "")
    path = "/" if path == "//" else path
    if not path.startswith(prefix):
        return None
    return origin + path


async def _read(response, limit: int) -> bytes | None:
    content = bytearray()
    async for chunk in response.aiter_bytes():
        content.extend(chunk)
        if len(content) > limit:
            return None
    return bytes(content)


async def _robots(client, origin: str) -> RobotFileParser:
    """The site's robots rules, read as RFC 9309 directs.

    A missing robots.txt (4xx) allows crawling; an unreachable one (5xx, a redirect
    or a connection failure) means the site's wishes are unknown, so nothing is read.
    """
    rules = RobotFileParser()
    try:
        async with client.stream("GET", origin + "/robots.txt") as response:
            if response.status_code == 200:
                body = await _read(response, MAX_ROBOTS_BYTES)
            elif 400 <= response.status_code < 500:
                body = b""
            else:
                body = None
    except httpx.HTTPError:
        body = None
    if body is None:
        raise CrawlError("crawl_robots_unavailable",
                         "The site's robots.txt could not be read, so the site was not crawled; nothing was saved.")
    rules.parse(body.decode("utf-8", errors="replace").splitlines())
    return rules


def _crawl_delay(rules: RobotFileParser) -> float:
    delay = rules.crawl_delay(USER_AGENT)
    return min(float(delay), MAX_CRAWL_DELAY) if delay else 0.0


def _meta_robots(parser: "_Readable") -> set[str]:
    return {word.strip().lower() for word in ",".join(parser.robots).split(",") if word.strip()}


@dataclass
class _Fetched:
    address: str  # Where the page was actually served, after any followed redirect.
    kind: str
    document: str


class _Fetcher:
    """One polite reader for one site: robots rules, Crawl-delay and bounded redirects.

    Every hop is a separate request through the Lua-wrapped client, and a redirect is
    followed only when its target passes the same checks as a link: same origin,
    beneath the starting path and allowed by robots.txt.
    """

    def __init__(self, client, rules: RobotFileParser, origin: str, prefix: str, limits: CrawlLimits,
                 deadline: float, result: CrawlResult):
        self.client, self.rules, self.origin, self.prefix = client, rules, origin, prefix
        self.limits, self.deadline, self.result = limits, deadline, result
        self.delay, self.fetched = _crawl_delay(rules), False

    async def _pause(self) -> bool:
        """Wait the site's Crawl-delay; False when the next request would not finish in time."""
        if self.fetched and self.delay:
            loop = asyncio.get_running_loop()
            if loop.time() + self.delay >= self.deadline:
                return False
            await asyncio.sleep(self.delay)
        self.fetched = True
        return True

    async def fetch(self, address: str, visited: set[str]) -> _Fetched | None:
        """The page at address, or None (the reason is counted). Raises _OutOfTime."""
        for _hop in range(MAX_REDIRECTS + 1):
            if not self.rules.can_fetch(USER_AGENT, address):
                self.result.skip("robots")
                return None
            if not await self._pause():
                raise _OutOfTime()
            try:
                async with self.client.stream("GET", address) as response:
                    if response.is_redirect:
                        target = _normalized(address, response.headers.get("location", ""), self.origin, self.prefix)
                        if target is None or target in visited:
                            self.result.skip("redirect")
                            return None
                        visited.add(target)
                        address = target
                        continue
                    if response.status_code != 200:
                        self.result.skip("unavailable")
                        return None
                    kind = response.headers.get("content-type", "").split(";")[0].strip().lower()
                    if kind not in TEXT_TYPES:
                        self.result.skip("unsupported_content")
                        return None
                    body = await _read(response, self.limits.max_page_bytes)
                    encoding = response.encoding or "utf-8"
            except httpx.HTTPError:
                self.result.skip("unavailable")
                return None
            if body is None:
                self.result.skip("too_large")
                return None
            return _Fetched(address, kind, body.decode(encoding, errors="replace"))
        self.result.skip("redirect")
        return None


class _OutOfTime(Exception):
    """The next polite request would end after the crawl's time budget."""


async def crawl(url: str, limits: CrawlLimits | None = None) -> CrawlResult:
    limits = limits or CrawlLimits()
    start, origin, prefix = _start(url)
    deadline = asyncio.get_running_loop().time() + limits.total_timeout
    result = CrawlResult(start)
    try:
        await asyncio.wait_for(_crawl(result, origin, prefix, limits, deadline), limits.total_timeout)
    except (TimeoutError, _OutOfTime):
        # The time budget ends the crawl; pages already read are kept, the rest is not.
        result.truncated = True
    if not result.pages:
        if result.truncated:
            raise CrawlError("crawl_too_large", "The site exceeded the crawl's size or time limit before any "
                                                "page could be read; nothing was saved.")
        raise CrawlError("crawl_found_no_text", "No readable text was found at that address; nothing was saved.")
    return result


async def _crawl(result: CrawlResult, origin: str, prefix: str, limits: CrawlLimits, deadline: float):
    queue, seen, digests, total = [(result.start_url, 0)], {result.start_url}, set(), 0
    async with async_client(timeout=limits.request_timeout, follow_redirects=False, trust_env=False,
                            headers={"User-Agent": USER_AGENT,
                                     "Accept": "text/html, text/markdown, text/plain;q=0.9"}) as client:
        fetcher = _Fetcher(client, await _robots(client, origin), origin, prefix, limits, deadline, result)
        while queue and len(result.pages) < limits.max_pages:
            address, depth = queue.pop(0)
            page = await fetcher.fetch(address, seen)
            if page is None:
                continue
            index = follow = True
            if page.kind in PLAIN_TYPES:
                title = posixpath.basename(urlsplit(page.address).path) or page.address
                text, links = page.document.strip(), []
            else:
                parser = _Readable()
                parser.feed(page.document)
                parser.close()
                title, text, links = parser.title.strip() or page.address, parser.text(), parser.links
                directives = _meta_robots(parser)
                index = not directives & {"noindex", "none"}
                follow = not directives & {"nofollow", "none"}
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if not index:
                result.skip("noindex")
            elif text and digest in digests:
                result.skip("duplicate")
            elif text:
                size = len(text.encode("utf-8"))
                if total + size > limits.max_total_bytes:
                    result.truncated = True
                    return
                result.pages.append(CrawledPage(page.address, re.sub(r"\s+", " ", title)[:200], text))
                digests.add(digest)
                total += size
            if follow and depth < limits.max_depth:
                for href in links:
                    target = _normalized(page.address, href, origin, prefix)
                    if target and target not in seen:
                        seen.add(target)
                        queue.append((target, depth + 1))
        if queue:
            result.truncated = True
