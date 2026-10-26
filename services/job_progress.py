"""Content-free AI progress and the lease heartbeat over Phoenix Channels.

Progress hints and lease renewals share one socket per processing attempt. It
is never an execution transport: completion stays on the durable HTTP
callback/outbox, and the HTTP heartbeat stays the lease fallback.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
import logging
import re
from urllib.parse import urlsplit, urlunsplit


STAGES = frozenset({"routing", "planning", "executing", "synthesizing", "waiting_provider"})
# How long a lease renewal waits for the socket before the HTTP heartbeat is used.
LEASE_REPLY_TIMEOUT_SECONDS = 10.0
_current: ContextVar["JobProgress | None"] = ContextVar("job_progress", default=None)
# Sockets by (job_id, attempt), so that attempt's heartbeat can renew over them.
_channels: dict[tuple[str, int], "JobProgress"] = {}
_wire_logger = logging.getLogger("itsm.progress.websocket")
_wire_logger.addHandler(logging.NullHandler())
_wire_logger.propagate = False
_wire_logger.disabled = True  # WebSocket handshake headers contain the internal credential.


def progress_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password:
        raise ValueError("Invalid Phoenix origin")
    if parts.query or parts.fragment:
        raise ValueError("Phoenix origin cannot contain a query or fragment")
    return urlunsplit(("wss" if parts.scheme == "https" else "ws", parts.netloc,
                       parts.path.rstrip("/") + "/ai_socket/websocket", "vsn=2.0.0", ""))


async def renew_lease(job_id: str, attempt: int) -> dict | None:
    """Phoenix's reply to renewing this attempt's lease over its joined socket.

    None means no joined socket answered; the caller then uses the HTTP
    heartbeat.
    """
    channel = _channels.get((job_id, attempt))
    return None if channel is None else await channel.renew_lease()


def report_progress(stage: str) -> None:
    """An observer failure must never change tool execution or cancellation."""
    publisher = _current.get()
    if publisher is not None and stage in STAGES:
        try:
            publisher.publish(stage)
        except Exception:
            pass


class JobProgress:
    def __init__(self, base_url: str, token: str, job_id: str, attempt: int, *, connect=None,
                 heartbeat_interval: float = 20.0):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id) or type(attempt) is not int or attempt < 1:
            raise ValueError("Invalid job identity")
        if not token:
            raise ValueError("Internal credential is required")
        self.url = progress_url(base_url)
        self.token = token
        self.topic = "ai_jobs:" + job_id
        self.attempt = attempt
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        self.lease_requests: asyncio.Queue[asyncio.Future] = asyncio.Queue()
        self.latest: str | None = None
        self.joined = False
        self._lease_in_flight: asyncio.Future | None = None
        self.connect = connect
        self.heartbeat_interval = heartbeat_interval

    def publish(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError("Unsupported progress stage")
        self.latest = stage
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(stage)

    async def renew_lease(self) -> dict | None:
        """Renew this attempt's lease over the joined socket (request and reply).

        None when the socket is not joined or does not answer in time.
        """
        if not self.joined:
            return None
        request = asyncio.get_running_loop().create_future()
        self.lease_requests.put_nowait(request)
        try:
            return await asyncio.wait_for(request, LEASE_REPLY_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return None

    async def _reply(self, socket, ref: str, join_ref: str | None, *, topic: str | None = None,
                     accept_error: bool = False) -> dict:
        expected_topic = self.topic if topic is None else topic
        async with asyncio.timeout(5):
            while True:
                frame = json.loads(await socket.recv())
                if not isinstance(frame, list) or len(frame) != 5:
                    continue
                if frame[2] in {self.topic, expected_topic} and frame[3] in {"phx_error", "phx_close"}:
                    raise ConnectionError("Progress channel closed")
                if (frame[2] == expected_topic and frame[3] == "phx_reply"
                        and frame[0] == join_ref and frame[1] == ref):
                    payload = frame[4]
                    if not isinstance(payload, dict) or (payload.get("status") != "ok" and not accept_error):
                        raise ConnectionError("Progress channel rejected")
                    return payload

    async def _next_items(self) -> list[tuple[str, object]]:
        """Lease requests first, then the latest progress stage; empty when idle."""
        getters = {
            asyncio.ensure_future(self.lease_requests.get()): "lease",
            asyncio.ensure_future(self.queue.get()): "progress",
        }
        try:
            await asyncio.wait(getters, timeout=self.heartbeat_interval, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for getter in getters:
                getter.cancel()
            await asyncio.gather(*getters, return_exceptions=True)
        # A getter that finished before the cancellation keeps its item.
        ready = [(kind, getter.result()) for getter, kind in getters.items()
                 if getter.done() and not getter.cancelled()]
        return sorted(ready, key=lambda item: item[0] != "lease")

    def _release_lease_requests(self) -> None:
        """Waiting heartbeats fall back to HTTP at once when the socket goes away."""
        pending = [self._lease_in_flight] if self._lease_in_flight is not None else []
        while not self.lease_requests.empty():
            pending.append(self.lease_requests.get_nowait())
        for request in pending:
            if not request.done():
                request.set_result(None)
        self._lease_in_flight = None

    async def _join(self, socket) -> str:
        join_ref = "1"
        await socket.send(json.dumps([join_ref, "1", self.topic, "phx_join", {"attempt": self.attempt}]))
        await self._reply(socket, "1", join_ref)
        return join_ref

    async def _serve(self, socket, join_ref: str) -> None:
        """Send lease renewals, progress and idle heartbeats until the socket fails."""
        self.joined = True
        try:
            if self.latest is not None:
                self.publish(self.latest)
            sequence = 1
            while True:
                items = await self._next_items()
                if not items:
                    # Phoenix requires its application heartbeat in addition to protocol pings.
                    sequence += 1
                    ref = str(sequence)
                    await socket.send(json.dumps([None, ref, "phoenix", "heartbeat", {}]))
                    # Consume application heartbeat acknowledgments as
                    # they arrive. Otherwise an idle model fills the
                    # bounded receive queue and stalls control pongs.
                    await self._reply(socket, ref, None, topic="phoenix")
                    continue
                for kind, value in items:
                    sequence += 1
                    ref = str(sequence)
                    if kind == "progress":
                        await socket.send(json.dumps([join_ref, ref, self.topic, "progress", {"stage": value}]))
                        await self._reply(socket, ref, join_ref)
                    elif not value.done():  # a caller that stopped waiting already used HTTP
                        self._lease_in_flight = value
                        await socket.send(json.dumps([join_ref, ref, self.topic, "renew_lease", {}]))
                        reply = await self._reply(socket, ref, join_ref, accept_error=True)
                        if not value.done():
                            value.set_result(reply)
                        self._lease_in_flight = None
        finally:
            self.joined = False
            self._release_lease_requests()

    async def run(self) -> None:
        if self.connect is None:
            try:
                from websockets.asyncio.client import connect
            except ImportError:
                # This optional observer must not leave an unhandled task
                # exception while the durable job continues without it.
                return
        else:
            connect = self.connect
        delay = 1.0
        while True:
            try:
                async with connect(
                    self.url, additional_headers={"X-Internal-Token": self.token},
                    open_timeout=3, close_timeout=1, ping_interval=15, ping_timeout=10,
                    max_size=8192, max_queue=4, logger=_wire_logger,
                ) as socket:
                    join_ref = await self._join(socket)
                    delay = 1.0
                    await self._serve(socket, join_ref)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Only a current progress hint is replayable. No jobs, approvals,
                # arguments, results or credentials enter this queue or its logs.
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)


@asynccontextmanager
async def job_progress(settings, job_id: str, attempt: int):
    from services.job_metrics import job_metrics
    async with job_metrics(settings, job_id, attempt):
        async with _socket_job_progress(settings, job_id, attempt) as publisher:
            yield publisher


@asynccontextmanager
async def _socket_job_progress(settings, job_id: str, attempt: int):
    publisher = None
    task = None
    token = None
    key = (job_id, attempt)
    try:
        if getattr(settings, "job_websocket_enabled", True):
            publisher = JobProgress(settings.require_phoenix_base_url(), settings.require_internal_job_token(), job_id, attempt)
            token = _current.set(publisher)
            _channels[key] = publisher
            publisher.publish("routing")
            task = asyncio.create_task(publisher.run())
    except Exception:
        pass
    try:
        yield publisher
    finally:
        if publisher is not None and _channels.get(key) is publisher:
            del _channels[key]
        if token is not None:
            _current.reset(token)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
