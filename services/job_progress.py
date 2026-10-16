"""Content-free AI progress over Phoenix Channels; never an execution transport."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import json
import logging
import re
from urllib.parse import urlsplit, urlunsplit


STAGES = frozenset({"routing", "planning", "executing", "synthesizing", "waiting_provider"})
_current: ContextVar["JobProgress | None"] = ContextVar("job_progress", default=None)
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
        self.latest: str | None = None
        self.connect = connect
        self.heartbeat_interval = heartbeat_interval

    def publish(self, stage: str) -> None:
        if stage not in STAGES:
            raise ValueError("Unsupported progress stage")
        self.latest = stage
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(stage)

    async def _reply(self, socket, ref: str, join_ref: str | None, *, topic: str | None = None):
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
                    if not isinstance(frame[4], dict) or frame[4].get("status") != "ok":
                        raise ConnectionError("Progress channel rejected")
                    return

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
                    join_ref = "1"
                    await socket.send(json.dumps([join_ref, "1", self.topic, "phx_join", {"attempt": self.attempt}]))
                    await self._reply(socket, "1", join_ref)
                    delay = 1.0
                    if self.latest is not None:
                        self.publish(self.latest)
                    sequence = 1
                    while True:
                        try:
                            stage = await asyncio.wait_for(self.queue.get(), timeout=self.heartbeat_interval)
                        except asyncio.TimeoutError:
                            # Phoenix requires its application heartbeat in addition to protocol pings.
                            sequence += 1
                            ref = str(sequence)
                            await socket.send(json.dumps([None, ref, "phoenix", "heartbeat", {}]))
                            # Consume application heartbeat acknowledgments as
                            # they arrive. Otherwise an idle model fills the
                            # bounded receive queue and stalls control pongs.
                            await self._reply(socket, ref, None, topic="phoenix")
                            continue
                        sequence += 1
                        ref = str(sequence)
                        await socket.send(json.dumps([join_ref, ref, self.topic, "progress", {"stage": stage}]))
                        await self._reply(socket, ref, join_ref)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Only a current progress hint is replayable. No jobs, approvals,
                # arguments, results or credentials enter this queue or its logs.
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)


@asynccontextmanager
async def job_progress(settings, job_id: str, attempt: int):
    publisher = None
    task = None
    token = None
    try:
        if getattr(settings, "job_websocket_enabled", True):
            publisher = JobProgress(settings.require_phoenix_base_url(), settings.require_internal_job_token(), job_id, attempt)
            token = _current.set(publisher)
            publisher.publish("routing")
            task = asyncio.create_task(publisher.run())
    except Exception:
        pass
    try:
        yield publisher
    finally:
        if token is not None:
            _current.reset(token)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
