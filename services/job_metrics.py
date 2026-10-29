"""Best-effort, content-free inference observations for one durable job attempt.

Phoenix owns identity and persistence. These observations cannot authorize work,
change its outcome, or establish provider completion. No model/context/error text
or credentials are part of a metric payload.
"""
from __future__ import annotations

from services.lua_adapters.http_clients import async_client as lua_async_http_client

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
import math
import re
from urllib.parse import urlsplit
import uuid

from subagents.llm.runtime.stages import InferenceStage


_current: ContextVar["JobMetricPublisher | None"] = ContextVar("job_metrics", default=None)
_STAGES = frozenset(stage.value for stage in InferenceStage) | {"INVALID"}
_ADAPTERS = frozenset({"local", "hf-causal", "http", "pi", "opencode", "unresolved"})
_OUTCOMES = frozenset({"success", "error", "cancelled", "budget_rejected", "skipped"})
_MAX_MS = 86_400_000
_MAX_COUNT = 1_000_000_000


def _milliseconds(value, *, required: bool = False):
    if value is None and not required:
        return None
    if (type(value) not in {int, float} or not math.isfinite(value)
            or value < 0 or value * 1000 > _MAX_MS):
        raise ValueError("Invalid observed duration")
    return value * 1000


def _count(value):
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise ValueError("Invalid observed count")
    return value


def metric_payload(metric, *, attempt: int, event_id: str) -> dict:
    """Export an allowlist of actual observations, keeping missing values null."""
    if (type(attempt) is not int or attempt < 1
            or not isinstance(event_id, str) or not re.fullmatch(r"[0-9a-f]{32}", event_id)):
        raise ValueError("Invalid metric identity")
    stage, repair = metric.stage, metric.repair_of
    if stage not in _STAGES or (repair is not None and repair not in _STAGES):
        raise ValueError("Invalid observed stage")
    if metric.operation not in {"generate", "warm"} or metric.adapter not in _ADAPTERS or metric.outcome not in _OUTCOMES:
        raise ValueError("Invalid observation category")
    role = metric.role
    if role is not None and (not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", role)):
        raise ValueError("Invalid observed role")
    return {
        "attempt": attempt, "event_id": event_id, "stage": stage,
        "operation": metric.operation, "adapter": metric.adapter, "outcome": metric.outcome,
        "role": role, "repair_of": repair,
        "duration_ms": _milliseconds(metric.total_seconds, required=True),
        "queue_ms": _milliseconds(metric.queue_wait_seconds),
        "execution_ms": _milliseconds(metric.execution_seconds),
        "first_token_ms": _milliseconds(metric.first_token_seconds),
        **{target: _count(getattr(metric, source)) for source, target in (
            ("generated_tokens", "generated_tokens"), ("estimated_input_tokens", "estimated_input_tokens"),
            ("request_token_budget", "request_token_budget"), ("output_token_cap", "output_token_cap"),
            ("dropped_messages", "dropped_messages"),
        )},
    }


def report_inference_metric(metric) -> bool:
    """Called from the existing inference observation boundary; never raises."""
    publisher = _current.get()
    if publisher is None:
        return False
    try:
        return publisher.publish(metric)
    except (Exception, asyncio.CancelledError):
        return False


class JobMetricPublisher:
    def __init__(self, base_url: str, token: str, job_id: str, attempt: int, *,
                 send=None, queue_size: int = 64, flush_timeout: float = 1.0):
        parts = urlsplit(base_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.query or parts.fragment or not isinstance(token, str) or not token):
            raise ValueError("Invalid metrics transport configuration")
        if (not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id)
                or type(attempt) is not int or attempt < 1
                or type(queue_size) is not int or not 1 <= queue_size <= 64
                or type(flush_timeout) not in {int, float} or not math.isfinite(flush_timeout)
                or not 0 < flush_timeout <= 1):
            raise ValueError("Invalid metrics scope")
        self.url = base_url.rstrip("/") + f"/api/internal/v1/jobs/{job_id}/metrics"
        self.token, self.attempt = token, attempt
        self.queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=queue_size)
        self.worker: asyncio.Task | None = None
        self.send = send or self._send
        self.flush_timeout = flush_timeout

    def publish(self, metric) -> bool:
        payload = metric_payload(metric, attempt=self.attempt, event_id=uuid.uuid4().hex)
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            return False
        if self.worker is None:
            self.worker = asyncio.create_task(self._run())
        return True

    async def _send(self, payload: dict) -> None:
        import httpx
        # No redirects/proxies can forward the internal credential elsewhere.
        # An uncertain observational write may retry only its same immutable ID.
        async with lua_async_http_client(timeout=0.75, follow_redirects=False, trust_env=False) as client:
            for index in range(2):
                try:
                    response = await client.post(self.url, json=payload, headers={"X-Internal-Token": self.token})
                    if response.status_code in {200, 201} or response.status_code < 500:
                        return
                except httpx.TransportError:
                    pass
                if index == 1:
                    return

    async def _run(self):
        while True:
            payload = await self.queue.get()
            try:
                await self.send(payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # Never log transport objects, response bodies or headers.
            finally:
                self.queue.task_done()

    async def close(self):
        if self.worker is None:
            return
        try:
            async with asyncio.timeout(self.flush_timeout):
                await self.queue.join()
        except TimeoutError:
            pass
        finally:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)


@asynccontextmanager
async def job_metrics(settings, job_id: str, attempt: int):
    publisher = None
    if getattr(settings, "job_metrics_enabled", True):
        try:
            publisher = JobMetricPublisher(settings.require_phoenix_base_url(),
                settings.require_internal_job_token(), job_id, attempt)
        except Exception:
            pass
    # An unavailable/disabled child scope must not publish into its parent's job.
    reset = _current.set(publisher)
    try:
        yield publisher
    finally:
        _current.reset(reset)
        if publisher is not None:
            await publisher.close()
