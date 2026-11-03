"""Execute a Lua protocol mapping without granting it transport or tool authority.

Every call uses a separate isolated process that serves that one event only.
Processes are started ahead of time, so an event does not wait for the Python
interpreter and its imports. Failure never substitutes a Python
mapping, another provider or an invented successful result. Callers retain exact
target/owner, budget, approval and provider-receipt checks around this boundary.
"""
from __future__ import annotations

import asyncio
import atexit
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import signal
import threading
import time
from typing import Any

from services.extensions.transform import MAX_SOURCE_BYTES, SOURCE_PATTERN, worker_environment
from services.sensitive_fields import CREDENTIAL_VALUE_FIELD_NAMES, is_credential_value_field

WORKER = Path(__file__).with_name("worker.py")
DEFAULT_SOURCE = Path(__file__).with_name("default.lua")
EVENT_SCHEMA = "lua-adapter-event.v1"
RECEIPT_SCHEMA = "lua-adapter-receipt.v1"
KINDS = frozenset({"model", "tool", "connection", "source"})
PHASES = frozenset({"admission", "request", "response", "warm"})
# A whole admitted dataset (2,000,000 bytes) plus its metadata and JSON escaping must fit.
MAX_INPUT_BYTES = 4_000_000
MAX_BINDING_BYTES = 16_000
# Workers kept started and waiting; a burst beyond this starts workers on demand.
SPARE_WORKERS = 2
# Workers serving events at once. A burst beyond this waits, within each event's
# own wall-time limit, for a worker to finish; it never starts more processes.
ACTIVE_WORKERS = 8
REQUIRED_ISOLATION = frozenset({"process", "no_new_privileges", "resource_limits", "no_new_descriptors"})
SECRET_FIELDS = CREDENTIAL_VALUE_FIELD_NAMES
ERROR_MESSAGES = {
    "lua_adapter_invalid_input": "The Lua adapter requires bounded JSON data.",
    "lua_adapter_input_too_large": "The Lua adapter input exceeds its size limit.",
    "lua_adapter_binding_invalid": "The Lua adapter requires a nonempty host-owned binding.",
    "lua_adapter_secret_input": "Credentials must remain with the host, outside Lua.",
    "lua_adapter_invalid_source": "The Lua adapter script must be bounded plain text.",
    "lua_adapter_configuration_invalid": "The Lua adapter configuration is invalid or its script identity changed.",
    "lua_adapter_unavailable": "The Lua adapter is unavailable on this host.",
    "lua_adapter_busy": "Every Lua adapter worker stayed busy for this event's whole time limit.",
    "lua_adapter_worker_terminated": "The Lua adapter worker terminated without a receipt.",
    "lua_adapter_time_limit": "The Lua adapter exceeded its time limit.",
    "lua_adapter_memory_limit": "The Lua adapter exceeded its memory limit.",
    "lua_adapter_instruction_limit": "The Lua adapter exceeded its instruction limit.",
    "lua_adapter_syntax_error": "The Lua adapter script has invalid syntax.",
    "lua_adapter_contract_invalid": "The Lua adapter must return an event-mapping function.",
    "lua_adapter_output_invalid": "The Lua adapter returned unsupported data.",
    "lua_adapter_output_too_large": "The Lua adapter output exceeds its size limit.",
    "lua_adapter_binding_changed": "The Lua adapter changed the immutable request binding.",
    "lua_adapter_isolation_unavailable": "The Lua adapter could not be isolated.",
    "lua_adapter_failed": "The Lua adapter script failed.",
}


class LuaAdapterError(ValueError):
    """Content-free diagnostic; raw Lua errors and payloads stay out of logs."""

    def __init__(self, code: str):
        self.code = code if type(code) is str and code in ERROR_MESSAGES else "lua_adapter_unavailable"
        self.message = ERROR_MESSAGES[self.code]
        super().__init__(self.code)


@dataclass(frozen=True)
class AdapterLimits:
    instructions: int = 2_000_000
    memory_bytes: int = 32 * 1024 * 1024
    output_bytes: int = MAX_INPUT_BYTES
    cpu_seconds: int = 2
    address_space_bytes: int = 768 * 1024 * 1024
    wall_seconds: float = 5.0

    def __post_init__(self):
        ceilings = {"instructions": 20_000_000, "memory_bytes": 64 * 1024 * 1024,
                    "output_bytes": MAX_INPUT_BYTES, "cpu_seconds": 10,
                    "address_space_bytes": 768 * 1024 * 1024}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= ceiling:
                raise ValueError(f"Invalid Lua adapter limit: {name}")
        if type(self.wall_seconds) not in {int, float} or not math.isfinite(self.wall_seconds) or not 0 < self.wall_seconds <= 20:
            raise ValueError("Invalid Lua adapter limit: wall_seconds")


@dataclass(frozen=True)
class AdapterResult:
    payload: Any
    receipt: dict


class WorkersBusy(Exception):
    """No worker became free within the event's wall-time limit."""


class WorkerPool:
    """Lua worker processes started ahead of time; each serves exactly one event.

    A waiting worker has finished its imports and is blocked reading its request.
    It locks itself down after reading the request, exactly like a worker started
    on demand, so while it waits it holds no request data and no authority.
    Taking a worker starts its replacement; an empty pool starts one on demand.
    At most ``active`` workers serve events at once.
    """

    def __init__(self, spares: int = SPARE_WORKERS, active: int = ACTIVE_WORKERS):
        if type(active) is not int or active < 1:
            raise ValueError("The Lua worker pool needs at least one active worker")
        self._size = spares
        self._active_limit = active
        self._active = threading.BoundedSemaphore(active)
        self._waiting: deque[subprocess.Popen] = deque()
        self._lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _launch() -> subprocess.Popen:
        return subprocess.Popen([sys.executable, "-I", "-B", str(WORKER)], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=worker_environment(),
                                cwd="/", close_fds=True, start_new_session=True)

    def _take(self) -> subprocess.Popen:
        with self._lock:
            worker = None
            while self._waiting and worker is None:
                candidate = self._waiting.popleft()
                if candidate.poll() is None:
                    worker = candidate
                else:
                    candidate.communicate()  # ended while waiting; reap it and use another
            if worker is None:
                worker = self._launch()
            try:
                while not self._closed and len(self._waiting) < self._size:
                    self._waiting.append(self._launch())
            except OSError:
                pass  # Waiting workers only save time; this event already has its own.
            return worker

    def exchange(self, request: bytes, timeout: float) -> subprocess.CompletedProcess:
        """Send one request to its own worker and collect the answer, like ``subprocess.run``.

        Waiting for a free worker counts against the same ``timeout``.
        """
        deadline = time.monotonic() + timeout
        if not self._active.acquire(timeout=timeout):
            raise WorkersBusy()
        try:
            with self._take() as worker:
                try:
                    stdout, stderr = worker.communicate(request, timeout=max(0.0, deadline - time.monotonic()))
                except BaseException:
                    worker.kill()  # A timed-out or interrupted worker never answers later.
                    raise
                return subprocess.CompletedProcess(worker.args, worker.returncode, stdout, stderr)
        finally:
            self._active.release()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            waiting, self._waiting = list(self._waiting), deque()
        for worker in waiting:
            worker.kill()
            worker.communicate()

    def _forget_after_fork(self) -> None:
        # A forked child must not use, stop or hold open its parent's workers: an
        # inherited request pipe would keep a worker from ever seeing its request end.
        for worker in self._waiting:
            for stream in (worker.stdin, worker.stdout, worker.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._waiting, self._lock = deque(), threading.Lock()
        # Slots held by the parent's threads are never returned in the child.
        self._active = threading.BoundedSemaphore(self._active_limit)


WORKERS = WorkerPool()
atexit.register(WORKERS.close)
register_at_fork = getattr(os, "register_at_fork", None)  # absent on Windows
if register_at_fork is not None:
    register_at_fork(after_in_child=WORKERS._forget_after_fork)


def is_secret_field(name: str) -> bool:
    """A credential value never enters Lua; a credential's name may (shared policy)."""
    return is_credential_value_field(name)


def _validate_json(value, depth=0, *, secrets=True):
    if depth > 16:
        raise LuaAdapterError("lua_adapter_invalid_input")
    if value is None or type(value) in {bool, str}:
        return
    if type(value) is int:
        if not -(2**63) <= value <= 2**63 - 1:
            raise LuaAdapterError("lua_adapter_invalid_input")
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _validate_json(item, depth + 1, secrets=secrets)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            if secrets and is_secret_field(key):
                raise LuaAdapterError("lua_adapter_secret_input")
            _validate_json(item, depth + 1, secrets=secrets)
        return
    raise LuaAdapterError("lua_adapter_invalid_input")


def _canonical(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeEncodeError, RecursionError):
        raise LuaAdapterError("lua_adapter_invalid_input") from None


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def run_adapter(*, kind: str, phase: str, payload: Any, binding: dict,
                source: str | None = None, limits: AdapterLimits = AdapterLimits(),
                read_links: bool = False, read_profile: bool = False,
                require_mapping: bool = False) -> AdapterResult:
    """Map one event in actual Lua; the returned payload still needs host validation.

    ``binding`` is trusted host metadata, never credentials. Source text can be
    selected by an admitted connection, but cannot change this binding. API
    credentials and passwords must be stripped/replaced by host-owned opaque
    markers before calling this function, then reattached only by the transport.
    ``read_links`` retains the link-only inspection contract. ``read_profile``
    returns both declarative links and reviewed plugin-file references.
    Named protocol plugins set ``require_mapping`` so a declarations-only
    profile cannot silently stand in for executable codec behavior.
    """
    started = time.perf_counter()
    if (type(kind) is not str or type(phase) is not str or kind not in KINDS or phase not in PHASES
            or not isinstance(limits, AdapterLimits) or type(read_links) is not bool
            or type(read_profile) is not bool or type(require_mapping) is not bool
            or (read_links and read_profile) or (require_mapping and (read_links or read_profile))):
        raise LuaAdapterError("lua_adapter_invalid_input")
    if type(binding) is not dict or not binding:
        raise LuaAdapterError("lua_adapter_binding_invalid")
    _validate_json(binding)
    _validate_json(payload)
    if len(_canonical(binding)) > MAX_BINDING_BYTES:
        raise LuaAdapterError("lua_adapter_binding_invalid")
    if len(_canonical(payload)) > MAX_INPUT_BYTES:
        raise LuaAdapterError("lua_adapter_input_too_large")
    if source is None:
        try:
            source = DEFAULT_SOURCE.read_bytes().decode("utf-8")
        except (OSError, UnicodeError):
            raise LuaAdapterError("lua_adapter_unavailable") from None
    try:
        valid_source = type(source) is str and bool(source.strip()) and len(source.encode("utf-8")) <= MAX_SOURCE_BYTES and SOURCE_PATTERN.fullmatch(source)
    except UnicodeEncodeError:
        valid_source = False
    if not valid_source:
        raise LuaAdapterError("lua_adapter_invalid_source")
    # Snapshot caller-owned structures before dispatch to another thread/process.
    event = json.loads(_canonical({"schema": EVENT_SCHEMA, "kind": kind, "phase": phase,
                                   "binding": binding, "payload": payload}))
    request = {"source": source, "event": event, "limits": asdict(limits)}
    if read_links or read_profile:
        request["read"] = "profile" if read_profile else "links"
    if require_mapping:
        request["require_mapping"] = True
    try:
        completed = WORKERS.exchange(_canonical(request), timeout=limits.wall_seconds)
    except WorkersBusy:
        raise LuaAdapterError("lua_adapter_busy") from None
    except subprocess.TimeoutExpired:
        raise LuaAdapterError("lua_adapter_time_limit") from None
    except OSError:
        raise LuaAdapterError("lua_adapter_unavailable") from None
    if len(completed.stdout) > limits.output_bytes + MAX_BINDING_BYTES + 4000:
        raise LuaAdapterError("lua_adapter_output_too_large")
    try:
        answer = json.loads(completed.stdout)
    except (ValueError, UnicodeDecodeError):
        answer = None
    if type(answer) is not dict:
        code = ("lua_adapter_time_limit" if completed.returncode == -signal.SIGXCPU else
                "lua_adapter_worker_terminated" if completed.returncode < 0 else "lua_adapter_unavailable")
        raise LuaAdapterError(code)
    if "error" in answer:
        raise LuaAdapterError(answer["error"])
    if completed.returncode != 0 or set(answer) != {"schema", "event", "elapsed_ms", "isolation"}:
        raise LuaAdapterError("lua_adapter_unavailable")
    result = answer["event"]
    if answer["schema"] != "lua-adapter-worker.v1" or type(result) is not dict or set(result) != set(event):
        raise LuaAdapterError("lua_adapter_output_invalid")
    if any(_canonical(result[key]) != _canonical(event[key]) for key in ("schema", "kind", "phase", "binding")):
        raise LuaAdapterError("lua_adapter_binding_changed")
    try:
        _validate_json(result["payload"])
    except LuaAdapterError:
        raise LuaAdapterError("lua_adapter_output_invalid") from None
    if len(_canonical(result["payload"])) > limits.output_bytes:
        raise LuaAdapterError("lua_adapter_output_too_large")
    layers, elapsed = answer["isolation"], answer["elapsed_ms"]
    if type(layers) is not list or not all(type(item) is str for item in layers) or not REQUIRED_ISOLATION <= set(layers):
        raise LuaAdapterError("lua_adapter_isolation_unavailable")
    if type(elapsed) is not int or elapsed < 0:
        raise LuaAdapterError("lua_adapter_output_invalid")
    receipt = {"schema": RECEIPT_SCHEMA, "runtime": "lua54-isolated.v1", "kind": kind, "phase": phase,
               "script_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
               "binding_sha256": _digest(event["binding"]), "input_sha256": _digest(event["payload"]),
               "output_sha256": _digest(result["payload"]),
               "elapsed_ms": round((time.perf_counter() - started) * 1000),
               "worker_elapsed_ms": elapsed, "isolation": list(layers)}
    return AdapterResult(result["payload"], receipt)


async def run_adapter_async(**kwargs) -> AdapterResult:
    """Keep bounded worker I/O off the application's event loop."""
    return await asyncio.to_thread(run_adapter, **kwargs)
