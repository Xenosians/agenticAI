"""Mandatory Lua adapters with host-owned identity, secrets and effect receipts.

Lua translates data; existing host contracts still authorize and dispatch it.
No interpreter, script or mapping failure selects a Python-only fallback.
"""
from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re

from services.lua_adapters.runtime import LuaAdapterError, is_secret_field, run_adapter, run_adapter_async

LOG = logging.getLogger(__name__)
MAX_SCRIPT_BYTES = 20_000


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def configured_source():
    """A configured adapter is an explicit immutable operator artifact."""
    filename = os.environ.get("ITSM_LUA_ADAPTER_FILE", "")
    digest = os.environ.get("ITSM_LUA_ADAPTER_SHA256", "")
    if not filename and not digest:
        return None  # The actual built-in Lua adapter, never a Python bypass.
    if not filename or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise LuaAdapterError("lua_adapter_configuration_invalid")
    path = Path(filename)
    if not path.is_absolute() or path.is_symlink():
        raise LuaAdapterError("lua_adapter_configuration_invalid")
    try:
        with path.open("rb") as handle:
            content = handle.read(MAX_SCRIPT_BYTES + 1)
        if len(content) > MAX_SCRIPT_BYTES or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError
        return content.decode("utf-8")
    except (OSError, ValueError, UnicodeError):
        raise LuaAdapterError("lua_adapter_configuration_invalid") from None


def _public(value):
    """Omit credential fields entirely; keep their values only in this host."""
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if not is_secret_field(key)}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _record(result):
    receipt = result.receipt
    LOG.info("lua_adapter kind=%s phase=%s script=%s input=%s output=%s elapsed_ms=%s",
             receipt["kind"], receipt["phase"], receipt["script_sha256"],
             receipt["input_sha256"], receipt["output_sha256"], receipt["elapsed_ms"])


def inspect_data(kind, phase, payload, binding):
    """Approved targets and effect receipts cannot be rewritten by an adapter."""
    public = _public(payload)
    result = run_adapter(kind=kind, phase=phase, payload=public, binding=binding, source=configured_source())
    if _canonical(result.payload) != _canonical(public):
        raise LuaAdapterError("lua_adapter_binding_changed")
    _record(result)
    return result.receipt


async def inspect_data_async(kind, phase, payload, binding):
    return await asyncio.to_thread(inspect_data, kind, phase, payload, binding)


def inspect_link(transport: str, origin: str, *, operation: str, details: dict | None = None):
    """Every non-HTTP outbound link (directory, mail, git) crosses Lua before connecting.

    Only the destination and operation enter Lua; credentials stay with the caller,
    and a refusal stops the connection before it is opened.
    """
    payload = {"origin": origin, "operation": operation, **(details or {})}
    return inspect_data("connection", "request", payload, {"transport": transport, "origin": origin})


async def model_request(messages, max_new_tokens, *, binding, warm=False):
    original = copy.deepcopy(messages)
    result = await run_adapter_async(kind="model", phase="warm" if warm else "request",
        payload={"messages": original, "max_new_tokens": max_new_tokens}, binding=binding,
        source=configured_source())
    value = result.payload
    if not isinstance(value, dict) or set(value) != {"messages", "max_new_tokens"}:
        raise LuaAdapterError("lua_adapter_output_invalid")
    rows, limit = value["messages"], value["max_new_tokens"]
    if type(limit) is not int or not 0 <= limit <= max_new_tokens or (not warm and limit == 0):
        raise LuaAdapterError("lua_adapter_output_invalid")
    if not isinstance(rows, list) or len(rows) != len(original):
        raise LuaAdapterError("lua_adapter_binding_changed")
    # History, system instructions and versioned references keep their identities.
    if _canonical(rows[:-1]) != _canonical(original[:-1]):
        raise LuaAdapterError("lua_adapter_binding_changed")
    if rows:
        row, previous = rows[-1], original[-1]
        if (not isinstance(row, dict) or set(row) != set(previous) or row.get("role") != previous.get("role")
                or not isinstance(row.get("content"), str)
                or any(_canonical(row[key]) != _canonical(previous[key]) for key in row if key != "content")):
            raise LuaAdapterError("lua_adapter_output_invalid")
    _record(result)
    return rows, limit


async def model_response(text, *, binding):
    result = await run_adapter_async(kind="model", phase="response", payload={"text": text},
        binding=binding, source=configured_source())
    if not isinstance(result.payload, dict) or set(result.payload) != {"text"} or not isinstance(result.payload["text"], str):
        raise LuaAdapterError("lua_adapter_output_invalid")
    _record(result)
    return result.payload["text"]


def adapt_tool_callable(fn):
    """All registered MCP capabilities use Lua before their real implementation.

    A post-effect mapping failure keeps the actual provider receipt. It never
    makes an acknowledged mutation look safely retryable.
    """
    signature = inspect.signature(fn)
    binding = {"capability": fn.__name__}

    def arguments(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def response(value):
        wire = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        try:
            inspect_data("tool", "response", wire, binding)
        except (LuaAdapterError, TypeError, ValueError):
            LOG.error("lua_adapter_response_failed capability=%s actual_provider_receipt_retained=true", fn.__name__)
        return value

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_call(*args, **kwargs):
            await inspect_data_async("tool", "request", arguments(args, kwargs), binding)
            value = await fn(*args, **kwargs)
            return await asyncio.to_thread(response, value)
        return async_call

    @functools.wraps(fn)
    def sync_call(*args, **kwargs):
        inspect_data("tool", "request", arguments(args, kwargs), binding)
        return response(fn(*args, **kwargs))
    return sync_call
