"""Project HTTP clients: every actual request crosses the Lua adapter first.

Credentials, headers and request bodies stay with the host transport. The Lua
event describes the exact link, and cannot redirect an already admitted target.
"""
from __future__ import annotations

import hashlib

import httpx

from services.lua_adapters.host_boundaries import inspect_data, inspect_data_async


def _event(request):
    url = request.url
    host = f"[{url.host}]" if ":" in url.host else url.host
    origin = f"{url.scheme}://{host}" + (f":{url.port}" if url.port is not None else "")
    payload = {"origin": origin, "path_sha256": hashlib.sha256(url.path.encode()).hexdigest(), "method": request.method,
               "query_fields": sorted(set(url.params.keys()))}
    binding = {"transport": "httpx", "origin": origin, "method": request.method,
               "target_sha256": hashlib.sha256(str(url).encode()).hexdigest()}
    return payload, binding


def _request(request):
    payload, binding = _event(request)
    request.extensions["lua_adapter_receipt"] = inspect_data("connection", "request", payload, binding)


async def _async_request(request):
    payload, binding = _event(request)
    request.extensions["lua_adapter_receipt"] = await inspect_data_async("connection", "request", payload, binding)


def _options(options, hook):
    options = dict(options)
    hooks = {name: list(callbacks) for name, callbacks in options.pop("event_hooks", {}).items()}
    # Run last, so the adapter observes the request after other host hooks.
    hooks.setdefault("request", []).append(hook)
    options["event_hooks"] = hooks
    return options


def client(*args, **kwargs):
    return httpx.Client(*args, **_options(kwargs, _request))


def async_client(*args, **kwargs):
    return httpx.AsyncClient(*args, **_options(kwargs, _async_request))
