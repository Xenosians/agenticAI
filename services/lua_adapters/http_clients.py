"""Project HTTP clients: every actual request crosses the Lua adapter first.

Credentials, headers and request bodies stay with the host transport. The Lua
event describes the exact link, and cannot redirect an already admitted target.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import os
import weakref

import httpx

from services.lua_adapters.host_boundaries import inspect_data, inspect_data_async
from services.transport_tls import configured_context

# Shared clients keep connections open between calls, so a call does not pay a
# new TCP and TLS handshake first (33-52 ms to the model providers from the
# development host). An idle connection is closed after this many seconds.
KEEPALIVE_SECONDS = 30.0

_shared_clients: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


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
    if options.get("verify", True) is True:
        context = configured_context()
        if context is not None:
            options["verify"] = context
    hooks = {name: list(callbacks) for name, callbacks in options.pop("event_hooks", {}).items()}
    # Run last, so the adapter observes the request after other host hooks.
    hooks.setdefault("request", []).append(hook)
    options["event_hooks"] = hooks
    return options


def client(*args, **kwargs):
    return httpx.Client(*args, **_options(kwargs, _request))


def async_client(*args, **kwargs):
    return httpx.AsyncClient(*args, **_options(kwargs, _async_request))


@asynccontextmanager
async def shared_async_client(*, timeout, follow_redirects=False, trust_env=False):
    """Borrow this event loop's reusable client; leaving the block does not close it.

    Each request through it still crosses the Lua connection adapter. Clients are
    kept per event loop (a connection belongs to the loop that opened it) and per
    option set, including the configured CA bundle, so a trust change takes effect
    on the next call. close_shared_clients() closes them when the service stops.
    """
    clients = _shared_clients.setdefault(asyncio.get_running_loop(), {})
    key = (timeout, follow_redirects, trust_env, os.environ.get("ITSM_HTTP_CA_BUNDLE"))
    client = clients.get(key)
    if client is None or client.is_closed:
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=KEEPALIVE_SECONDS)
        client = async_client(timeout=timeout, follow_redirects=follow_redirects, trust_env=trust_env, limits=limits)
        clients[key] = client
    yield client


async def close_shared_clients():
    """Close this event loop's shared clients (service shutdown, end of a script or test)."""
    for client in _shared_clients.pop(asyncio.get_running_loop(), {}).values():
        await client.aclose()
