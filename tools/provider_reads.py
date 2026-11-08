"""Provider reads end with observed data or a typed failure that says why.

An unhandled exception inside an MCP tool reaches the gateway as an unconfirmed
outcome ("reconcile before retrying"). That is right for a change, whose effect
may have happened, and wrong for a read, which changes nothing: the user would
be told to reconcile a lookup. Read tools therefore turn the provider's known
failures into a result that names the cause, and a missing setup into one that
names the missing settings.

Messages are fixed or come from the provider adapters, which never echo the
provider's answer or credentials; transport errors are reduced to their kind,
because their text can carry the provider URL. An unexpected error (a bug)
still raises, so it stays visible instead of looking like a provider failure.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from json import JSONDecodeError

import httpx

from services.lua_adapters.runtime import LuaAdapterError

UNREACHABLE = "could not be reached, or it closed the connection before answering."


def failure(provider: str, error_code: str, error: str) -> dict:
    return {"ok": False, "status": "error", "provider": provider, "error_code": error_code, "error": error}


def not_configured(provider: str, label: str, missing: Iterable[str] = ()) -> dict:
    names = [name for name in missing if isinstance(name, str) and name]
    detail = f" Missing settings: {', '.join(names)}." if names else ""
    return failure(provider, "not_configured", f"{label} is not set up on this Janus host.{detail}")


def _failure_of(provider: str, label: str, error: Exception) -> dict | None:
    if isinstance(error, LuaAdapterError):
        return failure(provider, error.code, error.message)
    if isinstance(error, httpx.TimeoutException):
        return failure(provider, "provider_timeout", f"{label} did not answer in time.")
    if isinstance(error, httpx.HTTPStatusError):
        return failure(provider, "provider_refused", f"{label} refused the read (HTTP {error.response.status_code}).")
    if isinstance(error, (httpx.TransportError, ConnectionError)):
        return failure(provider, "provider_unreachable", f"{label} {UNREACHABLE}")
    if isinstance(error, (JSONDecodeError, UnicodeDecodeError)):
        return failure(provider, "provider_error", f"{label} returned an invalid answer.")
    if isinstance(error, RuntimeError):
        # The adapters' own fixed messages (refusals, malformed or oversized answers).
        return failure(provider, "provider_error", str(error))
    if isinstance(error, ValueError):
        # An argument the adapter rejected before sending anything.
        return failure(provider, "invalid_argument", str(error))
    return None


def _typed_failure(provider: str, label: str, error: Exception) -> dict:
    result = _failure_of(provider, label, error)
    if result is None:
        raise error
    return result


def provider_read(provider: str, label: str, read: Callable[[], dict]) -> dict:
    """Run one read; a dropped connection is read once more, since a read has no effect to repeat."""
    try:
        return read()
    except (httpx.RemoteProtocolError, httpx.ConnectError):
        # A server may close an idle kept-alive connection just as the request
        # goes out; one more read on a fresh connection settles it.
        pass
    except Exception as error:
        return _typed_failure(provider, label, error)
    try:
        return read()
    except Exception as error:
        return _typed_failure(provider, label, error)
