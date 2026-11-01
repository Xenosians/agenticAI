"""Send a provider request again when the provider turned it away.

Only failures where the provider did not do the work are retried: it refused
the request (rate limit), was overloaded or failed with a server error, or the
connection could not be made. A timeout is not retried, because the provider
may still be working on the request and a second one would be paid for twice.
Authentication, billing, truncation and malformed answers would fail the same
way again. Each retry waits twice as long as the one before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from services.extensions.diagnostics import ExternalProviderError, HarnessFailure

RETRYABLE = frozenset({
    "provider_rate_limited", "provider_overloaded", "provider_unavailable", "provider_connection_failed",
})
DEFAULT_RETRY_WAIT_SECONDS = 1.0

T = TypeVar("T")
_sleep = asyncio.sleep


def retryable(error: BaseException) -> bool:
    return isinstance(error, (ExternalProviderError, HarnessFailure)) and error.category in RETRYABLE


async def with_retries(profile: dict, send: Callable[[], Awaitable[T]]) -> T:
    """Run ``send``; repeat it after a turned-away request, up to the connection's limit."""
    attempts = int(profile.get("retry_attempts") or 0)
    wait = float(profile.get("retry_wait_seconds", DEFAULT_RETRY_WAIT_SECONDS))
    for attempt in range(attempts + 1):
        try:
            return await send()
        except (ExternalProviderError, HarnessFailure) as error:
            if attempt == attempts or not retryable(error):
                raise
            print(f"[PROVIDER] Retrying category={error.category} attempt={attempt + 1}/{attempts}")
            await _sleep(wait * 2 ** attempt)
    raise AssertionError("unreachable")
