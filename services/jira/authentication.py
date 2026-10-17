"""Jira Cloud authentication guard shared by every Jira HTTP client.

Jira Cloud does not reject a request whose Basic credentials are wrong.
It serves the request anonymously and reports the failure only in the
``X-Seraph-LoginReason`` header. An anonymous project search then returns
HTTP 200 with zero projects, which would look like a successful empty read.

This hook converts those responses into HTTP 401 before any client logic
sees them, so every caller reports ``authentication_failed`` instead of
silently acting on anonymous data.
"""

from __future__ import annotations

import httpx


_FAILED_LOGIN_REASONS = frozenset({
    "AUTHENTICATED_FAILED",
    "AUTHENTICATION_DENIED",
})


def failed_login_reason(response: httpx.Response) -> str | None:
    header = response.headers.get("x-seraph-loginreason", "")
    for reason in header.replace(",", " ").split():
        if reason.strip().upper() in _FAILED_LOGIN_REASONS:
            return reason.strip().upper()
    return None


def reject_anonymous_fallback(response: httpx.Response) -> None:
    """httpx response hook: a failed login is an authentication failure."""
    if failed_login_reason(response) is not None and response.status_code < 400:
        response.status_code = 401


AUTH_EVENT_HOOKS = {"response": [reject_anonymous_fallback]}
