"""Secret-safe diagnostics for external inference.

Provider and harness diagnostics may be written to server logs, but credentials
must never become exception text, durable job payloads, frontend responses, or
provider-request payloads.

Model/provider output remains data, never authority.
"""

from __future__ import annotations

import re
from typing import Iterable


_REDACTION_PATTERNS: tuple[
    tuple[re.Pattern[str], str],
    ...
] = (
    (
        re.compile(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"
        ),
        "Bearer [REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|password|secret|credential|authorization)"
            r"\s*[:=]\s*['\"]?[^,'\"\s}\]]{6,}"
        ),
        r"\1=[REDACTED]",
    ),
    (
        re.compile(
            r"\bsk-[A-Za-z0-9_-]{12,}\b"
        ),
        "[REDACTED_KEY]",
    ),
    (
        re.compile(
            r"\borg_[A-Za-z0-9_-]{8,}\b"
        ),
        "[REDACTED_ORG]",
    ),
    (
        re.compile(
            r"\breq_[A-Za-z0-9_-]{8,}\b"
        ),
        "[REDACTED_REQUEST]",
    ),
)


def sanitize_diagnostic(
    value: object,
    *,
    secrets: Iterable[str] = (),
    max_chars: int = 3000,
) -> str:
    text = str(value)

    for secret in secrets:
        if (
            isinstance(secret, str)
            and secret
        ):
            text = text.replace(
                secret,
                "[REDACTED]",
            )

    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(
            replacement,
            text,
        )

    text = text.replace(
        "\x00",
        "",
    )

    if len(text) > max_chars:
        text = (
            text[:max_chars]
            + "...[TRUNCATED]"
        )

    return text


def classify_external_failure(
    text: str,
    *,
    status_code: int | None = None,
) -> str:
    lower = text.lower()

    if (
        "tokens per minute" in lower
        or "request too large" in lower
        or (
            "rate_limit_exceeded" in lower
            and "token" in lower
        )
    ):
        return (
            "provider_token_budget_exceeded"
        )

    if (
        status_code == 429
        or "rate limit" in lower
        or "too many requests" in lower
    ):
        return (
            "provider_rate_limited"
        )

    if (
        status_code in {
            401,
            403,
        }
        or "unauthorized" in lower
        or "forbidden" in lower
        or "invalid api key" in lower
        or "authentication" in lower
    ):
        return (
            "provider_authentication_failed"
        )

    if (
        "model not found" in lower
        or "model_not_found" in lower
        or "unknown model" in lower
        or (
            "model" in lower
            and "does not exist" in lower
        )
    ):
        return (
            "provider_model_unavailable"
        )

    if (
        "timeout" in lower
        or "timed out" in lower
    ):
        return (
            "provider_timeout"
        )

    if (
        "connection refused" in lower
        or "network" in lower
        or "dns" in lower
        or "connection failed" in lower
    ):
        return (
            "provider_connection_failed"
        )

    if (
        "invalid json" in lower
        or "malformed" in lower
    ):
        return (
            "provider_malformed_response"
        )

    return (
        "provider_error"
    )


class ExternalProviderError(
    ValueError
):
    """Public exception with secret-free text."""

    def __init__(
        self,
        category: str,
        *,
        status_code: int | None = None,
    ):
        self.category = category
        self.status_code = status_code

        suffix = (
            f", HTTP {status_code}"
            if status_code is not None
            else ""
        )

        super().__init__(
            "External provider request failed "
            f"({category}{suffix})"
        )


class HarnessFailure(
    ValueError
):
    """Public harness exception with secret-free text."""

    def __init__(
        self,
        backend: str,
        provider: str,
        category: str,
        *,
        exit_code: int | None = None,
    ):
        self.backend = backend
        self.provider = provider
        self.category = category
        self.exit_code = exit_code

        exit_suffix = (
            f", exit={exit_code}"
            if exit_code is not None
            else ""
        )

        super().__init__(
            "External inference harness failed "
            f"(backend={backend}, "
            f"provider={provider}, "
            f"category={category}"
            f"{exit_suffix})"
        )
