"""Best-effort redaction of common secret forms in service log text."""

from __future__ import annotations

import re


SENSITIVE_ASSIGNMENT_PATTERN = re.compile(
    (
        r"(?i)"
        r"\b("
        r"password|passwd|pwd|secret|token|"
        r"api[_-]?key|authorization"
        r")"
        r"(\s*[:=]\s*)"
        r"((?:Bearer\s+)?[^\s,;]+)"
    )
)


BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
)


URL_CREDENTIAL_PATTERN = re.compile(
    (
        r"(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)"
        r"(?P<user>[^/\s:@]+)"
        r":"
        r"(?P<password>[^/\s@]+)"
        r"@"
    )
)


def redact_log_text(
    value: str,
) -> tuple[
    str,
    bool,
]:
    """
    Apply deterministic best-effort redaction for common secret
    forms before service log output leaves the trusted adapter.

    This is intentionally conservative but not claimed to be an
    exhaustive secret-detection system.
    """

    redacted = False

    def replace_assignment(
        match: re.Match,
    ) -> str:
        nonlocal redacted

        redacted = True

        return (
            f"{match.group(1)}"
            f"{match.group(2)}"
            "[REDACTED]"
        )

    value = (
        SENSITIVE_ASSIGNMENT_PATTERN
        .sub(
            replace_assignment,
            value,
        )
    )

    if BEARER_PATTERN.search(
        value
    ):
        redacted = True

        value = (
            BEARER_PATTERN.sub(
                "Bearer [REDACTED]",
                value,
            )
        )

    def replace_url_credentials(
        match: re.Match,
    ) -> str:
        nonlocal redacted

        redacted = True

        return (
            f"{match.group('scheme')}"
            f"{match.group('user')}"
            ":[REDACTED]@"
        )

    value = (
        URL_CREDENTIAL_PATTERN
        .sub(
            replace_url_credentials,
            value,
        )
    )

    return (
        value,
        redacted,
    )
