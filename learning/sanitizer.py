from __future__ import annotations

import re

from typing import (
    Any,
)


REDACTED = (
    "<redacted>"
)

REDACTED_EMAIL = (
    "<redacted-email>"
)


SENSITIVE_KEY_NAMES = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "set_cookie",
    "credentials",
    "credential",
    "private_key",
    "client_secret",
    "access_token",
    "refresh_token",
    "jira_api_token",
    "ad_bind_password",
    "ad_write_bind_password",
    "itsm_internal_job_token",
}


BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
)

BASIC_PATTERN = re.compile(
    r"(?i)\bBasic\s+[A-Za-z0-9+/=]+"
)

SECRET_ASSIGNMENT_PATTERN = re.compile(
    (
        r"(?i)"
        r"\b("
        r"password|passwd|secret|"
        r"api[_-]?key|token|"
        r"client[_-]?secret"
        r")"
        r"\s*[:=]\s*"
        r"([^\s,;]+)"
    )
)

EMAIL_PATTERN = re.compile(
    (
        r"\b"
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@"
        r"[A-Za-z0-9-]+"
        r"(?:\.[A-Za-z0-9-]+)+"
        r"\b"
    )
)

URL_USERINFO_PATTERN = re.compile(
    (
        r"(?i)"
        r"(https?://)"
        r"[^/\s:@]+:"
        r"[^@\s/]+@"
    )
)


def _normalized_key(
    value: str,
) -> str:

    return (
        re.sub(
            r"[^a-z0-9]+",
            "_",
            value
            .strip()
            .lower(),
        )
        .strip("_")
    )


def _sensitive_key(
    value: str,
) -> bool:

    normalized = (
        _normalized_key(
            value
        )
    )

    if (
        normalized
        in SENSITIVE_KEY_NAMES
    ):
        return True

    sensitive_suffixes = (
        "_password",
        "_secret",
        "_token",
        "_api_key",
        "_private_key",
    )

    return (
        normalized.endswith(
            sensitive_suffixes
        )
    )


def sanitize_text(
    value: str,
) -> str:

    result = (
        value
    )

    result = (
        BEARER_PATTERN.sub(
            "Bearer <redacted>",
            result,
        )
    )

    result = (
        BASIC_PATTERN.sub(
            "Basic <redacted>",
            result,
        )
    )

    result = (
        SECRET_ASSIGNMENT_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}="
                f"{REDACTED}"
            ),
            result,
        )
    )

    result = (
        URL_USERINFO_PATTERN.sub(
            lambda match: (
                f"{match.group(1)}"
                "<redacted>@"
            ),
            result,
        )
    )

    result = (
        EMAIL_PATTERN.sub(
            REDACTED_EMAIL,
            result,
        )
    )

    return (
        result
    )


def sanitize_value(
    value: Any,
    *,
    key: str | None = None,
) -> Any:
    """
    Recursively sanitize one trajectory payload.

    Sanitization happens before persistence.

    Secret-bearing fields are replaced wholesale.
    Free-form strings receive conservative credential-pattern
    redaction.
    """

    if (
        key is not None
        and _sensitive_key(
            key
        )
    ):
        return (
            REDACTED
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                child_key
            ):
                sanitize_value(
                    child_value,
                    key=str(
                        child_key
                    ),
                )

            for (
                child_key,
                child_value,
            )
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            sanitize_value(
                item
            )

            for item
            in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            sanitize_value(
                item
            )

            for item
            in value
        ]

    if isinstance(
        value,
        str,
    ):
        return (
            sanitize_text(
                value
            )
        )

    return (
        value
    )