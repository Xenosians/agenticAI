"""Shared field-name policy for values kept out of untrusted processing boundaries."""

from __future__ import annotations

import re


SENSITIVE_FIELD_NAMES = frozenset(
    {
        "access_token",
        "ad_bind_password",
        "ad_write_bind_password",
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "itsm_internal_job_token",
        "jira_api_token",
        "password",
        "password_hash",
        "passwd",
        "private_key",
        "proxy_authorization",
        "pwd",
        "refresh_token",
        "secret",
        "set_cookie",
    }
)

# Field names whose value is itself a credential. The Lua boundary refuses
# exactly these, so a credential value never enters an adapter. It is narrower
# than the redaction policy on purpose: a machine profile names a protected
# credential with `secret = "<name>"`, and provider data carries fields such as
# `next_page_token` or `has_password` that are not credentials.
CREDENTIAL_VALUE_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer_token",
        "client_secret",
        "credentials",
        "password",
        "password_hash",
        "private_key",
        "proxy_authorization",
        "refresh_token",
    }
)

_SENSITIVE_SUFFIXES = (
    "_password",
    "_secret",
    "_token",
    "_api_key",
    "_private_key",
)
_SENSITIVE_MARKERS = (
    "_authorization_",
    "_cookie_",
    "_password_",
    "_passwd_",
    "_private_key_",
    "_secret_",
    "_token_",
    "_api_key_",
)
_SENSITIVE_PREFIXES = (
    "api_key_",
    "private_key_",
    "secret_",
)


def is_sensitive_field(name: str) -> bool:
    """Recognize credential field names in snake, kebab or camel case."""
    if type(name) is not str:
        return False

    normalized = _normalized(name)
    return (
        normalized in SENSITIVE_FIELD_NAMES
        or normalized.endswith(_SENSITIVE_SUFFIXES)
        or normalized.startswith(_SENSITIVE_PREFIXES)
        or any(marker in normalized for marker in _SENSITIVE_MARKERS)
    )


def is_credential_value_field(name: str) -> bool:
    """Whether a field holds a credential value, in snake, kebab or camel case."""
    return type(name) is str and _normalized(name) in CREDENTIAL_VALUE_FIELD_NAMES


def _normalized(name: str) -> str:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
