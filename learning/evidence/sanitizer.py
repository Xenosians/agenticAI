from __future__ import annotations

import base64
import binascii
import re

from typing import (
    Any,
)

from services.sensitive_fields import (
    SENSITIVE_FIELD_NAMES as SENSITIVE_KEY_NAMES,
    is_sensitive_field,
)


REDACTED = (
    "<redacted>"
)

REDACTED_EMAIL = (
    "<redacted-email>"
)


BEARER_PATTERN = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"
)

# In an Authorization header, whatever follows Basic is a credential.
AUTHORIZATION_BASIC_PATTERN = re.compile(
    r"(?i)\b(Authorization\s*:\s*Basic)\s+[A-Za-z0-9+/=]+"
)

# Any (Proxy-)Authorization value, as a header, an assignment or a quoted
# mapping entry, is a credential. A leading scheme word (Digest, Token, ...)
# is kept so the redacted text still says what kind of credential it was.
AUTHORIZATION_VALUE_PATTERN = re.compile(
    r"(?i)\b((?:proxy[_-])?authorization['\"]?\s*(?::=|[:=])\s*)"
    # A quoted value runs to its closing quote; a header value to the line end.
    r"(?:(['\"])([^\r\n]*?)\2|([^\r\n]+))"
)
AUTHORIZATION_SCHEME_PATTERN = re.compile(r"([A-Za-z][\w.~+/-]*)[ \t]+\S")


def _redact_authorization_value(match: re.Match) -> str:
    name, quote, quoted, bare = match.groups()
    value = quoted if quote else bare
    if REDACTED in value:
        return match.group(0)  # Already redacted by the Bearer/Basic patterns.
    scheme = AUTHORIZATION_SCHEME_PATTERN.match(value)
    kept = f"{scheme.group(1)} " if scheme else ""
    return f"{name}{quote or ''}{kept}{REDACTED}{quote or ''}"

# Elsewhere "Basic" is often an ordinary word ("Basic setup"); the token after
# it is a credential only when _basic_credential decodes it to user:password.
BASIC_PATTERN = re.compile(
    r"(?i)\bBasic\s+([A-Za-z0-9+/]+=*)"
)

SECRET_ASSIGNMENT_PATTERN = re.compile(
    (
        r"(?i)"
        r"\b("
        # Authorization headers are redacted by the header patterns above;
        # matching them here again would redact the scheme, not the credential.
        r"password|passwd|pwd|secret|credential|cookie|"
        r"private[_-]?key|api[_-]?key|token|"
        r"client[_-]?secret"
        r")"
        r"\s*(?::=|[:=])\s*"
        # A quoted value is redacted whole, an unterminated one from its quote;
        # an empty one ("", '', “”) holds no secret.
        r"("
        r"\"[^\"\n]+\"|'[^'\n]+'"
        r"|\u201c[^\u201d\n]+\u201d|\u2018[^\u2019\n]+\u2019"
        r"|[\"'\u201c\u2018]?[^\s,;\"'\u201c\u201d\u2018\u2019][^\s,;]*"
        r")"
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


def is_sensitive_key(
    value: str,
) -> bool:
    return is_sensitive_field(value)


def _basic_credential(
    token: str,
) -> bool:
    """
    Whether the token after "Basic" is a credential rather than a word:
    the base64 form of user-id ":" password in UTF-8 (RFC 7617), with or
    without padding, or a long opaque token such as an API key (16 or
    more characters with a digit, "+", "/", "=" or an inner capital).
    """

    if (
        len(token) >= 16
        and (
            re.search(r"[0-9+/=]", token)
            or any(character.isupper() for character in token[1:])
        )
    ):
        return True

    data = token.rstrip(
        "="
    )

    try:
        decoded = base64.b64decode(
            data + "=" * (-len(data) % 4),
            validate=True,
        ).decode(
            "utf-8"
        )
    except (binascii.Error, UnicodeDecodeError):
        return False

    return (
        ":" in decoded
        and decoded.isprintable()
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
        AUTHORIZATION_BASIC_PATTERN.sub(
            lambda match: (
                f"{match.group(1)} <redacted>"
            ),
            result,
        )
    )

    result = (
        BASIC_PATTERN.sub(
            lambda match: (
                "Basic <redacted>"
                if _basic_credential(
                    match.group(1)
                )
                else match.group(0)
            ),
            result,
        )
    )

    result = AUTHORIZATION_VALUE_PATTERN.sub(_redact_authorization_value, result)

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
        and is_sensitive_key(
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
