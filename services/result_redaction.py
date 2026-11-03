"""Remove credential material before provider results enter later model prompts."""

from __future__ import annotations

from typing import Any

from services.log_redaction import redact_log_text
from services.sensitive_fields import is_sensitive_field


def redact_result_value(value: Any) -> tuple[Any, bool]:
    """Return JSON-shaped result data with secret fields and common secret text removed.

    User identifiers, email addresses, ticket keys and ordinary provider fields
    are preserved. The boolean reports whether any detail was redacted.
    """
    if isinstance(value, dict):
        redacted = False
        sanitized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str) or is_sensitive_field(key):
                redacted = True
                continue
            sanitized_child, child_redacted = redact_result_value(child)
            sanitized[key] = sanitized_child
            redacted = redacted or child_redacted
        return sanitized, redacted

    if isinstance(value, list):
        sanitized_items = [redact_result_value(item) for item in value]
        return (
            [item for item, _ in sanitized_items],
            any(was_redacted for _, was_redacted in sanitized_items),
        )

    if isinstance(value, str):
        sanitized, redacted = redact_log_text(value)
        return sanitized, redacted

    return value, False
