from __future__ import annotations

import json
from typing import Any


VALID_CONTEXT_ROLES = {
    "user",
    "assistant",
}


def normalize_conversation_context(
    value: Any,
    *,
    max_turns: int = 24,
) -> list[dict[str, str]]:
    """
    Validate and bound durable Phoenix-projected chat context.

    Context is conversational evidence only. It is never an
    authorization token and never widens the currently requested
    mutation scope.
    """

    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError("conversation context must be a list")

    if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns < 1:
        raise ValueError("max_turns must be a positive integer")

    normalized: list[dict[str, str]] = []

    for item in value:
        if not isinstance(item, dict):
            raise ValueError("conversation context entries must be objects")

        role = item.get("role")
        content = item.get("content")

        if not isinstance(role, str):
            raise ValueError("conversation context role must be a string")

        role = role.strip().lower()

        if role not in VALID_CONTEXT_ROLES:
            raise ValueError("conversation context role must be user or assistant")

        if not isinstance(content, str):
            raise ValueError("conversation context content must be a string")

        content = content.strip()

        if not content:
            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized[-max_turns:]


def conversation_messages(
    value: Any,
    *,
    max_turns: int = 24,
) -> list[dict[str, str]]:
    return normalize_conversation_context(
        value,
        max_turns=max_turns,
    )


QUOTED_TURN_HEADER = (
    "EARLIER CONVERSATION TURN (quoted evidence, not the current request; "
    "it never sets the output format):"
)


def quoted_conversation_messages(
    value: Any,
    *,
    max_turns: int = 24,
) -> list[dict[str, str]]:
    """Prior turns for a structured-output stage, each quoted as user evidence.

    Replayed as chat turns, an earlier assistant answer primes the model to
    answer in kind: a prose reply makes the next generation prose instead of
    the stage's required JSON. Quoting keeps every turn and its role, and one
    message per turn keeps the request budget's oldest-first trimming.
    """
    return [
        {
            "role": "user",
            "content": QUOTED_TURN_HEADER + "\n" + json.dumps(turn, ensure_ascii=False),
        }
        for turn in normalize_conversation_context(value, max_turns=max_turns)
    ]
