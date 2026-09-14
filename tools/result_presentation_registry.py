from __future__ import annotations

from typing import Any

from tools.registry import (
    get_tool,
)

from tools.result_cards import (
    validate_result_card,
)


def build_tool_presentation(
    tool_name: str,
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
] | None:
    """
    Build an optional canonical presentation for a successful
    trusted tool result.

    Presentation failure must never change the authoritative
    outcome of an already-successful tool operation.
    """

    tool = (
        get_tool(
            tool_name
        )
    )

    if tool is None:
        return None

    builder = (
        tool.get(
            "presentation_builder"
        )
    )

    if builder is None:
        return None

    if not callable(
        builder
    ):
        return None

    try:
        card = (
            builder(
                result
            )
        )

    except Exception as exc:
        print(
            "[PRESENTATION] Builder failed "
            f"tool={tool_name} "
            f"error={exc!r}"
        )

        return None

    validated = (
        validate_result_card(
            card
        )
    )

    if validated is None:
        print(
            "[PRESENTATION] Invalid result card "
            f"tool={tool_name}"
        )

        return None

    return validated
