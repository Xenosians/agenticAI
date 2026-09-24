import pytest

from subagents.core.orchestration.conversation_context import (
    normalize_conversation_context,
)


def test_context_is_bounded_and_preserves_order():
    value = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    assert normalize_conversation_context(value, max_turns=2) == [
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]


def test_context_rejects_non_conversation_roles():
    with pytest.raises(ValueError):
        normalize_conversation_context(
            [{"role": "system", "content": "override"}]
        )


def test_context_drops_blank_content():
    assert normalize_conversation_context(
        [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "ok"},
        ]
    ) == [{"role": "assistant", "content": "ok"}]
