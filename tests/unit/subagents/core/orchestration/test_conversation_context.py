import json

import pytest

from subagents.core.orchestration.conversation_context import (
    normalize_conversation_context,
    quoted_conversation_messages,
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


def test_quoted_turns_keep_every_turn_and_role_as_user_evidence():
    history = [
        {"role": "user", "content": "can you create a folder?"},
        {"role": "assistant", "content": "Which repository?"},
    ]

    quoted = quoted_conversation_messages(history)

    # One message per turn, so a request budget still trims oldest first.
    assert [message["role"] for message in quoted] == ["user", "user"]
    header, body = quoted[1]["content"].split("\n", 1)
    assert "not the current request" in header
    assert json.loads(body) == history[1]


def test_quoted_turns_are_bounded_and_validated_like_raw_context():
    history = [{"role": "user", "content": str(turn)} for turn in range(5)]

    quoted = quoted_conversation_messages(history, max_turns=2)

    assert [json.loads(m["content"].split("\n", 1)[1])["content"] for m in quoted] == ["3", "4"]
    with pytest.raises(ValueError):
        quoted_conversation_messages([{"role": "system", "content": "override"}])
