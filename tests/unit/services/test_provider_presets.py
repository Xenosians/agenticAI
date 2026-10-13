import asyncio

import pytest

from services.extensions.providers import (
    get_provider_preset,
    provider_origins,
    provider_presets,
)
from services.extensions.transport import (
    remote_generate,
    validate_url,
)


def test_provider_presets_are_unique_and_public():
    presets = provider_presets()

    ids = [
        item["id"]
        for item in presets
    ]

    assert len(ids) == len(set(ids))

    for item in presets:
        assert set(item).isdisjoint(
            {
                "api_key",
                "authorization",
                "credential",
                "password",
            }
        )

        assert (
            item["native_tools"]
            == "disabled_by_design"
        )

        validate_url(
            str(item["url"])
        )


def test_provider_origins_match_public_presets():
    expected = set()

    for item in provider_presets():
        url = str(item["url"])

        if url.startswith(
            "https://api.groq.com/"
        ):
            expected.add(
                "https://api.groq.com"
            )

        elif url.startswith(
            "https://openrouter.ai/"
        ):
            expected.add(
                "https://openrouter.ai"
            )

        elif url.startswith(
            "http://127.0.0.1:11434/"
        ):
            expected.add(
                "http://127.0.0.1:11434"
            )

        elif url.startswith(
            "http://127.0.0.1:1234/"
        ):
            expected.add(
                "http://127.0.0.1:1234"
            )

    assert provider_origins() == expected


def test_groq_preset_uses_current_openai_compatible_shape():
    groq = get_provider_preset(
        "groq"
    )

    assert (
        groq["backend"]
        == "http"
    )

    assert (
        groq["url"]
        == "https://api.groq.com/openai/v1"
    )

    assert (
        groq["recommended_model"]
        == "openai/gpt-oss-120b"
    )

    assert (
        groq["requires_api_key"]
        is True
    )


def test_unknown_preset_fails_closed():
    with pytest.raises(KeyError):
        get_provider_preset(
            "made-up-provider"
        )


def test_remote_generate_is_inference_only(
    monkeypatch,
):
    captured = {}

    async def fake_request_json(
        url,
        *,
        payload=None,
        api_key="",
        max_bytes=2_000_000,
    ):
        captured["url"] = url
        captured["payload"] = payload
        captured["api_key"] = api_key
        captured["max_bytes"] = max_bytes

        return {
            "choices": [
                {
                    "message": {
                        "content": "safe response"
                    },
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(
        "services.extensions.transport.request_json",
        fake_request_json,
    )

    response = asyncio.run(
        remote_generate(
            {
                "backend": "http",
                "model": "openai/gpt-oss-120b",
                "url": (
                    "https://api.groq.com/openai/v1"
                ),
                "api_key": "private-test-key",
            },
            [
                {
                    "role": "user",
                    "content": "read-only test",
                }
            ],
            256,
        )
    )

    assert response == "safe response"

    assert (
        captured["url"]
        == (
            "https://api.groq.com/"
            "openai/v1/chat/completions"
        )
    )

    assert (
        captured["api_key"]
        == "private-test-key"
    )

    assert captured["payload"] == {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {
                "role": "user",
                "content": "read-only test",
            }
        ],
        "max_tokens": 256,
        "stream": False,
    }

    # Critical authority boundary:
    assert (
        "tools"
        not in captured["payload"]
    )

    assert (
        "tool_choice"
        not in captured["payload"]
    )

    assert (
        "functions"
        not in captured["payload"]
    )

    assert (
        "private-test-key"
        not in str(captured["payload"])
    )
