import asyncio
from types import SimpleNamespace

import pytest

from services.extensions.providers import (
    connection_presets,
)

from services.extensions.service import (
    _model_connection_requires_credential,
    command,
    extension_store,
)


def test_connection_presets_contain_no_credentials():
    presets = (
        connection_presets()
    )

    expected = {
        "anthropic-http",
        "pi-anthropic",
        "opencode-anthropic",
        "groq-http",
        "openrouter-http",
        "pi-groq",
        "pi-openrouter",
        "opencode-groq",
        "opencode-openrouter",
    }

    assert {
        row["id"]
        for row in presets
    } == expected

    for row in presets:
        assert (
            row[
                "requires_api_key"
            ]
            is True
        )

        assert set(
            row
        ).isdisjoint(
            {
                "api_key",
                "credential",
                "password",
                "secret",
                "authorization",
            }
        )


def test_expected_provider_model_shapes():
    presets = {
        row["id"]:
            row
        for row
        in connection_presets()
    }

    assert (
        presets[
            "groq-http"
        ][
            "model"
        ]
        == "openai/gpt-oss-120b"
    )

    assert (
        presets[
            "openrouter-http"
        ][
            "model"
        ]
        == "openrouter/free"
    )

    assert (
        presets[
            "pi-openrouter"
        ][
            "model"
        ]
        == "openrouter/openrouter/free"
    )

    assert (
        presets[
            "opencode-openrouter"
        ][
            "model"
        ]
        == "openrouter/openrouter/free"
    )


def test_local_http_can_remain_keyless():
    assert (
        _model_connection_requires_credential(
            "http",
            "qwen2.5",
            "http://127.0.0.1:11434/v1",
        )
        is False
    )


def test_remote_http_requires_explicit_key():
    assert (
        _model_connection_requires_credential(
            "http",
            "openrouter/free",
            "https://openrouter.ai/api/v1",
        )
        is True
    )


def test_hosted_harness_requires_explicit_key():
    assert (
        _model_connection_requires_credential(
            "pi",
            "groq/openai/gpt-oss-120b",
        )
        is True
    )

    assert (
        _model_connection_requires_credential(
            "opencode",
            "openrouter/openrouter/free",
        )
        is True
    )


def test_remote_model_is_rejected_before_storage_without_key(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "EXTENSION_STORAGE_ROOT",
        str(
            tmp_path
            / "extensions"
        ),
    )

    extension_store.cache_clear()

    settings = SimpleNamespace(
        model_profiles={}
    )

    payload = {
        "name":
            "OpenRouter Direct",

        "backend":
            "http",

        "model":
            "openrouter/free",

        "url":
            "https://openrouter.ai/api/v1",

        "api_key":
            "",
    }

    try:
        with pytest.raises(
            ValueError,
            match="requires an API key",
        ):
            asyncio.run(
                command(
                    "alice",
                    "model",
                    payload,
                    settings,
                )
            )

        assert (
            extension_store()
            .list(
                "alice"
            )
            == []
        )

    finally:
        extension_store.cache_clear()


def test_remote_model_credential_is_encrypted_and_not_listed(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "EXTENSION_STORAGE_ROOT",
        str(
            tmp_path
            / "extensions"
        ),
    )

    extension_store.cache_clear()

    settings = SimpleNamespace(
        model_profiles={}
    )

    key = (
        "test-openrouter-key-123456"
    )

    payload = {
        "name":
            "OpenRouter Direct",

        "backend":
            "http",

        "model":
            "openrouter/free",

        "url":
            "https://openrouter.ai/api/v1",

        "api_key":
            key,
    }

    try:
        result = asyncio.run(
            command(
                "alice",
                "model",
                payload,
                settings,
            )
        )

        assert (
            result[
                "credential_configured"
            ]
            is True
        )

        listed = (
            extension_store()
            .list(
                "alice"
            )
        )

        assert (
            "api_key"
            not in listed[0]
        )

        assert (
            key.encode()
            not in extension_store()
            .path
            .read_bytes()
        )

    finally:
        extension_store.cache_clear()
