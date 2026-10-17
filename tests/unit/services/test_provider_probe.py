import asyncio

import pytest

import services.extensions.probe as probe


def profile():
    return {
        "backend":
            "http",

        "model":
            "openrouter/free",

        "url":
            "https://openrouter.ai/api/v1",

        "api_key":
            "unit-test-secret",
    }


def test_probe_accepts_complete_visible_response(
    monkeypatch,
):
    async def fake_request_json(
        *args,
        **kwargs,
    ):
        return {
            "model":
                "dots-studio/dots-3-note-preview:free",

            "provider":
                "AtlasCloud",

            "choices": [
                {
                    "finish_reason":
                        "stop",

                    "message": {
                        "role":
                            "assistant",

                        "content":
                            "\n\nPROVIDER_OK",

                        "reasoning":
                            "internal reasoning",
                    },
                }
            ],
        }

    monkeypatch.setattr(
        probe,
        "request_json",
        fake_request_json,
    )

    result = asyncio.run(
        probe.remote_probe(
            profile()
        )
    )

    assert (
        result[
            "state"
        ]
        == "reachable"
    )

    assert (
        result[
            "probe_complete"
        ]
        is True
    )

    assert (
        result[
            "response"
        ]
        == "PROVIDER_OK"
    )

    assert (
        result[
            "provider"
        ]
        == "AtlasCloud"
    )


def test_probe_treats_reasoning_length_as_reachable(
    monkeypatch,
):
    async def fake_request_json(
        *args,
        **kwargs,
    ):
        return {
            "model":
                "cohere/north-mini-code:free",

            "provider":
                "Cohere",

            "choices": [
                {
                    "finish_reason":
                        "length",

                    "message": {
                        "role":
                            "assistant",

                        "content":
                            None,

                        "reasoning":
                            (
                                "The user requested "
                                "PROVIDER_OK"
                            ),
                    },
                }
            ],
        }

    monkeypatch.setattr(
        probe,
        "request_json",
        fake_request_json,
    )

    result = asyncio.run(
        probe.remote_probe(
            profile()
        )
    )

    assert (
        result[
            "state"
        ]
        == "reachable"
    )

    assert (
        result[
            "probe_complete"
        ]
        is False
    )

    assert (
        result[
            "finish_reason"
        ]
        == "length"
    )

    assert (
        "response"
        not in result
    )


def test_probe_maps_authentication_failure(
    monkeypatch,
):
    async def fake_request_json(
        *args,
        **kwargs,
    ):
        raise ValueError(
            (
                "Provider returned HTTP 401; "
                "check access, quota and endpoint"
            )
        )

    monkeypatch.setattr(
        probe,
        "request_json",
        fake_request_json,
    )

    with pytest.raises(
        probe.ProviderProbeError
    ) as exc:
        asyncio.run(
            probe.remote_probe(
                profile()
            )
        )

    assert (
        exc.value.code
        == "provider_authentication_failed"
    )

    assert (
        exc.value.provider_status
        == 401
    )

    assert (
        "unit-test-secret"
        not in str(
            exc.value.public_detail()
        )
    )


def test_probe_maps_rate_limit_failure(
    monkeypatch,
):
    async def fake_request_json(
        *args,
        **kwargs,
    ):
        raise ValueError(
            (
                "Provider returned HTTP 429; "
                "check access, quota and endpoint"
            )
        )

    monkeypatch.setattr(
        probe,
        "request_json",
        fake_request_json,
    )

    with pytest.raises(
        probe.ProviderProbeError
    ) as exc:
        asyncio.run(
            probe.remote_probe(
                profile()
            )
        )

    assert (
        exc.value.code
        == "provider_rate_limited"
    )

    assert (
        exc.value.provider_status
        == 429
    )


def test_probe_rejects_malformed_success_response(
    monkeypatch,
):
    async def fake_request_json(
        *args,
        **kwargs,
    ):
        return {
            "hello":
                "world"
        }

    monkeypatch.setattr(
        probe,
        "request_json",
        fake_request_json,
    )

    with pytest.raises(
        probe.ProviderProbeError
    ) as exc:
        asyncio.run(
            probe.remote_probe(
                profile()
            )
        )

    assert (
        exc.value.code
        == "provider_invalid_response"
    )
