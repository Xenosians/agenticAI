import pytest

from services.extensions.budget import (
    ExternalInferenceBudgetError,
    default_external_request_budget,
    prepare_external_request,
)


def test_groq_http_default_budget():
    assert (
        default_external_request_budget(
            {
                "backend":
                    "http",

                "url":
                    "https://api.groq.com/openai/v1",

                "model":
                    "openai/gpt-oss-120b",
            },
            backend="http",
        )
        == 7000
    )


def test_groq_harness_default_leaves_tpm_headroom():
    assert (
        default_external_request_budget(
            {
                "backend":
                    "pi",

                "model":
                    "groq/openai/gpt-oss-120b",
            },
            backend="pi",
        )
        == 6000
    )


def test_openrouter_http_default_budget():
    assert (
        default_external_request_budget(
            {
                "backend":
                    "http",

                "url":
                    "https://openrouter.ai/api/v1",

                "model":
                    "openrouter/free",
            },
            backend="http",
        )
        == 9000
    )


def test_openrouter_harness_default_budget():
    assert (
        default_external_request_budget(
            {
                "backend":
                    "opencode",

                "model":
                    "openrouter/openrouter/free",
            },
            backend="opencode",
        )
        == 8000
    )


def test_provider_specific_env_override(
    monkeypatch,
):
    monkeypatch.setenv(
        "EXTERNAL_GROQ_HARNESS_REQUEST_TOKEN_BUDGET",
        "5800",
    )

    assert (
        default_external_request_budget(
            {
                "backend":
                    "pi",

                "model":
                    "groq/openai/gpt-oss-120b",
            },
            backend="pi",
        )
        == 5800
    )


def test_budget_rejection_exposes_counts_not_content():
    profile = {
        "request_token_budget":
            512,

        "max_output_tokens":
            64,
    }

    secret_marker = (
        "PRIVATE-MANDATORY-CONTENT-DO-NOT-LOG"
    )

    messages = [
        {
            "role":
                "system",

            "content":
                secret_marker
                + (
                    " x"
                    * 2500
                ),
        },
        {
            "role":
                "user",

            "content":
                "hello",
        },
    ]

    with pytest.raises(
        ExternalInferenceBudgetError
    ) as caught:
        prepare_external_request(
            profile,
            messages,
            64,
            default_request_budget=7000,
        )

    exc = caught.value

    assert (
        exc.request_token_budget
        == 512
    )

    assert (
        exc.output_tokens
        == 64
    )

    assert (
        exc.input_budget
        == 448
    )

    assert (
        exc.mandatory_tokens
        is not None
    )

    assert (
        exc.mandatory_tokens
        > exc.input_budget
    )

    assert (
        exc.system_tokens
        is not None
    )

    assert (
        exc.newest_tokens
        is not None
    )

    assert (
        exc.message_count
        == 2
    )

    assert (
        secret_marker
        not in str(
            exc
        )
    )
