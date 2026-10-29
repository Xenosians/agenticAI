"""Bounded direct HTTP inference transport.

Provider responses are data, never execution authority.

This module deliberately sends ordinary chat-completion requests without
provider-native tools. Jira, Git, shell, directory and other side effects must
continue through Agentic ITSM's specialist -> SemanticGuard -> ToolGateway
boundary.
"""

from __future__ import annotations

from services.lua_adapters.http_clients import async_client as lua_async_http_client

import json
import os
from urllib.parse import urlsplit

import httpx

from services.extensions.admission import admit_external_request, record_external_usage
from services.extensions.options import reasoning_effort
from services.extensions.response_format import HttpResponseFormat, supported_http_response_formats

from services.extensions.budget import (
    ExternalInferenceBudgetError,
    default_external_request_budget,
    external_provider,
    log_external_budget_rejection,
    prepare_external_request,
)

from services.extensions.diagnostics import (
    ExternalProviderError,
    classify_external_failure,
    sanitize_diagnostic,
)

from services.extensions.providers import (
    provider_origins,
)

from services.extensions import anthropic


_STATIC_ORIGINS = {
    "https://opencode.ai",
    "https://generativelanguage.googleapis.com",
    "https://api.mistral.ai",
}


DEFAULT_ORIGINS = (
    provider_origins()
    | _STATIC_ORIGINS
)


MAX_PROVIDER_ERROR_BYTES = 64_000


def _origin_setting(name: str) -> tuple[str, bool]:
    from services.lua_adapters import profile
    if "models" in profile.load_links().declared:
        return profile.process_policy().get(name, ""), True
    return os.environ.get(name, ""), False


def _configured_origins() -> set[str]:
    value, _ = _origin_setting("EXTENSION_ALLOWED_ORIGINS")

    return {
        entry.strip().rstrip("/")
        for entry
        in value.split(",")
        if entry.strip()
    }


def insecure_http_origins() -> set[str]:
    """Exact remote origins an administrator allows over plain HTTP.

    Plain HTTP exposes the credential and every prompt in transit, so this is
    an explicit, per-origin operator decision (typically a lab gateway with no
    TLS). Each origin must also be listed in EXTENSION_ALLOWED_ORIGINS.
    """
    value, _ = _origin_setting("EXTENSION_INSECURE_HTTP_ORIGINS")

    origins = set()

    for entry in value.split(","):
        entry = entry.strip().rstrip("/")

        if not entry:
            continue

        parts = urlsplit(entry)

        if (
            parts.scheme != "http"
            or not parts.hostname
            or parts.path
            or parts.query
            or parts.fragment
            or parts.username
            or parts.password
        ):
            raise ValueError(
                "EXTENSION_INSECURE_HTTP_ORIGINS entries "
                "must be exact http://host[:port] origins"
            )

        origins.add(entry)

    return origins


def is_insecure_transport(url: str) -> bool:
    """True when a validated endpoint is remote and not protected by TLS."""
    parts = urlsplit(str(url))

    return (
        parts.scheme == "http"
        and parts.hostname != "127.0.0.1"
    )


def validate_url(
    url: str,
) -> str:
    if not isinstance(
        url,
        str,
    ):
        raise ValueError(
            "Provider endpoint must be text"
        )

    normalized = (
        url.strip()
    )

    parts = urlsplit(
        normalized
    )

    if (
        parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or not parts.hostname
    ):
        raise ValueError(
            "Use an endpoint URL without "
            "credentials, query parameters "
            "or fragments"
        )

    origin = (
        f"{parts.scheme}://"
        f"{parts.netloc}"
    )

    _, declared = _origin_setting("EXTENSION_ALLOWED_ORIGINS")
    origins = _configured_origins() if declared else DEFAULT_ORIGINS | _configured_origins()

    if origin not in origins:
        raise ValueError(
            "Endpoint origin is not in "
            "the server's "
            "EXTENSION_ALLOWED_ORIGINS"
        )

    if (
        parts.scheme
        != "https"
        and parts.hostname
        != "127.0.0.1"
        and origin
        not in insecure_http_origins()
    ):
        raise ValueError(
            "Remote endpoints "
            "require HTTPS"
        )

    return normalized.rstrip(
        "/"
    )


async def read_bounded(
    response,
    *,
    max_bytes: int,
) -> bytes:
    content = bytearray()

    async for chunk in (
        response
        .aiter_bytes()
    ):
        content.extend(
            chunk
        )

        if (
            len(content)
            > max_bytes
        ):
            raise ValueError(
                "Provider response "
                "exceeds size limit"
            )

    return bytes(
        content
    )


async def request_json(
    url: str,
    *,
    payload=None,
    api_key: str = "",
    max_bytes: int = 2_000_000,
):
    url = validate_url(
        url
    )

    headers = {
        "Accept":
            "application/json",
    }

    if api_key:
        headers[
            "Authorization"
        ] = (
            f"Bearer {api_key}"
        )

    try:
        async with (
            lua_async_http_client(
                timeout=90,
                follow_redirects=False,
                trust_env=False,
            )
        ) as client:
            async with client.stream(
                (
                    "POST"
                    if payload
                    is not None
                    else "GET"
                ),
                url,
                json=payload,
                headers=headers,
            ) as response:
                if (
                    response.status_code
                    != 200
                ):
                    content = (
                        await read_bounded(
                            response,
                            max_bytes=(
                                MAX_PROVIDER_ERROR_BYTES
                            ),
                        )
                    )

                    diagnostic = (
                        sanitize_diagnostic(
                            content.decode(
                                "utf-8",
                                errors="replace",
                            ),
                            secrets=[
                                api_key,
                            ],
                        )
                    )

                    category = (
                        classify_external_failure(
                            diagnostic,
                            status_code=(
                                response
                                .status_code
                            ),
                        )
                    )

                    print(
                        "[PROVIDER] Request rejected "
                        f"status={response.status_code} "
                        f"category={category} "
                        f"detail={diagnostic!r}"
                    )

                    raise ExternalProviderError(
                        category,
                        status_code=(
                            response
                            .status_code
                        ),
                    )

                content = (
                    await read_bounded(
                        response,
                        max_bytes=(
                            max_bytes
                        ),
                    )
                )

    except ExternalProviderError:
        raise

    except httpx.HTTPError as exc:
        diagnostic = (
            sanitize_diagnostic(
                repr(
                    exc
                ),
                secrets=[
                    api_key,
                ],
            )
        )

        print(
            "[PROVIDER] Connection failure "
            f"detail={diagnostic!r}"
        )

        raise ExternalProviderError(
            "provider_connection_failed"
        ) from None

    try:
        return json.loads(
            content
        )

    except json.JSONDecodeError:
        raise ExternalProviderError(
            "provider_malformed_response"
        ) from None


async def remote_generate(
    profile: dict,
    messages: list[dict],
    max_tokens: int,
    *,
    response_format: HttpResponseFormat | None = None,
) -> str:
    if response_format is not None and (
        not isinstance(response_format, HttpResponseFormat)
        or response_format not in supported_http_response_formats(profile)
    ):
        raise ValueError("Unsupported HTTP response format for the selected provider/model.")
    effort = reasoning_effort(profile)
    provider = external_provider(
        profile,
        backend="http",
    )

    try:
        budget = prepare_external_request(
            profile,
            messages,
            max_tokens,
            default_request_budget=(
                default_external_request_budget(
                    profile,
                    backend="http",
                )
            ),
        )

    except ExternalInferenceBudgetError as exc:
        log_external_budget_rejection(
            exc,
            backend="http",
            provider=provider,
            model=str(
                profile.get(
                    "model",
                    "",
                )
            ),
        )

        raise

    print(
        "[PROVIDER] Request budget "
        "backend=http "
        f"provider={provider} "
        f"model={profile.get('model')!r} "
        "estimated_input="
        f"{budget.estimated_input_tokens} "
        "output_cap="
        f"{budget.max_output_tokens} "
        "estimated_request="
        f"{budget.estimated_request_tokens} "
        "budget="
        f"{budget.request_token_budget} "
        "dropped_messages="
        f"{budget.dropped_messages}"
    )

    if anthropic.uses_messages_api(profile):
        return await _messages_generate(profile, budget, effort)

    # Intentionally DO NOT add:
    #
    # tools
    # tool_choice
    # functions
    # provider-native code execution
    # provider-native browser/search
    #
    # A remote model is an inference engine,
    # not an authorization boundary.
    from services.lua_adapters.plugins import openai_request_async
    body = await openai_request_async(
        profile, budget, effort=effort,
        response_format=response_format.value if response_format is not None else None,
    )

    admission = await admit_external_request(profile, budget.estimated_request_tokens)

    data = await request_json(
        (
            profile[
                "url"
            ]
            + "/chat/completions"
        ),
        payload=body,
        api_key=(
            profile.get(
                "api_key",
                "",
            )
        ),
    )

    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        record_external_usage(admission, usage.get("total_tokens"))

    try:
        choice = (
            data[
                "choices"
            ][0]
        )

        content = (
            choice[
                "message"
            ][
                "content"
            ]
        )

        if (
            choice.get(
                "finish_reason"
            )
            == "length"
        ):
            raise ExternalProviderError("provider_output_truncated")

        if (
            not isinstance(
                content,
                str,
            )
            or not content.strip()
        ):
            raise ExternalProviderError("provider_empty_response")

        from services.lua_adapters.plugins import openai_response_async
        return await openai_response_async(data, binding={"backend": "http", "model": profile["model"]})

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        raise ExternalProviderError(
            "provider_malformed_response"
        ) from None


async def _messages_generate(profile: dict, budget, effort: str | None) -> str:
    """Anthropic Messages API inference with the same budget and admission rules."""
    plan = await anthropic.request_plan(profile, effort)
    from services.lua_adapters.plugins import anthropic_request_async
    body = await anthropic_request_async(
        profile,
        budget.messages,
        budget.max_output_tokens + plan.thinking_allowance,
        effort=effort,
        thinking=plan.thinking,
        max_tokens_limit=plan.max_tokens_limit,
    )
    admission = await admit_external_request(
        profile, budget.estimated_request_tokens + plan.thinking_allowance,
    )
    data = await anthropic.post_messages(
        profile["url"], body, profile.get("api_key", ""),
    )
    total = anthropic.total_usage_tokens(data)
    if total is not None:
        record_external_usage(admission, total)
    return anthropic.response_text(data)
