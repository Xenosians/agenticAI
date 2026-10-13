"""Bounded inference transports.

Provider responses are data, never execution authority.

This module deliberately sends ordinary chat-completion requests without
provider-native tools. Jira, Git, shell, directory and other side effects must
continue through Agentic ITSM's own specialist -> SemanticGuard -> ToolGateway
boundary.
"""

from __future__ import annotations

import json
import os
from urllib.parse import urlsplit

import httpx

from services.extensions.providers import provider_origins


# Origins that are supported by another extension backend or are useful for
# manually configured compatible endpoints, but are not exposed as a quick
# provider preset.
_STATIC_ORIGINS = {
    "https://opencode.ai",
    "https://generativelanguage.googleapis.com",
    "https://api.mistral.ai",
}

DEFAULT_ORIGINS = (
    provider_origins()
    | _STATIC_ORIGINS
)


def _configured_origins() -> set[str]:
    value = os.environ.get(
        "EXTENSION_ALLOWED_ORIGINS",
        "",
    )

    return {
        entry.strip().rstrip("/")
        for entry in value.split(",")
        if entry.strip()
    }


def validate_url(
    url: str,
) -> str:
    if not isinstance(url, str):
        raise ValueError(
            "Provider endpoint must be text"
        )

    normalized = url.strip()

    parts = urlsplit(normalized)

    if (
        parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or not parts.hostname
    ):
        raise ValueError(
            "Use an endpoint URL without credentials, "
            "query parameters or fragments"
        )

    origin = (
        f"{parts.scheme}://{parts.netloc}"
    )

    origins = (
        DEFAULT_ORIGINS
        | _configured_origins()
    )

    if origin not in origins:
        raise ValueError(
            "Endpoint origin is not in the server's "
            "EXTENSION_ALLOWED_ORIGINS"
        )

    if (
        parts.scheme != "https"
        and parts.hostname != "127.0.0.1"
    ):
        raise ValueError(
            "Remote endpoints require HTTPS"
        )

    return normalized.rstrip("/")


async def request_json(
    url: str,
    *,
    payload=None,
    api_key: str = "",
    max_bytes: int = 2_000_000,
):
    url = validate_url(url)

    headers = {
        "Accept": "application/json",
    }

    if api_key:
        headers["Authorization"] = (
            f"Bearer {api_key}"
        )

    try:
        async with httpx.AsyncClient(
            timeout=90,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            async with client.stream(
                (
                    "POST"
                    if payload is not None
                    else "GET"
                ),
                url,
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    raise ValueError(
                        "Provider returned HTTP "
                        f"{response.status_code}; "
                        "check access, quota and endpoint"
                    )

                content = bytearray()

                async for chunk in response.aiter_bytes():
                    content.extend(chunk)

                    if len(content) > max_bytes:
                        raise ValueError(
                            "Provider response exceeds "
                            "size limit"
                        )

    except httpx.HTTPError:
        raise ValueError(
            "Provider connection failed"
        ) from None

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(
            "Provider returned invalid JSON"
        ) from None


async def remote_generate(
    profile: dict,
    messages: list[dict],
    max_tokens: int,
) -> str:
    # Intentionally DO NOT add:
    #
    #   tools
    #   tool_choice
    #   provider-native code execution
    #   provider-native browser/search tools
    #
    # A remote model is an inference engine, not an authorization boundary.
    body = {
        "model": profile["model"],
        "messages": messages,
        "max_tokens": min(
            max_tokens,
            8192,
        ),
        "stream": False,
    }

    data = await request_json(
        profile["url"] + "/chat/completions",
        payload=body,
        api_key=profile.get(
            "api_key",
            "",
        ),
    )

    try:
        choice = data["choices"][0]

        content = (
            choice["message"]["content"]
        )

        if (
            choice.get("finish_reason")
            == "length"
        ):
            raise ValueError(
                "Provider output was truncated; "
                "no tool proposal accepted"
            )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                "Provider did not return text"
            )

        return content

    except (
        KeyError,
        IndexError,
        TypeError,
    ):
        raise ValueError(
            "Unsupported provider response"
        ) from None
