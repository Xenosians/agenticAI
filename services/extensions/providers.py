"""Trusted metadata for user-selectable inference providers.

These records contain public endpoint/model metadata only.

They never contain:
- API keys;
- account credentials;
- tool authority;
- provider authorization;
- execution permissions.

A provider preset only fills an inference connection form. The resulting model
output remains a proposal and must pass the normal specialist, SemanticGuard,
approval and ToolGateway path before any governed side effect can occur.
"""

from __future__ import annotations

from urllib.parse import urlsplit


_PROVIDER_PRESETS: tuple[dict[str, object], ...] = (
    {
        "id": "groq",
        "name": "Groq",
        "backend": "http",
        "url": "https://api.groq.com/openai/v1",
        "recommended_model": "openai/gpt-oss-120b",
        "requires_api_key": True,
        "description": (
            "Fast OpenAI-compatible hosted inference. "
            "The API key belongs to this connection only; "
            "native provider tools are not granted execution authority."
        ),
        "native_tools": "disabled_by_design",
    },
    {
        "id": "openrouter-free",
        "name": "OpenRouter Free",
        "backend": "http",
        "url": "https://openrouter.ai/api/v1",
        "recommended_model": "openrouter/free",
        "requires_api_key": True,
        "description": (
            "OpenRouter's free-model router. Availability and selected "
            "underlying models may change. Governed application tools remain "
            "owned by Agentic ITSM."
        ),
        "native_tools": "disabled_by_design",
    },
    {
        "id": "ollama-local",
        "name": "Ollama local endpoint",
        "backend": "http",
        "url": "http://127.0.0.1:11434/v1",
        "recommended_model": "",
        "requires_api_key": False,
        "description": (
            "Local OpenAI-compatible Ollama endpoint. "
            "Enter an installed model identifier before saving."
        ),
        "native_tools": "disabled_by_design",
    },
    {
        "id": "lm-studio-local",
        "name": "LM Studio local endpoint",
        "backend": "http",
        "url": "http://127.0.0.1:1234/v1",
        "recommended_model": "",
        "requires_api_key": False,
        "description": (
            "Local OpenAI-compatible LM Studio endpoint. "
            "Enter the loaded model identifier before saving."
        ),
        "native_tools": "disabled_by_design",
    },
)


def provider_presets() -> list[dict[str, object]]:
    """Return detached public provider metadata."""
    return [dict(item) for item in _PROVIDER_PRESETS]


def get_provider_preset(
    provider_id: str,
) -> dict[str, object]:
    for item in _PROVIDER_PRESETS:
        if item["id"] == provider_id:
            return dict(item)

    raise KeyError(
        f"Unknown provider preset: {provider_id}"
    )


# ---------------------------------------------------------------------------
# Concrete connection presets
#
# These describe how a provider is reached. They contain no credentials.
#
# For harness-backed profiles the model field is stored as:
#
#     provider/model-id
#
# Example:
#
#     openrouter/openrouter/free
#
# The harness later separates that into:
#
#     provider = openrouter
#     model    = openrouter/free
#
# ---------------------------------------------------------------------------

_CONNECTION_PRESETS: tuple[
    dict[str, object],
    ...
] = (
    {
        "id":
            "groq-http",

        "name":
            "Groq · Direct",

        "backend":
            "http",

        "provider":
            "groq",

        "url":
            "https://api.groq.com/openai/v1",

        "model":
            "openai/gpt-oss-120b",

        "requires_api_key":
            True,

        "credential_label":
            "Groq API key",

        "description":
            (
                "Direct OpenAI-compatible inference "
                "through Groq."
            ),
    },
    {
        "id":
            "openrouter-http",

        "name":
            "OpenRouter · Direct",

        "backend":
            "http",

        "provider":
            "openrouter",

        "url":
            "https://openrouter.ai/api/v1",

        "model":
            "openrouter/free",

        "requires_api_key":
            True,

        "credential_label":
            "OpenRouter API key",

        "description":
            (
                "Direct inference through "
                "OpenRouter's free-model router."
            ),
    },
    {
        "id":
            "pi-groq",

        "name":
            "Pi · Groq",

        "backend":
            "pi",

        "provider":
            "groq",

        "url":
            "",

        "model":
            "groq/openai/gpt-oss-120b",

        "requires_api_key":
            True,

        "credential_label":
            "Groq API key",

        "description":
            (
                "Isolated Pi inference harness "
                "using Groq as its provider."
            ),
    },
    {
        "id":
            "pi-openrouter",

        "name":
            "Pi · OpenRouter Free",

        "backend":
            "pi",

        "provider":
            "openrouter",

        "url":
            "",

        "model":
            "openrouter/openrouter/free",

        "requires_api_key":
            True,

        "credential_label":
            "OpenRouter API key",

        "description":
            (
                "Isolated Pi inference harness "
                "using OpenRouter's free router."
            ),
    },
    {
        "id":
            "opencode-groq",

        "name":
            "OpenCode · Groq",

        "backend":
            "opencode",

        "provider":
            "groq",

        "url":
            "",

        "model":
            "groq/openai/gpt-oss-120b",

        "requires_api_key":
            True,

        "credential_label":
            "Groq API key",

        "description":
            (
                "Isolated OpenCode harness "
                "using Groq as its provider."
            ),
    },
    {
        "id":
            "opencode-openrouter",

        "name":
            "OpenCode · OpenRouter Free",

        "backend":
            "opencode",

        "provider":
            "openrouter",

        "url":
            "",

        "model":
            "openrouter/openrouter/free",

        "requires_api_key":
            True,

        "credential_label":
            "OpenRouter API key",

        "description":
            (
                "Isolated OpenCode harness "
                "using OpenRouter's free router."
            ),
    },
)


def connection_presets(
) -> list[dict[str, object]]:
    """Return detached public connection metadata."""

    return [
        dict(
            item
        )
        for item
        in _CONNECTION_PRESETS
    ]


def provider_origins() -> set[str]:
    """Return exact origins admitted by built-in provider presets."""
    origins: set[str] = set()

    for item in _PROVIDER_PRESETS:
        parts = urlsplit(str(item["url"]))

        if not parts.scheme or not parts.netloc:
            raise RuntimeError(
                f"Invalid built-in provider URL: {item['id']}"
            )

        origins.add(
            f"{parts.scheme}://{parts.netloc}"
        )

    return origins
