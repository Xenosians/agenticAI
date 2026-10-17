"""Allowlisted inference options; never provider-native tool parameters."""

from __future__ import annotations

from services.extensions.admission import LIMIT_FIELDS, provider_model_identity, validate_admission_settings


INFERENCE_OPTION_FIELDS = frozenset({
    *LIMIT_FIELDS, "admission_max_wait_seconds", "reasoning_effort", "thinking_mode",
})
NETWORKING_INFERENCE_MODES = ("dedicated_local", "selected_model")
MODEL_OPTION_FIELDS = INFERENCE_OPTION_FIELDS | {"networking_inference"}
_GROQ_REASONING_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
_GROQ_REASONING_BACKENDS = ("http", "pi")
_REASONING_EFFORTS = ("low", "medium", "high")
_THINKING_MODES = ("adaptive", "disabled")


def _is_anthropic_http(profile: dict) -> bool:
    from services.extensions.anthropic import uses_messages_api
    return uses_messages_api(profile)


def inference_option_support() -> list[dict]:
    """Detached trusted capability rows for connection forms and settings.

    A row without ``model`` applies to every model on that backend/url.
    Anthropic checks the selected model's effort support against its Models
    API record before each request.
    """
    from services.extensions.providers import get_provider_preset
    groq_url = get_provider_preset("groq")["url"]
    rows = [
        {"backend": backend, "model": model if backend == "http" else f"groq/{model}",
         "url": groq_url if backend == "http" else "", "reasoning_efforts": list(_REASONING_EFFORTS)}
        for backend in _GROQ_REASONING_BACKENDS for model in _GROQ_REASONING_MODELS
    ]
    rows.append({
        "backend": "http", "url": get_provider_preset("anthropic")["url"],
        "reasoning_efforts": list(_REASONING_EFFORTS),
        "thinking_modes": list(_THINKING_MODES),
    })
    return rows


def reasoning_effort(profile: dict) -> str | None:
    value = profile.get("reasoning_effort")
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in _REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium or high")
    if _is_anthropic_http(profile):
        # Sent as output_config.effort.
        return value
    provider, model = provider_model_identity(profile)
    # Groq documents these exact values for GPT-OSS. Pi's installed --thinking
    # contract maps the same levels. Other adapter/model mappings are not assumed.
    # https://console.groq.com/docs/api-reference
    if (provider != "groq" or model not in _GROQ_REASONING_MODELS
            or profile.get("backend", "http") not in _GROQ_REASONING_BACKENDS):
        raise ValueError("Reasoning effort is supported for Anthropic, or Groq GPT-OSS via HTTP or Pi")
    return value


def thinking_mode(profile: dict) -> str | None:
    value = profile.get("thinking_mode")
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in _THINKING_MODES:
        raise ValueError("thinking_mode must be adaptive or disabled")
    if not _is_anthropic_http(profile):
        raise ValueError("Thinking mode is supported for Anthropic connections")
    return value


def validate_inference_options(profile: dict) -> dict:
    result = validate_admission_settings(profile)
    effort = reasoning_effort(profile)
    if effort is not None:
        result["reasoning_effort"] = effort
    thinking = thinking_mode(profile)
    if thinking is not None:
        result["thinking_mode"] = thinking
    if result and profile.get("backend", "http") not in {"http", "pi", "opencode"}:
        raise ValueError("External inference options require an external model connection")
    return result


def networking_inference_mode(profile: dict) -> str:
    """A connection preference cannot alter tool policy or enable retrieval."""
    value = profile.get("networking_inference", "dedicated_local")
    if not isinstance(value, str) or value not in NETWORKING_INFERENCE_MODES:
        raise ValueError("networking_inference must be dedicated_local or selected_model")
    return value


def networking_inference_options() -> dict:
    return {
        "default": "dedicated_local",
        "choices": [
            {"id": "dedicated_local", "label": "Dedicated networking model"},
            {"id": "selected_model", "label": "Use this connection for networking"},
        ],
    }


def validate_model_options(profile: dict) -> dict:
    result = validate_inference_options(profile)
    mode = networking_inference_mode(profile)
    if "networking_inference" in profile:
        result["networking_inference"] = mode
    return result
