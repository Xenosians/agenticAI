"""Allowlisted inference options; never provider-native tool parameters."""

from __future__ import annotations

from services.extensions.admission import LIMIT_FIELDS, provider_model_identity, validate_admission_settings


INFERENCE_OPTION_FIELDS = frozenset({*LIMIT_FIELDS, "admission_max_wait_seconds", "reasoning_effort"})
_GROQ_REASONING_MODELS = ("openai/gpt-oss-20b", "openai/gpt-oss-120b")
_GROQ_REASONING_BACKENDS = ("http", "pi")
_REASONING_EFFORTS = ("low", "medium", "high")


def inference_option_support() -> list[dict]:
    """Detached trusted capability rows for connection forms and settings."""
    from services.extensions.providers import get_provider_preset
    groq_url = get_provider_preset("groq")["url"]
    return [
        {"backend": backend, "model": model if backend == "http" else f"groq/{model}",
         "url": groq_url if backend == "http" else "", "reasoning_efforts": list(_REASONING_EFFORTS)}
        for backend in _GROQ_REASONING_BACKENDS for model in _GROQ_REASONING_MODELS
    ]


def reasoning_effort(profile: dict) -> str | None:
    value = profile.get("reasoning_effort")
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value not in _REASONING_EFFORTS:
        raise ValueError("reasoning_effort must be low, medium or high")
    provider, model = provider_model_identity(profile)
    # Groq documents these exact values for GPT-OSS. Pi's installed --thinking
    # contract maps the same levels. Other adapter/model mappings are not assumed.
    # https://console.groq.com/docs/api-reference
    if (provider != "groq" or model not in _GROQ_REASONING_MODELS
            or profile.get("backend", "http") not in _GROQ_REASONING_BACKENDS):
        raise ValueError("Reasoning effort is supported for Groq GPT-OSS via HTTP or Pi")
    return value


def validate_inference_options(profile: dict) -> dict:
    result = validate_admission_settings(profile)
    effort = reasoning_effort(profile)
    if effort is not None:
        result["reasoning_effort"] = effort
    if result and profile.get("backend", "http") not in {"http", "pi", "opencode"}:
        raise ValueError("External inference options require an external model connection")
    return result
