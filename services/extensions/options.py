"""Allowlisted inference options; never provider-native tool parameters."""

from __future__ import annotations

import math

from services.extensions.admission import LIMIT_FIELDS, provider_model_identity, validate_admission_settings


# How long a request to the provider may take, and how often a request the
# provider turned away is sent again. Only external connections have them.
REQUEST_FIELDS = {
    "request_timeout_seconds": (5.0, 600.0),
    "retry_attempts": (0, 3),
    "retry_wait_seconds": (0.0, 10.0),
}
# The longest answer a user can ask for stays within the answer stages' own
# output ceiling (1024 tokens), so a setting can only shorten what they allow.
ANSWER_MAX_TOKENS = (32, 1024)
MAX_INSTRUCTION_CHARACTERS = 2000
INFERENCE_OPTION_FIELDS = frozenset({
    *LIMIT_FIELDS, *REQUEST_FIELDS, "admission_max_wait_seconds", "reasoning_effort", "thinking_mode",
})
NETWORKING_INFERENCE_MODES = ("dedicated_local", "selected_model")
# Settings that apply to every kind of connection, local models included.
ANSWER_OPTION_FIELDS = frozenset({"answer_max_tokens", "custom_instructions"})
MODEL_OPTION_FIELDS = INFERENCE_OPTION_FIELDS | ANSWER_OPTION_FIELDS | {"networking_inference"}
# An empty value removes these settings instead of saving an empty string.
CLEARED_BY_EMPTY_TEXT = frozenset({"reasoning_effort", "thinking_mode", "custom_instructions"})
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


def _number(name: str, value, minimum, maximum, *, whole: bool):
    kinds = (int,) if whole else (int, float)
    if type(value) not in kinds or (type(value) is float and not math.isfinite(value)) \
            or not minimum <= value <= maximum:
        kind = "a whole number" if whole else "a number"
        raise ValueError(f"{name} must be {kind} from {minimum:g} to {maximum:g}")
    return value


def request_timeout_seconds(profile: dict, default: float) -> float:
    """The connection's request timeout, or the transport's own default."""
    value = profile.get("request_timeout_seconds")
    return default if value is None else float(value)


def validate_answer_options(profile: dict) -> dict:
    """Answer length and the owner's instructions; valid for every connection."""
    from services.extensions.store import sensitive_content
    result = {}
    if "answer_max_tokens" in profile:
        result["answer_max_tokens"] = _number("answer_max_tokens", profile["answer_max_tokens"],
                                              *ANSWER_MAX_TOKENS, whole=True)
    if "custom_instructions" in profile:
        text = profile["custom_instructions"]
        if not isinstance(text, str):
            raise ValueError("custom_instructions must be text")
        text = text.strip()
        if len(text) > MAX_INSTRUCTION_CHARACTERS:
            raise ValueError(f"custom_instructions must be at most {MAX_INSTRUCTION_CHARACTERS} characters")
        if any(ord(character) < 32 and character not in "\n\t" for character in text):
            raise ValueError("custom_instructions cannot contain control characters")
        if sensitive_content(text):
            raise ValueError("custom_instructions cannot contain credentials or personal identifiers")
        if text:
            result["custom_instructions"] = text
    return result


def validate_inference_options(profile: dict) -> dict:
    result = validate_admission_settings(profile)
    for name, (minimum, maximum) in REQUEST_FIELDS.items():
        if name in profile:
            result[name] = _number(name, profile[name], minimum, maximum, whole=name == "retry_attempts")
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
    result.update(validate_answer_options(profile))
    mode = networking_inference_mode(profile)
    if "networking_inference" in profile:
        result["networking_inference"] = mode
    return result
