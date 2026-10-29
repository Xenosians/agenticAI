"""Anthropic Messages API inference transport.

Provider responses are data, never execution authority. Requests carry only
the system prompt and conversation text: no ``tools``, ``tool_choice``, MCP
servers, server tools or provider-native code execution. Jira, Git, shell,
directory and every other side effect stays behind the application's
specialist -> SemanticGuard -> ToolGateway boundary.

Contract references (Claude Platform docs):
- Messages: https://platform.claude.com/docs/en/api/messages
- Stop reasons: https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
- Thinking: https://platform.claude.com/docs/en/build-with-claude/thinking
- Models: https://platform.claude.com/docs/en/api/models/list
"""

from __future__ import annotations

from services.lua_adapters.http_clients import async_client as lua_async_http_client

from dataclasses import dataclass
import json
import os
import threading
import time
from urllib.parse import quote

import httpx

from services.extensions.diagnostics import (
    ExternalProviderError,
    classify_external_failure,
    sanitize_diagnostic,
)

API_VERSION = "2023-06-01"
API_FORMAT = "anthropic"
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_BYTES = 2_000_000
MAX_ERROR_BYTES = 64_000

# Stop reasons that represent a complete, usable answer.
_COMPLETE_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})

# Every other documented stop reason is mapped to a fail-closed category.
# An unknown future stop reason is also rejected rather than trusted.
_STOP_REASON_FAILURES = {
    "max_tokens": "provider_output_truncated",
    "model_context_window_exceeded": "provider_output_truncated",
    "refusal": "provider_refused",
    "pause_turn": "provider_incomplete_turn",
    "tool_use": "provider_unexpected_tool_use",
}

# System prompts at least this long are marked cacheable. Shorter prompts are
# below the provider's minimum cacheable size and are sent as plain text.
CACHE_MIN_SYSTEM_CHARS = 4096

THINKING_MODES = ("adaptive", "disabled")
EFFORT_LEVELS = ("low", "medium", "high")
DEFAULT_THINKING_ALLOWANCE = 4096
_CAPABILITY_TTL_SECONDS = 3600.0
_CAPABILITY_FAILURE_TTL_SECONDS = 60.0
_capability_cache: dict[tuple[str, str], tuple[float, dict | None]] = {}
_capability_lock = threading.Lock()


@dataclass(frozen=True)
class RequestPlan:
    """Per-request thinking configuration derived from model capabilities.

    ``thinking`` is the value sent as ``thinking.type`` (None keeps the model
    default). Thinking tokens count toward ``max_tokens``, so a request that
    may think reserves ``thinking_allowance`` extra output tokens. The visible
    answer keeps the stage's normal output cap and is still rejected when cut.
    """

    thinking: str | None
    thinking_allowance: int
    max_tokens_limit: int | None = None


def is_anthropic_url(url: str) -> bool:
    """True for the official Messages API origin."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(str(url))
    except ValueError:
        return False
    return parts.scheme == "https" and parts.hostname == "api.anthropic.com"


def uses_messages_api(profile: dict) -> bool:
    """Select the Messages protocol from trusted connection metadata only."""
    if profile.get("backend", "http") != "http":
        return False
    declared = profile.get("api_format")
    if declared is not None:
        if declared not in {API_FORMAT, "openai"}:
            raise ValueError("api_format must be anthropic or openai")
        return declared == API_FORMAT
    return is_anthropic_url(str(profile.get("url", "")))


def endpoint(url: str, path: str) -> str:
    """Join an API root with ``/v1/<path>``; accepts roots with or without /v1."""
    from services.extensions.transport import validate_url

    root = validate_url(url).rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return f"{root}/v1/{path.lstrip('/')}"


def _timeout() -> float:
    value = os.environ.get("ANTHROPIC_TIMEOUT_SECONDS")
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        resolved = float(value)
    except ValueError:
        raise ValueError("ANTHROPIC_TIMEOUT_SECONDS must be a number") from None
    if not 5.0 <= resolved <= 600.0:
        raise ValueError("ANTHROPIC_TIMEOUT_SECONDS must be between 5 and 600")
    return resolved


def thinking_mode(profile: dict) -> str | None:
    """Return the configured thinking mode, or None for automatic selection."""
    value = profile.get("thinking_mode")
    if value is None or value == "":
        return None
    if value not in THINKING_MODES:
        raise ValueError("thinking_mode must be adaptive or disabled")
    return value


def thinking_allowance() -> int:
    value = os.environ.get("ANTHROPIC_THINKING_TOKEN_ALLOWANCE")
    if value is None:
        return DEFAULT_THINKING_ALLOWANCE
    try:
        resolved = int(value)
    except ValueError:
        raise ValueError("ANTHROPIC_THINKING_TOKEN_ALLOWANCE must be an integer") from None
    if not 0 <= resolved <= 64_000:
        raise ValueError("ANTHROPIC_THINKING_TOKEN_ALLOWANCE must be between 0 and 64000")
    return resolved


def _supported(node, *path: str) -> bool | None:
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if isinstance(node, dict):
        value = node.get("supported")
        return value if isinstance(value, bool) else None
    return None


async def model_capabilities(profile: dict) -> dict | None:
    """Models API record for the configured model, cached; None when unknown.

    A lookup failure never blocks inference: the request then keeps the
    model's default thinking behaviour and reserves the thinking allowance.
    """
    url = str(profile.get("url", ""))
    model = str(profile.get("model", ""))
    key = (url.rstrip("/"), model)
    with _capability_lock:
        cached = _capability_cache.get(key)
    if cached is not None:
        age = time.monotonic() - cached[0]
        ttl = _CAPABILITY_TTL_SECONDS if cached[1] is not None else _CAPABILITY_FAILURE_TTL_SECONDS
        if age < ttl:
            return cached[1]
    try:
        record = normalize_model_record(
            await get_model(url, model, str(profile.get("api_key", "")))
        )
    except ExternalProviderError as exc:
        if exc.category in {"provider_authentication_failed", "provider_model_unavailable"}:
            raise
        record = None
    with _capability_lock:
        _capability_cache[key] = (time.monotonic(), record)
    return record


def normalize_model_record(record: dict | None) -> dict | None:
    """Return a Models API record in Anthropic's shape.

    Anthropic-format gateways (for example 9router) describe the same model
    facts with different field names: ``max_completion_tokens`` and flat
    capability flags such as ``thinkingCanDisable``. Facts a record does not
    state stay unknown, so request planning keeps its conservative defaults.
    """
    if not isinstance(record, dict) or isinstance(record.get("max_tokens"), int):
        return record
    normalized = dict(record)
    capabilities = record.get("capabilities")
    flags = capabilities if isinstance(capabilities, dict) else {}
    for limit in (record.get("max_completion_tokens"), flags.get("maxOutput")):
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            normalized["max_tokens"] = limit
            break
    if not flags or "thinking" in flags:
        return normalized
    mapped: dict = {}
    reasoning = flags.get("reasoning")
    if isinstance(reasoning, bool):
        types: dict = {}
        can_disable = flags.get("thinkingCanDisable")
        if isinstance(can_disable, bool):
            types["disabled"] = {"supported": can_disable}
        if reasoning and str(flags.get("thinkingFormat", "")).endswith("adaptive"):
            types["adaptive"] = {"supported": True}
        mapped["thinking"] = {"supported": reasoning, "types": types}
    effort = flags.get("thinkingEffortSupported")
    if isinstance(effort, bool):
        mapped["effort"] = {"supported": effort}
    normalized["capabilities"] = {**flags, **mapped}
    return normalized


def remember_capabilities(profile: dict, record: dict) -> None:
    """Cache a Models API record that was fetched for another purpose."""
    key = (str(profile.get("url", "")).rstrip("/"), str(profile.get("model", "")))
    with _capability_lock:
        _capability_cache[key] = (time.monotonic(), normalize_model_record(record))


async def request_plan(profile: dict, effort: str | None) -> RequestPlan:
    configured = thinking_mode(profile)
    allowance = thinking_allowance()
    record = await model_capabilities(profile)
    limit = record.get("max_tokens") if isinstance(record, dict) else None
    limit = limit if isinstance(limit, int) and limit > 0 else None
    if configured == "disabled":
        return RequestPlan("disabled", 0, limit)
    if configured == "adaptive":
        return RequestPlan("adaptive", allowance, limit)
    capabilities = record.get("capabilities") if isinstance(record, dict) else None
    if not isinstance(capabilities, dict):
        return RequestPlan(None, allowance, limit)
    if effort is not None and (
        _supported(capabilities, "effort") is False
        or _supported(capabilities, "effort", effort) is False
    ):
        raise ValueError("The selected Anthropic model does not support this effort level")
    if _supported(capabilities, "thinking") is False:
        return RequestPlan(None, 0, limit)
    if _supported(capabilities, "thinking", "types", "disabled"):
        # Fastest deterministic path for structured proposals; no truncation
        # risk from hidden reasoning consuming the visible output budget.
        return RequestPlan("disabled", 0, limit)
    return RequestPlan(None, allowance, limit)


def headers(api_key: str) -> dict[str, str]:
    # Accept is required: some Anthropic-format gateways answer clients that
    # accept anything with an event-stream body even when stream is false.
    result = {
        "Accept": "application/json",
        "anthropic-version": API_VERSION,
    }
    if api_key:
        result["x-api-key"] = api_key
    return result


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Map role messages to the Messages API shape.

    System content moves to the top-level ``system`` parameter. Adjacent turns
    with the same role are merged, so evidence or workflow notes inserted as
    separate messages remain in order without violating role alternation.
    The newest turn must be a user turn: assistant prefill is never used.
    """
    system_parts: list[str] = []
    turns: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Messages API content must be text")
        if role == "system":
            if content.strip():
                system_parts.append(content)
            continue
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported message role for the Messages API")
        if not content.strip():
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "Continue with the task described in the system instructions."})
    if turns[-1]["role"] != "user":
        raise ValueError("The newest Messages API turn must come from the user")
    return "\n\n".join(system_parts), turns


def build_request(
    profile: dict,
    messages: list[dict],
    max_tokens: int,
    *,
    effort: str | None = None,
    thinking: str | None = None,
    max_tokens_limit: int | None = None,
) -> dict:
    system, turns = split_system(messages)
    tokens = int(max_tokens)
    if max_tokens_limit is not None:
        tokens = min(tokens, max_tokens_limit)
    body: dict = {
        "model": profile["model"],
        "max_tokens": tokens,
        "messages": turns,
        "stream": False,
    }
    if system:
        if len(system) >= CACHE_MIN_SYSTEM_CHARS:
            # The routing catalog and specialist protocols repeat across
            # requests; caching them cuts cost and latency. Content unchanged.
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        else:
            body["system"] = system
    if thinking is not None:
        if thinking not in THINKING_MODES:
            raise ValueError("Unsupported thinking mode")
        body["thinking"] = {"type": thinking}
    if effort is not None:
        if effort not in EFFORT_LEVELS:
            raise ValueError("Unsupported Anthropic effort level")
        body["output_config"] = {"effort": effort}
    # Intentionally DO NOT add tools, tool_choice, mcp_servers or any server
    # tool. A remote model is an inference engine, not an authorization boundary.
    return body


async def post_messages(url: str, payload: dict, api_key: str) -> dict:
    """POST /v1/messages with bounded, secret-safe failure handling."""
    from services.extensions.transport import read_bounded

    target = endpoint(url, "messages")
    try:
        async with lua_async_http_client(
            timeout=_timeout(), follow_redirects=False, trust_env=False,
        ) as client:
            async with client.stream(
                "POST", target, json=payload, headers=headers(api_key),
            ) as response:
                if response.status_code != 200:
                    content = await read_bounded(response, max_bytes=MAX_ERROR_BYTES)
                    diagnostic = sanitize_diagnostic(
                        content.decode("utf-8", errors="replace"), secrets=[api_key],
                    )
                    category = classify_messages_failure(diagnostic, response.status_code)
                    print(
                        "[PROVIDER] Request rejected provider=anthropic "
                        f"status={response.status_code} category={category} "
                        f"detail={diagnostic!r}"
                    )
                    raise ExternalProviderError(category, status_code=response.status_code)
                content = await read_bounded(response, max_bytes=MAX_RESPONSE_BYTES)
    except ExternalProviderError:
        raise
    except httpx.TimeoutException:
        raise ExternalProviderError("provider_timeout") from None
    except httpx.HTTPError as exc:
        diagnostic = sanitize_diagnostic(repr(exc), secrets=[api_key])
        print(f"[PROVIDER] Connection failure provider=anthropic detail={diagnostic!r}")
        raise ExternalProviderError("provider_connection_failed") from None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ExternalProviderError("provider_malformed_response") from None
    if not isinstance(data, dict):
        raise ExternalProviderError("provider_malformed_response")
    return data


def classify_messages_failure(diagnostic: str, status_code: int) -> str:
    """Map documented Messages API errors to stable public categories."""
    lowered = diagnostic.lower()
    if status_code == 401 or "authentication_error" in lowered:
        return "provider_authentication_failed"
    if status_code == 402 or "billing_error" in lowered or "credit balance" in lowered:
        return "provider_billing_required"
    if status_code == 403 or "permission_error" in lowered:
        return "provider_permission_denied"
    if status_code == 404 or "not_found_error" in lowered:
        return "provider_model_unavailable"
    if status_code == 413 or "request_too_large" in lowered:
        return "provider_request_too_large"
    if status_code == 429 or "rate_limit_error" in lowered:
        return "provider_rate_limited"
    if status_code == 529 or "overloaded_error" in lowered:
        return "provider_overloaded"
    if status_code == 504 or "timeout_error" in lowered:
        return "provider_timeout"
    if 500 <= status_code <= 599 or "api_error" in lowered:
        return "provider_unavailable"
    if "spend limit" in lowered or "usage limit" in lowered:
        return "provider_billing_required"
    return classify_external_failure(diagnostic, status_code=status_code)


def response_text(data: dict) -> str:
    """Return visible text from a complete response; anything else fails closed."""
    stop_reason = data.get("stop_reason")
    if stop_reason not in _COMPLETE_STOP_REASONS:
        raise ExternalProviderError(
            _STOP_REASON_FAILURES.get(stop_reason, "provider_incomplete_turn")
        )
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ExternalProviderError("provider_malformed_response")
    text_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            raise ExternalProviderError("provider_malformed_response")
        kind = block.get("type")
        if kind == "text":
            value = block.get("text")
            if not isinstance(value, str):
                raise ExternalProviderError("provider_malformed_response")
            text_parts.append(value)
        elif kind in {"thinking", "redacted_thinking"}:
            # Hidden reasoning is never surfaced or treated as the answer.
            continue
        else:
            # Tool use or server-tool blocks were never requested.
            raise ExternalProviderError("provider_unexpected_tool_use")
    text = "".join(text_parts)
    if not text.strip():
        raise ExternalProviderError("provider_empty_response")
    return text


def total_usage_tokens(data: dict) -> int | None:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    total = 0
    seen = False
    for key in (
        "input_tokens", "output_tokens",
        "cache_creation_input_tokens", "cache_read_input_tokens",
    ):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            total += value
            seen = True
    return total if seen else None


async def _get_json(url: str, path: str, api_key: str, params: dict | None = None) -> dict:
    from services.extensions.transport import read_bounded

    target = endpoint(url, path)
    try:
        async with lua_async_http_client(
            timeout=_timeout(), follow_redirects=False, trust_env=False,
        ) as client:
            async with client.stream(
                "GET", target, params=params, headers=headers(api_key),
            ) as response:
                ok = response.status_code == 200
                content = await read_bounded(
                    response, max_bytes=MAX_RESPONSE_BYTES if ok else MAX_ERROR_BYTES,
                )
                if not ok:
                    diagnostic = sanitize_diagnostic(
                        content.decode("utf-8", errors="replace"), secrets=[api_key],
                    )
                    raise ExternalProviderError(
                        classify_messages_failure(diagnostic, response.status_code),
                        status_code=response.status_code,
                    )
    except ExternalProviderError:
        raise
    except httpx.TimeoutException:
        raise ExternalProviderError("provider_timeout") from None
    except httpx.HTTPError:
        raise ExternalProviderError("provider_connection_failed") from None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise ExternalProviderError("provider_malformed_response") from None
    if not isinstance(data, dict):
        raise ExternalProviderError("provider_malformed_response")
    return data


async def get_model(url: str, model: str, api_key: str) -> dict:
    """GET /v1/models/{id}: proves the credential can see this model."""
    if not model or len(model) > 300:
        raise ValueError("A model identifier is required")
    return await _get_json(url, "models/" + quote(model, safe=""), api_key)


async def list_models(url: str, api_key: str) -> list[dict]:
    """Public model identifiers available to the supplied credential."""
    data = await _get_json(url, "models", api_key, params={"limit": 100})
    rows = data.get("data")
    if not isinstance(rows, list):
        raise ExternalProviderError("provider_malformed_response")
    models = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            continue
        if row.get("lifecycle") == "retired":
            continue
        models.append({
            "id": row["id"][:200],
            "name": str(row.get("display_name") or row["id"])[:200],
        })
    return models
