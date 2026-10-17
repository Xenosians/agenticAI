"""Verified HTTP syntax support, selected separately by trusted caller intent.

This is output formatting, never tool authority or a replacement for parsing,
grounding, SemanticGuard, approvals, or execution policy.
"""
from __future__ import annotations

from enum import StrEnum

from services.extensions.providers import get_provider_preset
from subagents.llm.runtime.stages import InferenceStage, resolve_stage_policy


class HttpResponseFormat(StrEnum):
    JSON_OBJECT = "json_object"


# Primary provider contracts, verified 2026-10-16:
# https://console.groq.com/docs/structured-outputs#json-object-mode
# https://console.groq.com/docs/api-reference#chat-create
# Deliberately narrow compatibility metadata; model identity does not pick a stage.
_GROQ_JSON_OBJECT_MODELS = frozenset({"openai/gpt-oss-20b", "openai/gpt-oss-120b"})


def supported_http_response_formats(profile: dict) -> frozenset[HttpResponseFormat]:
    url, model = profile.get("url"), profile.get("model")
    if (profile.get("backend", "http") == "http"
            and isinstance(url, str) and isinstance(model, str)
            and url.rstrip("/") == get_provider_preset("groq")["url"]
            and model in _GROQ_JSON_OBJECT_MODELS):
        return frozenset({HttpResponseFormat.JSON_OBJECT})
    return frozenset()


def http_response_format_for_stage(
    profile: dict, *, stage: InferenceStage | str,
    repair_of: InferenceStage | str | None = None,
) -> HttpResponseFormat | None:
    resolve_stage_policy(stage, repair_of)
    origin = InferenceStage(repair_of) if InferenceStage(stage) is InferenceStage.REPAIR else InferenceStage(stage)
    if (origin in {InferenceStage.SEMANTIC_ROUTING, InferenceStage.SEMANTIC_ROUTING_BROAD}
            and HttpResponseFormat.JSON_OBJECT in supported_http_response_formats(profile)):
        # The existing protocol has sparse dynamic argument maps and omitted
        # conditional fields. Groq strict schemas require closed objects and all
        # fields required. JSON syntax mode preserves that protocol unchanged.
        return HttpResponseFormat.JSON_OBJECT
    return None
