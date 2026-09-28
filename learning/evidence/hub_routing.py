from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid

from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from learning.evidence.execution_provenance import (
    fingerprint_runtime_model_artifact,
)
from learning.evidence.sanitizer import sanitize_value
from learning.paths import RUNTIME_LEARNING_ROOT
from subagents.llm.runtime.hf_prompt import (
    resolve_prompt_renderer_sha256,
)


HubRoutingMode = Literal["normal", "repair"]
HubRoutingValidation = Literal["pending", "accepted", "rejected"]

DEFAULT_HUB_ROUTING_LEDGER = (
    RUNTIME_LEARNING_ROOT / "hub-routing-attempts.jsonl"
)

_TRACE: ContextVar[tuple["HubRoutingAttemptEvidence", ...]] = ContextVar(
    "learning_hub_routing_trace",
    default=(),
)

_ACTIVE: ContextVar[dict[str, Any] | None] = ContextVar(
    "learning_hub_routing_active_attempt",
    default=None,
)

_LEDGER_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


class HubModelIdentity(BaseModel):
    """Cryptographic identity of the Hub model/runtime environment."""

    model_config = ConfigDict(extra="forbid")

    complete: bool = False
    model_key: str

    backend: str | None = None
    quantization: str | None = None
    compute_dtype: str | None = None
    device_map: str | None = None

    model_profile_sha256: str | None = None

    model_artifact_sha256: str | None = None
    model_weights_sha256: str | None = None
    tokenizer_artifact_sha256: str | None = None
    model_config_sha256: str | None = None
    generation_config_sha256: str | None = None

    tokenizer_config_sha256: str | None = None
    tokenizer_json_sha256: str | None = None
    chat_template_sha256: str | None = None
    prompt_renderer_sha256: str | None = None

    incomplete_reason: str | None = None


class HubRoutingAttemptEvidence(BaseModel):
    """
    One Hub routing generation identity.

    Raw model output is persisted only after the existing learning sanitizer.
    raw_response_exact states whether sanitization preserved the generation
    byte-for-byte. Runtime evidence never promotes itself into training data.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="hub-routing-attempt.v1",
        alias="schema",
    )

    attempt_id: str
    created_at: str

    mode: HubRoutingMode
    include_workflow_protocol: bool

    model_identity: HubModelIdentity
    max_new_tokens: int = Field(ge=1)

    system_prompt_sha256: str
    capability_catalog_sha256: str
    messages_sha256: str
    conversation_context_sha256: str
    user_request_sha256: str

    # Sanitized exact historical model input. Older v4A records omit this.
    messages: list[dict[str, str]] | None = None
    sanitized_messages_sha256: str | None = None
    messages_exact: bool = False

    repair_error_sha256: str | None = None

    raw_response: str
    raw_response_sha256: str
    sanitized_response_sha256: str
    raw_response_exact: bool

    parsed_output: dict[str, Any] | None = None

    validation_status: HubRoutingValidation = "pending"
    validation_error: str | None = None
    validated_delegation_count: int | None = None

    training_eligible: bool = False


class HubRoutingLedgerRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(
        default="hub-routing-ledger-record.v1",
        alias="schema",
    )

    trajectory_id: str
    job_id: str
    job_attempt: int = Field(ge=1)

    attempt: HubRoutingAttemptEvidence
    training_eligible: bool = False


ModelProfileResolver = Callable[[str], Any]


def _model_profile_payload(profile: Any) -> dict[str, Any]:
    return {
        "backend": getattr(profile, "backend", None),
        "quantization": getattr(profile, "quantization", None),
        "compute_dtype": getattr(profile, "compute_dtype", None),
        "device_map": getattr(profile, "device_map", None),
        "worker_prompt_profile": getattr(
            profile,
            "worker_prompt_profile",
            None,
        ),
        "dequantize_fp8": getattr(profile, "dequantize_fp8", None),
        "bnb_4bit_quant_type": getattr(
            profile,
            "bnb_4bit_quant_type",
            None,
        ),
        "bnb_4bit_use_double_quant": getattr(
            profile,
            "bnb_4bit_use_double_quant",
            None,
        ),
    }


def build_hub_model_identity(
    *,
    model_key: str,
    model_profile_resolver: ModelProfileResolver | None,
) -> HubModelIdentity:
    normalized_model_key = model_key.strip()

    if not normalized_model_key:
        raise ValueError("Hub model key must not be empty.")

    if model_profile_resolver is None:
        return HubModelIdentity(
            complete=False,
            model_key=normalized_model_key,
            incomplete_reason="model-profile-resolver-unavailable",
        )

    try:
        profile = model_profile_resolver(normalized_model_key)
    except Exception:
        return HubModelIdentity(
            complete=False,
            model_key=normalized_model_key,
            incomplete_reason="model-profile-resolution-failed",
        )

    profile_hash = _sha256_json(_model_profile_payload(profile))
    model_path = getattr(profile, "model_path", None)

    common = {
        "model_key": normalized_model_key,
        "backend": getattr(profile, "backend", None),
        "quantization": getattr(profile, "quantization", None),
        "compute_dtype": getattr(profile, "compute_dtype", None),
        "device_map": getattr(profile, "device_map", None),
        "model_profile_sha256": profile_hash,
    }

    if model_path is None:
        return HubModelIdentity(
            complete=False,
            incomplete_reason="model-path-unavailable",
            **common,
        )

    try:
        fingerprint = fingerprint_runtime_model_artifact(Path(model_path))
        prompt_renderer_sha256 = resolve_prompt_renderer_sha256(
            backend=getattr(profile, "backend", ""),
            native_chat_template_sha256=fingerprint.chat_template_sha256,
        )
    except Exception:
        return HubModelIdentity(
            complete=False,
            incomplete_reason="model-artifact-fingerprint-failed",
            **common,
        )

    complete = bool(
        fingerprint.content_sha256
        and fingerprint.weights_sha256
        and fingerprint.tokenizer_sha256
        and fingerprint.config_sha256
        and prompt_renderer_sha256
    )

    return HubModelIdentity(
        complete=complete,
        model_artifact_sha256=fingerprint.content_sha256,
        model_weights_sha256=fingerprint.weights_sha256,
        tokenizer_artifact_sha256=fingerprint.tokenizer_sha256,
        model_config_sha256=fingerprint.config_sha256,
        generation_config_sha256=fingerprint.generation_config_sha256,
        tokenizer_config_sha256=fingerprint.tokenizer_config_sha256,
        tokenizer_json_sha256=fingerprint.tokenizer_json_sha256,
        chat_template_sha256=fingerprint.chat_template_sha256,
        prompt_renderer_sha256=prompt_renderer_sha256,
        incomplete_reason=(None if complete else "model-identity-incomplete"),
        **common,
    )


def reset_hub_routing_trace() -> None:
    _TRACE.set(())
    _ACTIVE.set(None)


def capture_hub_routing_generation(
    *,
    model_key: str,
    model_profile_resolver: ModelProfileResolver | None,
    specialists: list[dict[str, Any]],
    messages: list[dict[str, str]],
    context_messages: list[dict[str, str]],
    user_request: str,
    repair_mode: bool,
    include_workflow_protocol: bool,
    repair_error: str | None,
    max_new_tokens: int,
    response: str,
) -> HubRoutingAttemptEvidence:
    if not messages:
        raise ValueError("Hub routing messages must not be empty.")

    system_message = messages[0]

    if system_message.get("role") != "system":
        raise ValueError(
            "Hub routing messages must begin with a system message."
        )

    system_prompt = system_message.get("content")

    if not isinstance(system_prompt, str):
        raise ValueError("Hub routing system prompt is invalid.")

    sanitized_response = sanitize_value(response)

    if not isinstance(sanitized_response, str):
        sanitized_response = str(sanitized_response)

    sanitized_messages = sanitize_value(messages)

    messages_valid = (
        isinstance(sanitized_messages, list)
        and all(
            isinstance(item, dict)
            and isinstance(item.get("role"), str)
            and isinstance(item.get("content"), str)
            for item in sanitized_messages
        )
    )

    archived_messages = (
        [
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in sanitized_messages
        ]
        if messages_valid
        else None
    )

    messages_exact = (
        archived_messages == messages
        if archived_messages is not None
        else False
    )

    attempt = HubRoutingAttemptEvidence(
        attempt_id="hub-route-" + uuid.uuid4().hex,
        created_at=_utc_now(),
        mode=("repair" if repair_mode else "normal"),
        include_workflow_protocol=include_workflow_protocol,
        model_identity=build_hub_model_identity(
            model_key=model_key,
            model_profile_resolver=model_profile_resolver,
        ),
        max_new_tokens=max_new_tokens,
        system_prompt_sha256=_sha256_text(system_prompt),
        capability_catalog_sha256=_sha256_json(specialists),
        messages_sha256=_sha256_json(messages),
        conversation_context_sha256=_sha256_json(context_messages),
        user_request_sha256=_sha256_text(user_request),
        messages=archived_messages,
        sanitized_messages_sha256=(
            _sha256_json(archived_messages)
            if archived_messages is not None
            else None
        ),
        messages_exact=messages_exact,
        repair_error_sha256=(
            _sha256_text(repair_error.strip())
            if isinstance(repair_error, str) and repair_error.strip()
            else None
        ),
        raw_response=sanitized_response,
        raw_response_sha256=_sha256_text(response),
        sanitized_response_sha256=_sha256_text(sanitized_response),
        raw_response_exact=(sanitized_response == response),
        parsed_output=None,
        validation_status="pending",
        validation_error=None,
        validated_delegation_count=None,
        training_eligible=False,
    )

    _ACTIVE.set(
        attempt.model_dump(
            mode="python",
            by_alias=False,
        )
    )

    return attempt


def set_current_hub_routing_parsed_output(
    parsed_output: dict[str, Any],
) -> None:
    active = _ACTIVE.get()

    if active is None:
        return

    sanitized = sanitize_value(parsed_output)

    if not isinstance(sanitized, dict):
        sanitized = None

    updated = dict(active)
    updated["parsed_output"] = sanitized
    _ACTIVE.set(updated)


def _finish_active(
    *,
    validation_status: Literal["accepted", "rejected"],
    validation_error: str | None,
    validated_delegation_count: int | None,
) -> HubRoutingAttemptEvidence | None:
    active = _ACTIVE.get()

    if active is None:
        return None

    updated = dict(active)
    updated["validation_status"] = validation_status
    updated["validation_error"] = validation_error
    updated["validated_delegation_count"] = validated_delegation_count
    updated["training_eligible"] = False

    attempt = HubRoutingAttemptEvidence.model_validate(updated)

    trace = list(_TRACE.get())
    trace.append(attempt)
    _TRACE.set(tuple(trace))
    _ACTIVE.set(None)

    return attempt


def reject_current_hub_routing_attempt(
    error: str,
) -> HubRoutingAttemptEvidence | None:
    normalized_error = str(error).strip() or "hub-routing-contract-rejected"

    return _finish_active(
        validation_status="rejected",
        validation_error=normalized_error,
        validated_delegation_count=None,
    )


def accept_current_hub_routing_attempt(
    *,
    validated_delegation_count: int,
) -> HubRoutingAttemptEvidence | None:
    if validated_delegation_count < 0:
        raise ValueError(
            "validated_delegation_count must be non-negative."
        )

    return _finish_active(
        validation_status="accepted",
        validation_error=None,
        validated_delegation_count=validated_delegation_count,
    )


def current_hub_routing_trace() -> list[HubRoutingAttemptEvidence]:
    return [item.model_copy(deep=True) for item in _TRACE.get()]


def consume_hub_routing_trace() -> list[HubRoutingAttemptEvidence]:
    result = current_hub_routing_trace()
    reset_hub_routing_trace()
    return result


def record_current_hub_routing_ledger(
    *,
    trajectory_id: str,
    job_id: str,
    job_attempt: int,
    path: Path = DEFAULT_HUB_ROUTING_LEDGER,
) -> int:
    """
    Consume the current async task's Hub routing trace and append it to a
    durable ledger after the normal trajectory is already durable.
    """

    attempts = consume_hub_routing_trace()

    if not attempts:
        return 0

    normalized_trajectory_id = trajectory_id.strip()
    normalized_job_id = job_id.strip()

    if not normalized_trajectory_id:
        raise ValueError("trajectory_id must not be empty.")

    if not normalized_job_id:
        raise ValueError("job_id must not be empty.")

    if job_attempt < 1:
        raise ValueError("job_attempt must be positive.")

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    for attempt in attempts:
        record = HubRoutingLedgerRecord(
            trajectory_id=normalized_trajectory_id,
            job_id=normalized_job_id,
            job_attempt=job_attempt,
            attempt=attempt,
            training_eligible=False,
        )

        payload = sanitize_value(
            record.model_dump(
                mode="json",
                by_alias=True,
            )
        )

        if not isinstance(payload, dict):
            raise ValueError(
                "Sanitized Hub routing record must remain an object."
            )

        lines.append(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    with _LEDGER_LOCK:
        with resolved.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")

            handle.flush()
            os.fsync(handle.fileno())

    return len(lines)
