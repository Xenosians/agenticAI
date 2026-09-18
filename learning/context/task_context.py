from __future__ import annotations

from typing import Any

from learning.context.models import StructuredContextRecord, TaskContextSnapshot, ToolContextSnapshot
from learning.context.recorder import LearningContextRecorder


def record_task_context(
    recorder: LearningContextRecorder,
    *,
    subject: str,
    original_request: str,
    routing_context: list[dict[str, Any]] | None = None,
    specialist_instructions: str | None = None,
    semantic_contract: dict[str, Any] | None = None,
    previous_trusted_results: list[dict[str, Any]] | None = None,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
) -> StructuredContextRecord | None:
    snapshot = TaskContextSnapshot(
        original_request=original_request,
        routing_context=routing_context or [],
        specialist_instructions=specialist_instructions,
        semantic_contract=semantic_contract or {},
        previous_trusted_results=previous_trusted_results or [],
    )
    return recorder.record(
        kind="task",
        subject=subject,
        payload=snapshot.model_dump(mode="json"),
        trust="system",
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
    )


def record_tool_context(
    recorder: LearningContextRecorder,
    *,
    subject: str,
    proposed_calls: list[dict[str, Any]] | None = None,
    guard_outcomes: list[dict[str, Any]] | None = None,
    gateway_outcomes: list[dict[str, Any]] | None = None,
    approval_outcomes: list[dict[str, Any]] | None = None,
    trusted_provider_results: list[dict[str, Any]] | None = None,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
    source_tool: str | None = None,
) -> StructuredContextRecord | None:
    snapshot = ToolContextSnapshot(
        proposed_calls=proposed_calls or [],
        guard_outcomes=guard_outcomes or [],
        gateway_outcomes=gateway_outcomes or [],
        approval_outcomes=approval_outcomes or [],
        trusted_provider_results=trusted_provider_results or [],
    )
    return recorder.record(
        kind="tool",
        subject=subject,
        payload=snapshot.model_dump(mode="json"),
        trust="trusted_provider" if trusted_provider_results else "runtime",
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
        source_tool=source_tool,
    )
