from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learning.context.storage import append_jsonl, load_jsonl_models
from learning.context.models import ContextKind, ContextScope, ContextTrust, StructuredContextRecord
from learning.evidence.sanitizer import sanitize_value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope_for(*, task_id: str | None, trajectory_id: str | None, job_id: str | None) -> ContextScope:
    if task_id:
        return "task"
    if trajectory_id:
        return "trajectory"
    if job_id:
        return "job"
    return "global"


class LearningContextRecorder:
    """Append-only structured context ledger."""

    def __init__(self, *, path: Path, enabled: bool = True) -> None:
        self.path = path.expanduser().resolve()
        self.enabled = enabled

    def record(
        self,
        *,
        kind: ContextKind,
        subject: str,
        payload: dict[str, Any],
        trust: ContextTrust = "unknown",
        trajectory_id: str | None = None,
        task_id: str | None = None,
        job_id: str | None = None,
        source_tool: str | None = None,
        scope: ContextScope | None = None,
    ) -> StructuredContextRecord | None:
        if not self.enabled:
            return None

        sanitized_payload = sanitize_value(payload)
        if not isinstance(sanitized_payload, dict):
            raise ValueError("context payload must sanitize to an object")

        record = StructuredContextRecord(
            context_id=f"context-{uuid.uuid4().hex}",
            observed_at=_utc_now(),
            kind=kind,
            scope=scope or _scope_for(task_id=task_id, trajectory_id=trajectory_id, job_id=job_id),
            subject=subject,
            payload=sanitized_payload,
            trust=trust,
            trajectory_id=trajectory_id,
            task_id=task_id,
            job_id=job_id,
            source_tool=source_tool,
            sanitized=True,
            authority_grant=False,
            training_eligible=False,
        )
        append_jsonl(self.path, record)
        return record


def load_context_records(path: Path) -> list[StructuredContextRecord]:
    return load_jsonl_models(path, StructuredContextRecord)
