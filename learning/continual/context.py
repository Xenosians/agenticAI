"""Compatibility layer for the first Phase-5 ZIP context API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from learning.context.assembler import assemble_context_bundle
from learning.context.models import LearningContextBundle, StructuredContextRecord
from learning.context.recorder import LearningContextRecorder, load_context_records


ContextEvent = StructuredContextRecord
ContextBundle = LearningContextBundle


class ContextEventRecorder(LearningContextRecorder):
    def record(
        self,
        *,
        source: str,
        subject: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        trajectory_id: str | None = None,
        task_id: str | None = None,
        job_id: str | None = None,
    ) -> StructuredContextRecord | None:
        source_map = {
            "git": "git",
            "jira": "jira",
            "shell": "shell",
            "human": "human",
        }

        if source not in source_map:
            raise ValueError(f"unsupported context source: {source}")

        return super().record(
            kind=source_map[source],
            subject=subject,
            payload={
                "content": content,
                "metadata": metadata or {},
                "legacy": True,
            },
            trust="human" if source == "human" else "runtime",
            trajectory_id=trajectory_id,
            task_id=task_id,
            job_id=job_id,
        )


def load_context_events(path: Path) -> list[StructuredContextRecord]:
    return load_context_records(path)


def build_context_bundle(
    events: list[StructuredContextRecord],
    *,
    max_events: int = 64,
    max_chars: int = 24_000,
    per_source_limit: int = 24,
) -> LearningContextBundle:
    """
    Build a bounded bundle from an explicitly supplied legacy event list.

    The legacy API means "bundle these events" and therefore does not apply
    a second lineage filter here. The primary structured-context API remains
    lineage-aware through ``learning.context.assemble_context_bundle``.
    """

    return assemble_context_bundle(
        events,
        max_records=max_events,
        max_payload_chars=max_chars,
        per_kind_limit=per_source_limit,
        include_all_records=True,
    )
