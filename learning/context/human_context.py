from __future__ import annotations

from learning.context.models import HumanContextSnapshot, StructuredContextRecord
from learning.context.recorder import LearningContextRecorder


def record_human_context(
    recorder: LearningContextRecorder,
    *,
    subject: str,
    explicit_constraints: list[str] | None = None,
    workflow_preferences: list[str] | None = None,
    project_conventions: list[str] | None = None,
    corrections: list[str] | None = None,
    task_decisions: list[str] | None = None,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
) -> StructuredContextRecord | None:
    snapshot = HumanContextSnapshot(
        explicit_constraints=explicit_constraints or [],
        workflow_preferences=workflow_preferences or [],
        project_conventions=project_conventions or [],
        corrections=corrections or [],
        task_decisions=task_decisions or [],
    )
    return recorder.record(
        kind="human",
        subject=subject,
        payload=snapshot.model_dump(mode="json"),
        trust="human",
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
    )
