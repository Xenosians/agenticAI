from __future__ import annotations

from typing import Any

from learning.context.models import JiraContextSnapshot, StructuredContextRecord
from learning.context.recorder import LearningContextRecorder


def record_jira_context(
    recorder: LearningContextRecorder,
    *,
    project_key: str | None = None,
    ticket_key: str | None = None,
    summary: str | None = None,
    status: str | None = None,
    assignee: str | None = None,
    comments: list[dict[str, Any]] | None = None,
    relationships: list[dict[str, Any]] | None = None,
    prior_agent_actions: list[dict[str, Any]] | None = None,
    workflow_state: dict[str, Any] | None = None,
    user_request: str | None = None,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
    source_tool: str | None = None,
) -> StructuredContextRecord | None:
    if not project_key and not ticket_key:
        raise ValueError("jira context requires project_key or ticket_key")
    snapshot = JiraContextSnapshot(
        project_key=project_key,
        ticket_key=ticket_key,
        summary=summary,
        status=status,
        assignee=assignee,
        comments=comments or [],
        relationships=relationships or [],
        prior_agent_actions=prior_agent_actions or [],
        workflow_state=workflow_state or {},
        user_request=user_request,
    )
    subject = f"ticket:{ticket_key}" if ticket_key else f"project:{project_key}"
    return recorder.record(
        kind="jira",
        subject=subject,
        payload=snapshot.model_dump(mode="json"),
        trust="trusted_provider" if source_tool else "runtime",
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
        source_tool=source_tool,
    )
