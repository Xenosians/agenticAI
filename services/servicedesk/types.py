"""Result types of the Jira Service Management provider."""
from __future__ import annotations

from pydantic import BaseModel, Field


class ServiceDesk(BaseModel):
    id: str
    project_id: str
    project_key: str
    project_name: str


class RequestType(BaseModel):
    id: str
    name: str
    can_create: bool = True


class RequestTypeField(BaseModel):
    field_id: str
    name: str
    required: bool = False


class SlaMetric(BaseModel):
    name: str
    # ongoing, paused, completed, or not_started (no cycle yet).
    state: str
    breached: bool = False
    goal: str | None = None
    elapsed: str | None = None
    remaining: str | None = None
    completed_cycles: int = 0


class SlaStatusResult(BaseModel):
    ok: bool
    status: str
    provider: str = "jira"
    request_key: str | None = None
    metrics: list[SlaMetric] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    error: str | None = None
    error_code: str | None = None


class ServiceDeskQueue(BaseModel):
    id: str
    name: str
    issue_count: int | None = None
    jql: str | None = None


class QueueListResult(BaseModel):
    ok: bool
    status: str
    provider: str = "jira"
    service_desk: str | None = None
    service_desk_name: str | None = None
    queues: list[ServiceDeskQueue] = Field(default_factory=list)
    count: int = 0
    truncated: bool = False
    error: str | None = None
    error_code: str | None = None


class ServiceRequestResult(BaseModel):
    ok: bool
    status: str
    provider: str | None = "jira"
    operation: str | None = None
    ticket_key: str | None = None
    service_desk: str | None = None
    request_type: str | None = None
    summary: str | None = None
    request_status: str | None = None
    changed: bool = False
    mutation_performed: bool | None = None
    verification_ok: bool = False
    reconciled: bool = False
    message: str | None = None
    error: str | None = None
    error_code: str | None = None
