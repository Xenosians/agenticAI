from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


ContextKind = Literal["git", "jira", "shell", "human", "task", "tool"]
ContextScope = Literal["global", "job", "trajectory", "task"]
ContextTrust = Literal["runtime", "trusted_provider", "human", "system", "unknown"]


class GitContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str
    branch: str | None = None
    clean: bool | None = None
    ahead: int | None = None
    behind: int | None = None
    status_changes: list[dict[str, Any]] = Field(default_factory=list)
    recent_commits: list[dict[str, Any]] = Field(default_factory=list)
    diff_summary: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    remotes: list[str] = Field(default_factory=list)
    previous_results: list[dict[str, Any]] = Field(default_factory=list)
    task_goal: str | None = None


class JiraContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_key: str | None = None
    ticket_key: str | None = None
    summary: str | None = None
    status: str | None = None
    assignee: str | None = None
    comments: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)
    prior_agent_actions: list[dict[str, Any]] = Field(default_factory=list)
    workflow_state: dict[str, Any] = Field(default_factory=dict)
    user_request: str | None = None


class ShellContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: str | None = None
    repository: str | None = None
    files: list[str] = Field(default_factory=list)
    language: str | None = None
    toolchain: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    previous_stdout: str | None = None
    previous_stderr: str | None = None
    exit_code: int | None = None
    tests: list[dict[str, Any]] = Field(default_factory=list)
    builds: list[dict[str, Any]] = Field(default_factory=list)
    previous_commands: list[dict[str, Any]] = Field(default_factory=list)
    current_task: str | None = None


class HumanContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explicit_constraints: list[str] = Field(default_factory=list)
    workflow_preferences: list[str] = Field(default_factory=list)
    project_conventions: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    task_decisions: list[str] = Field(default_factory=list)


class TaskContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_request: str
    routing_context: list[dict[str, Any]] = Field(default_factory=list)
    specialist_instructions: str | None = None
    semantic_contract: dict[str, Any] = Field(default_factory=dict)
    previous_trusted_results: list[dict[str, Any]] = Field(default_factory=list)


class ToolContextSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposed_calls: list[dict[str, Any]] = Field(default_factory=list)
    guard_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    gateway_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    approval_outcomes: list[dict[str, Any]] = Field(default_factory=list)
    trusted_provider_results: list[dict[str, Any]] = Field(default_factory=list)


class StructuredContextRecord(BaseModel):
    """Sanitized contextual evidence. It can inform learning but never authority."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(default="learning-context-record.v2", alias="schema")
    context_id: str
    observed_at: str
    kind: ContextKind
    scope: ContextScope
    subject: str
    payload: dict[str, Any]
    trust: ContextTrust = "unknown"
    trajectory_id: str | None = None
    task_id: str | None = None
    job_id: str | None = None
    source_tool: str | None = None
    sanitized: bool = True
    authority_grant: bool = False
    training_eligible: bool = False

    @field_validator("subject")
    @classmethod
    def non_empty_subject(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("context subject must not be empty")
        return normalized

    @field_validator("sanitized")
    @classmethod
    def require_sanitized(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("context must be sanitized before persistence")
        return value

    @field_validator("authority_grant", "training_eligible")
    @classmethod
    def require_non_authoritative(cls, value: bool) -> bool:
        if value is not False:
            raise ValueError("context cannot grant authority or training eligibility")
        return value

    @model_validator(mode="after")
    def validate_scope_lineage(self) -> "StructuredContextRecord":
        if self.scope == "task" and not self.task_id:
            raise ValueError("task-scoped context requires task_id")
        if self.scope == "trajectory" and not self.trajectory_id:
            raise ValueError("trajectory-scoped context requires trajectory_id")
        if self.scope == "job" and not self.job_id:
            raise ValueError("job-scoped context requires job_id")
        return self


class LearningContextBundle(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_name: str = Field(default="learning-context-bundle.v2", alias="schema")
    created_at: str
    trajectory_id: str | None = None
    task_id: str | None = None
    job_id: str | None = None
    record_count: int
    kind_counts: dict[str, int] = Field(default_factory=dict)
    content_sha256: str
    records: list[StructuredContextRecord] = Field(default_factory=list)
