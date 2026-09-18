from __future__ import annotations

from typing import Any

from learning.context.models import GitContextSnapshot, StructuredContextRecord
from learning.context.recorder import LearningContextRecorder


def record_git_context(
    recorder: LearningContextRecorder,
    *,
    repository: str,
    branch: str | None = None,
    clean: bool | None = None,
    ahead: int | None = None,
    behind: int | None = None,
    status_changes: list[dict[str, Any]] | None = None,
    recent_commits: list[dict[str, Any]] | None = None,
    diff_summary: str | None = None,
    changed_files: list[str] | None = None,
    remotes: list[str] | None = None,
    previous_results: list[dict[str, Any]] | None = None,
    task_goal: str | None = None,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
    source_tool: str | None = None,
) -> StructuredContextRecord | None:
    snapshot = GitContextSnapshot(
        repository=repository,
        branch=branch,
        clean=clean,
        ahead=ahead,
        behind=behind,
        status_changes=status_changes or [],
        recent_commits=recent_commits or [],
        diff_summary=diff_summary,
        changed_files=changed_files or [],
        remotes=remotes or [],
        previous_results=previous_results or [],
        task_goal=task_goal,
    )
    return recorder.record(
        kind="git",
        subject=f"repository:{repository}",
        payload=snapshot.model_dump(mode="json"),
        trust="trusted_provider" if source_tool else "runtime",
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
        source_tool=source_tool,
    )
