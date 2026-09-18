from __future__ import annotations

from typing import Any

from learning.context.models import ShellContextSnapshot, StructuredContextRecord
from learning.context.recorder import LearningContextRecorder


def record_shell_context(
    recorder: LearningContextRecorder,
    *,
    subject: str,
    workspace: str | None = None,
    repository: str | None = None,
    files: list[str] | None = None,
    language: str | None = None,
    toolchain: list[str] | None = None,
    dependencies: list[str] | None = None,
    previous_stdout: str | None = None,
    previous_stderr: str | None = None,
    exit_code: int | None = None,
    tests: list[dict[str, Any]] | None = None,
    builds: list[dict[str, Any]] | None = None,
    previous_commands: list[dict[str, Any]] | None = None,
    current_task: str | None = None,
    trajectory_id: str | None = None,
    task_id: str | None = None,
    job_id: str | None = None,
    source_tool: str | None = None,
) -> StructuredContextRecord | None:
    snapshot = ShellContextSnapshot(
        workspace=workspace,
        repository=repository,
        files=files or [],
        language=language,
        toolchain=toolchain or [],
        dependencies=dependencies or [],
        previous_stdout=previous_stdout,
        previous_stderr=previous_stderr,
        exit_code=exit_code,
        tests=tests or [],
        builds=builds or [],
        previous_commands=previous_commands or [],
        current_task=current_task,
    )
    return recorder.record(
        kind="shell",
        subject=subject,
        payload=snapshot.model_dump(mode="json"),
        trust="trusted_provider" if source_tool else "runtime",
        trajectory_id=trajectory_id,
        task_id=task_id,
        job_id=job_id,
        source_tool=source_tool,
    )
