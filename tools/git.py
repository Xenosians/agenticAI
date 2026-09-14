from typing import Any

from services.process_runner import (
    run_process,
)


DEFAULT_TIMEOUT_SECONDS = 10


def workspace_git_status(
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Inspect the current branch and working-tree status.

    The command shape is owned by trusted application code.
    The model cannot supply arbitrary Git arguments.
    """

    return run_process(
        executable="git",
        args=[
            "status",
            "--short",
            "--branch",
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )


def workspace_git_branches(
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    List local Git branches.

    This is a fixed read-only Git operation.
    """

    return run_process(
        executable="git",
        args=[
            "branch",
            "--list",
            "--no-color",
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )


def workspace_git_log(
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Return the latest twenty commits in compact form.

    The history limit and output format are fixed by trusted code.
    """

    return run_process(
        executable="git",
        args=[
            "log",
            "--oneline",
            "--no-decorate",
            "-n",
            "20",
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )


def workspace_git_diff(
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Return the current unstaged working-tree diff.

    External diff helpers and terminal color are explicitly
    disabled so the result remains deterministic text.
    """

    return run_process(
        executable="git",
        args=[
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--unified=3",
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )


def workspace_git_changed_files(
    cwd: str | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Return paths currently changed in the working tree.

    This deliberately exposes filenames only.
    """

    return run_process(
        executable="git",
        args=[
            "diff",
            "--no-ext-diff",
            "--name-only",
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )