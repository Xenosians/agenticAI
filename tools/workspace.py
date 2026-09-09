from pathlib import Path
from typing import Any

from services.process_runner import (
    run_process,
    workspace_root,
)


MAX_TEXT_READ_CHARS = 16_000

ALLOWED_TEXT_SUFFIXES = {
    ".css",
    ".ex",
    ".exs",
    ".html",
    ".js",
    ".json",
    ".md",
    ".nim",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

ALLOWED_EXTENSIONLESS_FILES = {
    "LICENSE",
}

DENIED_NAME_FRAGMENTS = {
    "credential",
    "password",
    "secret",
    "token",
}


def workspace_mkdir(
    directory_name: str,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Create one direct-child directory through the governed
    native process runner.

    The caller supplies only the directory name.

    The trusted runner still owns:
    - mkdir executable selection
    - argument validation
    - workspace restriction
    - timeout policy
    - actual process execution
    """

    return run_process(
        executable="mkdir",
        args=[
            directory_name,
        ],
        cwd=cwd,
        timeout_seconds=(
            timeout_seconds
        ),
    )


def workspace_git_status(
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Inspect the Git status of one repository inside the approved
    developer workspace.

    The command shape is fixed by trusted code:
        git status --short --branch

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


def _validate_readable_text_path(
    relative_path: str,
) -> tuple[
    bool,
    str | None,
]:
    """
    Apply conservative file-type policy before reading.

    This first version intentionally supports source code,
    documentation, and common text configuration formats only.
    """

    path = Path(
        relative_path
    )

    file_name = (
        path.name
        .lower()
    )

    if any(
        fragment in file_name
        for fragment
        in DENIED_NAME_FRAGMENTS
    ):
        return (
            False,
            (
                "The requested file name is blocked "
                "by the sensitive-file policy."
            ),
        )

    if path.name in ALLOWED_EXTENSIONLESS_FILES:
        return True, None

    suffix = (
        path.suffix
        .lower()
    )

    if suffix not in ALLOWED_TEXT_SUFFIXES:
        return (
            False,
            (
                "The requested file type is not "
                "allowed by the current text-read policy."
            ),
        )

    return True, None


def workspace_read_text(
    relative_path: str,
) -> dict[str, Any]:
    """
    Read one UTF-8 text file from the approved developer
    workspace.

    Security properties:
    - path must be relative
    - resolved path must remain inside PROCESS_WORKSPACE_ROOT
    - directories cannot be read as files
    - only approved text/source file types are allowed
    - common sensitive-file names are denied
    - output is bounded
    - binary/non-UTF-8 content is rejected

    This capability does not invoke a shell or subprocess.
    """

    if not isinstance(
        relative_path,
        str,
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "relative_path must be a string."
            ),
        }

    relative_path = (
        relative_path.strip()
    )

    if not relative_path:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "relative_path cannot be empty."
            ),
        }

    if "\x00" in relative_path:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "relative_path contains an "
                "invalid null character."
            ),
        }

    requested = Path(
        relative_path
    )

    if requested.is_absolute():
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "Only workspace-relative file "
                "paths are allowed."
            ),
        }

    (
        readable,
        policy_error,
    ) = _validate_readable_text_path(
        relative_path
    )

    if not readable:
        return {
            "ok": False,
            "status": "denied",
            "error": policy_error,
        }

    root = workspace_root()

    try:
        candidate = (
            root
            / requested
        ).resolve()

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Could not resolve the requested "
                f"workspace file: {exc}"
            ),
        }

    if (
        candidate == root
        or root not in candidate.parents
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "Requested file is outside the "
                "approved workspace."
            ),
        }

    if not candidate.exists():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Requested workspace file "
                "does not exist."
            ),
        }

    if not candidate.is_file():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Requested workspace path "
                "is not a file."
            ),
        }

    try:
        with candidate.open(
            "r",
            encoding="utf-8",
        ) as handle:
            content = handle.read(
                MAX_TEXT_READ_CHARS
                + 1
            )

    except UnicodeDecodeError:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "Requested file is not valid "
                "UTF-8 text."
            ),
        }

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Workspace file read failed: "
                f"{exc}"
            ),
        }

    truncated = (
        len(content)
        > MAX_TEXT_READ_CHARS
    )

    if truncated:
        content = content[
            :MAX_TEXT_READ_CHARS
        ]

    normalized_path = (
        candidate
        .relative_to(root)
        .as_posix()
    )

    return {
        "ok": True,
        "status": "success",
        "path": normalized_path,
        "content": content,
        "truncated": truncated,
        "error": None,
    }