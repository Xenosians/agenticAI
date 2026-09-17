from pathlib import Path
from typing import Any

from services.process_runner import (
    run_process,
    workspace_root,
)

from tools.workspace.policy import (
    validate_readable_text_path,
)


MAX_TEXT_READ_CHARS = 16_000


def workspace_mkdir(
    directory_name: str,
    cwd: str | None = None,
    timeout_seconds: int = 10,
) -> dict[str, Any]:
    """
    Create one direct-child directory through the governed
    native process runner.
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


def workspace_read_text(
    relative_path: str,
) -> dict[str, Any]:
    """
    Read one approved UTF-8 text file from the developer
    workspace.

    Security properties:

    - path must be relative
    - resolved path remains inside PROCESS_WORKSPACE_ROOT
    - directories cannot be read as files
    - only approved source/text formats are allowed
    - sensitive paths are denied
    - output is bounded
    - binary/non-UTF-8 data is rejected
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
    ) = validate_readable_text_path(
        relative_path
    )

    if not readable:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                policy_error
            ),
        }

    root = (
        workspace_root()
        .resolve()
    )

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

    if candidate.is_symlink():
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "Symbolic links cannot be read "
                "through workspace_read_text."
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

        "path":
            normalized_path,

        "content":
            content,

        "truncated":
            truncated,

        "error":
            None,
    }