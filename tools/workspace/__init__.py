from pathlib import Path
from typing import Any

from services.process_runner import (
    run_process,
    workspace_root,
)

from services.workspace_repositories import (
    resolve_workspace_repository,
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

    Logical-repository-aware mutation targeting is introduced in
    the mutation-specific hardening pass. B2a changes only safe
    discovery/read operations.
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


def _workspace_root_for_repository(
    repository: (
        str
        | None
    ),
) -> tuple[
    Path | None,
    str | None,
    str | None,
]:
    """
    Resolve the trust root for one workspace read.

    Backward compatibility:

        repository is None
            -> existing PROCESS_WORKSPACE_ROOT behavior

        repository is supplied
            -> trusted configured logical repository root
    """

    if repository is None:
        try:
            return (
                workspace_root()
                .resolve(),
                None,
                None,
            )

        except Exception as exc:
            return (
                None,
                None,
                str(
                    exc
                ),
            )

    try:
        target = (
            resolve_workspace_repository(
                repository
            )
        )

    except ValueError as exc:
        return (
            None,
            None,
            str(
                exc
            ),
        )

    return (
        target.path.resolve(),
        target.name,
        None,
    )


def workspace_read_text(
    relative_path: str,
    repository: (
        str
        | None
    ) = None,
) -> dict[str, Any]:
    """
    Read one approved UTF-8 source/text file.

    The optional repository argument is a logical identifier such
    as:

        ai
        backend
        frontend

    Physical repository roots remain trusted configuration.

    Security properties:

    - repository root is trusted runtime state
    - path must be repository/workspace relative
    - path remains inside the selected trust root
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
            "repository":
                repository,
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
            "repository":
                repository,
            "error": (
                "relative_path cannot be empty."
            ),
        }

    if "\x00" in relative_path:
        return {
            "ok": False,
            "status": "denied",
            "repository":
                repository,
            "error": (
                "relative_path contains an "
                "invalid null character."
            ),
        }

    requested = (
        Path(
            relative_path
        )
    )

    if requested.is_absolute():
        return {
            "ok": False,
            "status": "denied",
            "repository":
                repository,
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
            "repository":
                repository,
            "error":
                policy_error,
        }

    (
        root,
        repository_name,
        root_error,
    ) = _workspace_root_for_repository(
        repository
    )

    if root is None:
        return {
            "ok": False,
            "status": "denied",
            "repository":
                repository,
            "error":
                root_error,
        }

    try:
        candidate = (
            root
            / requested
        ).resolve()

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "repository":
                repository_name,
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
            "repository":
                repository_name,
            "error": (
                "Requested file is outside the "
                "approved repository workspace."
            ),
        }

    if not candidate.exists():
        return {
            "ok": False,
            "status": "error",
            "repository":
                repository_name,
            "error": (
                "Requested workspace file "
                "does not exist."
            ),
        }

    if not candidate.is_file():
        return {
            "ok": False,
            "status": "error",
            "repository":
                repository_name,
            "error": (
                "Requested workspace path "
                "is not a file."
            ),
        }

    if candidate.is_symlink():
        return {
            "ok": False,
            "status": "denied",
            "repository":
                repository_name,
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

            content = (
                handle.read(
                    MAX_TEXT_READ_CHARS
                    + 1
                )
            )

    except UnicodeDecodeError:
        return {
            "ok": False,
            "status": "denied",
            "repository":
                repository_name,
            "error": (
                "Requested file is not valid "
                "UTF-8 text."
            ),
        }

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "repository":
                repository_name,
            "error": (
                "Workspace file read failed: "
                f"{exc}"
            ),
        }

    truncated = (
        len(
            content
        )
        > MAX_TEXT_READ_CHARS
    )

    if truncated:
        content = (
            content[
                :MAX_TEXT_READ_CHARS
            ]
        )

    normalized_path = (
        candidate
        .relative_to(
            root
        )
        .as_posix()
    )

    return {
        "ok": True,
        "status": "success",

        "repository":
            repository_name,

        "path":
            normalized_path,

        "content":
            content,

        "truncated":
            truncated,

        "error":
            None,
    }
