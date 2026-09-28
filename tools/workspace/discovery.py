import os

from pathlib import Path
from typing import Any

from services.process_runner import (
    workspace_root,
)

from services.workspace_repositories import (
    resolve_workspace_repository,
)

from tools.workspace.policy import (
    DENIED_DIRECTORY_NAMES,
    validate_discoverable_path,
    validate_readable_text_path,
)


MAX_LIST_ENTRIES = 200
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_QUERY_CHARS = 128
MAX_SEARCH_FILE_BYTES = 1_000_000
MAX_MATCH_LINE_CHARS = 500


class WorkspacePathError(
    ValueError
):
    pass


def _resolve_workspace_path(
    relative_path: str,
    repository: (
        str
        | None
    ) = None,
) -> tuple[
    Path,
    Path,
    str | None,
]:
    """
    Resolve one safe path inside either the configured global
    workspace root or one explicitly selected logical repository.

    Logical repository identifiers such as:

        ai
        backend
        frontend

    are resolved through trusted runtime configuration.

    They are never interpreted as relative filesystem paths.
    """

    if not isinstance(
        relative_path,
        str,
    ):
        raise WorkspacePathError(
            "relative_path must be a string."
        )

    relative_path = (
        relative_path.strip()
    )

    if not relative_path:
        raise WorkspacePathError(
            "relative_path cannot be empty."
        )

    if "\x00" in relative_path:
        raise WorkspacePathError(
            "relative_path contains an invalid "
            "null character."
        )

    requested = (
        Path(
            relative_path
        )
    )

    if requested.is_absolute():
        raise WorkspacePathError(
            "Only workspace-relative paths "
            "are allowed."
        )

    (
        discoverable,
        policy_error,
    ) = validate_discoverable_path(
        relative_path
    )

    if not discoverable:
        raise WorkspacePathError(
            policy_error
            or "Workspace path is not allowed."
        )

    repository_name = None

    if repository is None:

        root = (
            workspace_root()
            .resolve()
        )

    else:

        try:
            target = (
                resolve_workspace_repository(
                    repository
                )
            )

        except ValueError as exc:
            raise WorkspacePathError(
                str(
                    exc
                )
            ) from exc

        root = (
            target.path
            .resolve()
        )

        repository_name = (
            target.name
        )

    try:
        candidate = (
            root
            / requested
        ).resolve()

    except OSError as exc:
        raise WorkspacePathError(
            "Could not resolve workspace path: "
            f"{exc}"
        ) from exc

    if (
        candidate != root
        and root not in candidate.parents
    ):
        raise WorkspacePathError(
            "Requested path is outside the "
            "approved workspace."
        )

    return (
        root,
        candidate,
        repository_name,
    )


def _normalized_relative_path(
    root: Path,
    candidate: Path,
) -> str:
    if candidate == root:
        return "."

    return (
        candidate
        .relative_to(root)
        .as_posix()
    )


def _entry_is_discoverable(
    root: Path,
    candidate: Path,
) -> bool:
    try:
        relative_path = (
            candidate
            .relative_to(root)
            .as_posix()
        )

    except ValueError:
        return False

    (
        allowed,
        _error,
    ) = validate_discoverable_path(
        relative_path
    )

    return allowed


def workspace_list(
    relative_path: str = ".",
    repository: (
        str
        | None
    ) = None,
) -> dict[str, Any]:
    """
    List direct children of one workspace directory.

    This does not invoke a shell and does not recursively walk
    the repository.
    """

    try:
        (
            root,
            candidate,
            repository_name,
        ) = _resolve_workspace_path(
            relative_path,
            repository=(
                repository
            ),
        )

    except WorkspacePathError as exc:
        return {
            "ok": False,
            "status": "denied",
            "error": str(
                exc
            ),
        }

    if not candidate.exists():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Requested workspace path "
                "does not exist."
            ),
        }

    if not candidate.is_dir():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "workspace_list requires "
                "a directory."
            ),
        }

    entries = []

    try:
        children = sorted(
            candidate.iterdir(),
            key=lambda item: (
                not item.is_dir(),
                item.name.lower(),
            ),
        )

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Workspace listing failed: "
                f"{exc}"
            ),
        }

    truncated = False

    for child in children:
        if not _entry_is_discoverable(
            root,
            child,
        ):
            continue

        if len(
            entries
        ) >= MAX_LIST_ENTRIES:
            truncated = True
            break

        relative_child = (
            child
            .relative_to(root)
            .as_posix()
        )

        try:
            stat_result = (
                child.lstat()
            )

        except OSError:
            continue

        if child.is_symlink():
            entry_type = (
                "symlink"
            )

        elif child.is_dir():
            entry_type = (
                "directory"
            )

        elif child.is_file():
            entry_type = (
                "file"
            )

        else:
            entry_type = (
                "other"
            )

        entries.append(
            {
                "name":
                    child.name,

                "path":
                    relative_child,

                "type":
                    entry_type,

                "size_bytes": (
                    stat_result.st_size
                    if entry_type
                    in {
                        "file",
                        "symlink",
                    }
                    else None
                ),
            }
        )

    return {
        "ok": True,
        "status": "success",

        "repository":
            repository_name,

        "path":
            _normalized_relative_path(
                root,
                candidate,
            ),

        "entries":
            entries,

        "truncated":
            truncated,

        "error":
            None,
    }


def workspace_search(
    query: str,
    relative_path: str = ".",
    repository: (
        str
        | None
    ) = None,
) -> dict[str, Any]:
    """
    Search approved UTF-8 source/text files recursively.

    Search behavior is deliberately bounded:

    - literal case-insensitive matching
    - safe text/source files only
    - sensitive/runtime directories skipped
    - symbolic links skipped
    - large files skipped
    - maximum result count enforced
    """

    if not isinstance(
        query,
        str,
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "query must be a string."
            ),
        }

    query = (
        query.strip()
    )

    if not query:
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "query cannot be empty."
            ),
        }

    if (
        len(query)
        > MAX_SEARCH_QUERY_CHARS
    ):
        return {
            "ok": False,
            "status": "denied",
            "error": (
                "query exceeds the maximum "
                "search length."
            ),
        }

    try:
        (
            root,
            candidate,
            repository_name,
        ) = _resolve_workspace_path(
            relative_path,
            repository=(
                repository
            ),
        )

    except WorkspacePathError as exc:
        return {
            "ok": False,
            "status": "denied",
            "error": str(
                exc
            ),
        }

    if not candidate.exists():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Requested workspace path "
                "does not exist."
            ),
        }

    if not candidate.is_dir():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "workspace_search requires "
                "a directory."
            ),
        }

    normalized_query = (
        query.casefold()
    )

    matches = []
    files_scanned = 0
    truncated = False

    for (
        current_root,
        directory_names,
        file_names,
    ) in os.walk(
        candidate,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(
            current_root
        )

        allowed_directories = []

        for directory_name in sorted(
            directory_names
        ):
            if (
                directory_name.lower()
                in DENIED_DIRECTORY_NAMES
            ):
                continue

            directory_path = (
                current_path
                / directory_name
            )

            if directory_path.is_symlink():
                continue

            if not _entry_is_discoverable(
                root,
                directory_path,
            ):
                continue

            allowed_directories.append(
                directory_name
            )

        directory_names[:] = (
            allowed_directories
        )

        for file_name in sorted(
            file_names
        ):
            file_path = (
                current_path
                / file_name
            )

            if file_path.is_symlink():
                continue

            if not _entry_is_discoverable(
                root,
                file_path,
            ):
                continue

            try:
                relative_file = (
                    file_path
                    .relative_to(root)
                    .as_posix()
                )

            except ValueError:
                continue

            (
                readable,
                _policy_error,
            ) = validate_readable_text_path(
                relative_file
            )

            if not readable:
                continue

            try:
                size_bytes = (
                    file_path.stat()
                    .st_size
                )

            except OSError:
                continue

            if (
                size_bytes
                > MAX_SEARCH_FILE_BYTES
            ):
                continue

            files_scanned += 1

            try:
                with file_path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:
                    for (
                        line_number,
                        line,
                    ) in enumerate(
                        handle,
                        start=1,
                    ):
                        if (
                            normalized_query
                            not in line.casefold()
                        ):
                            continue

                        display_line = (
                            line.rstrip(
                                "\r\n"
                            )
                        )

                        if (
                            len(display_line)
                            > MAX_MATCH_LINE_CHARS
                        ):
                            display_line = (
                                display_line[
                                    :MAX_MATCH_LINE_CHARS
                                ]
                                + "...[truncated]"
                            )

                        matches.append(
                            {
                                "path":
                                    relative_file,

                                "line_number":
                                    line_number,

                                "line":
                                    display_line,
                            }
                        )

                        if (
                            len(matches)
                            >= MAX_SEARCH_RESULTS
                        ):
                            truncated = True
                            break

            except (
                UnicodeDecodeError,
                OSError,
            ):
                continue

            if truncated:
                break

        if truncated:
            break

    return {
        "ok": True,
        "status": "success",

        "repository":
            repository_name,

        "query":
            query,

        "path":
            _normalized_relative_path(
                root,
                candidate,
            ),

        "matches":
            matches,

        "files_scanned":
            files_scanned,

        "truncated":
            truncated,

        "error":
            None,
    }


def workspace_file_info(
    relative_path: str,
    repository: (
        str
        | None
    ) = None,
) -> dict[str, Any]:
    """
    Return bounded metadata for one workspace path.

    File contents are not returned.
    """

    try:
        (
            root,
            candidate,
            repository_name,
        ) = _resolve_workspace_path(
            relative_path,
            repository=(
                repository
            ),
        )

    except WorkspacePathError as exc:
        return {
            "ok": False,
            "status": "denied",
            "error": str(
                exc
            ),
        }

    if not candidate.exists():
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Requested workspace path "
                "does not exist."
            ),
        }

    try:
        stat_result = (
            candidate.lstat()
        )

    except OSError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": (
                "Workspace metadata lookup failed: "
                f"{exc}"
            ),
        }

    if candidate.is_symlink():
        entry_type = (
            "symlink"
        )

    elif candidate.is_dir():
        entry_type = (
            "directory"
        )

    elif candidate.is_file():
        entry_type = (
            "file"
        )

    else:
        entry_type = (
            "other"
        )

    normalized_path = (
        _normalized_relative_path(
            root,
            candidate,
        )
    )

    readable_text = False

    if entry_type == "file":
        (
            readable_text,
            _policy_error,
        ) = validate_readable_text_path(
            normalized_path
        )

    return {
        "ok": True,
        "status": "success",

        "repository":
            repository_name,

        "path":
            normalized_path,

        "name":
            candidate.name,

        "type":
            entry_type,

        "size_bytes":
            stat_result.st_size,

        "suffix":
            candidate.suffix.lower(),

        "readable_text":
            readable_text,

        "error":
            None,
    }
