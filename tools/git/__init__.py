from __future__ import annotations

from typing import (
    Any,
)

from services.git_repositories import (
    GitRepositoryTarget,
    resolve_git_repository,
)

from services.process_runner import (
    run_trusted_process,
)


DEFAULT_TIMEOUT_SECONDS = 10

OUTPUT_TRUNCATION_MARKER = (
    "...[output truncated]"
)


CONFLICT_STATUS_CODES = {
    "DD",
    "AU",
    "UD",
    "UA",
    "DU",
    "AA",
    "UU",
}


# ============================================================
# REPOSITORY RESOLUTION
# ============================================================


def _resolve_target(
    repository: (
        str | None
    ),
) -> tuple[
    GitRepositoryTarget | None,
    dict[
        str,
        Any,
    ] | None,
]:

    try:

        target = (
            resolve_git_repository(
                repository
            )
        )

    except ValueError as exc:

        return (
            None,

            {
                "ok":
                    False,

                "status":
                    "denied",

                "repository":
                    repository,

                "error":
                    str(
                        exc
                    ),
            },
        )

    return (
        target,
        None,
    )


# ============================================================
# PROCESS HELPERS
# ============================================================


def _process_failure(
    *,
    repository: str,
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    error = (
        result.get(
            "error"
        )
    )

    if not isinstance(
        error,
        str,
    ):

        stderr = (
            result.get(
                "stderr"
            )
        )

        if (
            isinstance(
                stderr,
                str,
            )
            and stderr.strip()
        ):

            error = (
                stderr.strip()
            )

    if not isinstance(
        error,
        str,
    ):

        error = (
            "Git operation failed."
        )

    return {
        "ok":
            False,

        "status":
            str(
                result.get(
                    "status",
                    "error",
                )
            ),

        "repository":
            repository,

        "error":
            error,
    }


def _stdout(
    result: dict[
        str,
        Any,
    ],
) -> str:

    value = (
        result.get(
            "stdout"
        )
    )

    if not isinstance(
        value,
        str,
    ):

        return ""

    return value


def _stdout_lines(
    result: dict[
        str,
        Any,
    ],
) -> list[str]:

    lines = []

    for line in (
        _stdout(
            result
        )
        .splitlines()
    ):

        if not line.strip():
            continue

        if (
            line.strip()
            == OUTPUT_TRUNCATION_MARKER
        ):

            continue

        lines.append(
            line
        )

    return lines


def _is_truncated(
    value: str,
) -> bool:

    return (
        OUTPUT_TRUNCATION_MARKER
        in value
    )


def _run_git(
    *,
    target: GitRepositoryTarget,
    args: list[str],
    timeout_seconds: int,
) -> dict[
    str,
    Any,
]:
    """
    Execute one Git command whose complete command shape is owned by
    trusted application code.

    This intentionally uses run_trusted_process rather than the
    generic model-facing process_exec policy.

    The model may select only the bounded logical repository ID.

    The model does NOT select:
        executable
        Git subcommand
        Git flags
        cwd
    """

    return (
        run_trusted_process(
            executable="git",

            args=(
                args
            ),

            cwd=(
                str(
                    target.path
                )
            ),

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )


# ============================================================
# STATUS PARSING
# ============================================================


def _parse_branch_header(
    value: str,
) -> tuple[
    str | None,
    str | None,
    int,
    int,
]:

    line = (
        value.strip()
    )

    if line.startswith(
        "## "
    ):

        line = (
            line[
                3:
            ]
            .strip()
        )

    if not line:

        return (
            None,
            None,
            0,
            0,
        )

    # Fresh repository with no commits yet.
    if line.startswith(
        "No commits yet on "
    ):

        return (
            line[
                len(
                    "No commits yet on "
                ):
            ].strip()
            or None,

            None,
            0,
            0,
        )

    if line.startswith(
        "Initial commit on "
    ):

        return (
            line[
                len(
                    "Initial commit on "
                ):
            ].strip()
            or None,

            None,
            0,
            0,
        )

    ahead = 0
    behind = 0

    tracking_part = None

    if " [" in line:

        line, tracking_part = (
            line.split(
                " [",
                1,
            )
        )

        tracking_part = (
            tracking_part
            .rstrip("]")
        )

    upstream = None

    if "..." in line:

        branch, upstream = (
            line.split(
                "...",
                1,
            )
        )

        branch = (
            branch.strip()
        )

        upstream = (
            upstream.strip()
        )

    else:

        branch = (
            line.strip()
        )

    if tracking_part:

        for component in (
            tracking_part.split(
                ","
            )
        ):

            component = (
                component.strip()
            )

            if component.startswith(
                "ahead "
            ):

                try:

                    ahead = (
                        int(
                            component[
                                6:
                            ]
                        )
                    )

                except ValueError:

                    ahead = 0

            elif component.startswith(
                "behind "
            ):

                try:

                    behind = (
                        int(
                            component[
                                7:
                            ]
                        )
                    )

                except ValueError:

                    behind = 0

    return (
        branch
        or None,

        upstream
        or None,

        ahead,

        behind,
    )


def _parse_status_change(
    line: str,
) -> dict[
    str,
    Any,
] | None:
    """
    Parse one Git porcelain-v1 short-status line.

    XY semantics:

        X = index / staged state
        Y = work-tree / unstaged state

    Special cases:

        ?? = untracked
        !! = ignored
    """

    if len(
        line
    ) < 3:

        return None

    code = (
        line[
            :2
        ]
    )

    path = (
        line[
            3:
        ]
        .strip()
    )

    if not path:

        return None

    untracked = (
        code
        == "??"
    )

    ignored = (
        code
        == "!!"
    )

    conflicted = (
        code
        in CONFLICT_STATUS_CODES
    )

    staged = False
    unstaged = False

    if not (
        untracked
        or ignored
        or conflicted
    ):

        staged = (
            code[
                0
            ]
            != " "
        )

        unstaged = (
            code[
                1
            ]
            != " "
        )

    return {
        "code":
            code,

        "path":
            path,

        "staged":
            staged,

        "unstaged":
            unstaged,

        "untracked":
            untracked,

        "conflicted":
            conflicted,
    }


def _parse_status_changes(
    lines: list[str],
) -> list[
    dict[
        str,
        Any,
    ]
]:

    changes: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for line in lines:

        parsed = (
            _parse_status_change(
                line
            )
        )

        if parsed is None:

            continue

        changes.append(
            parsed
        )

    return changes


def _unique_paths(
    changes: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    flag: (
        str
        | None
    ) = None,
) -> list[str]:

    paths: list[str] = []

    seen: set[str] = set()

    for change in changes:

        if (
            flag is not None
            and change.get(
                flag
            )
            is not True
        ):

            continue

        path = (
            change.get(
                "path"
            )
        )

        if not isinstance(
            path,
            str,
        ):

            continue

        if path in seen:

            continue

        seen.add(
            path
        )

        paths.append(
            path
        )

    return paths


def _working_tree_summary(
    changes: list[
        dict[
            str,
            Any,
        ]
    ],
) -> dict[
    str,
    Any,
]:

    files = (
        _unique_paths(
            changes
        )
    )

    staged_files = (
        _unique_paths(
            changes,
            flag="staged",
        )
    )

    unstaged_files = (
        _unique_paths(
            changes,
            flag="unstaged",
        )
    )

    untracked_files = (
        _unique_paths(
            changes,
            flag="untracked",
        )
    )

    conflicted_files = (
        _unique_paths(
            changes,
            flag="conflicted",
        )
    )

    return {
        "files":
            files,

        "count":
            len(
                files
            ),

        "staged_files":
            staged_files,

        "staged_count":
            len(
                staged_files
            ),

        "unstaged_files":
            unstaged_files,

        "unstaged_count":
            len(
                unstaged_files
            ),

        "untracked_files":
            untracked_files,

        "untracked_count":
            len(
                untracked_files
            ),

        "conflicted_files":
            conflicted_files,

        "conflicted_count":
            len(
                conflicted_files
            ),
    }


# ============================================================
# STATUS
# ============================================================


def workspace_git_status(
    repository: (
        str | None
    ) = None,

    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Inspect branch and complete working-tree state for one
    configured logical repository.

    Includes:

        staged changes
        unstaged changes
        untracked files
        merge conflicts

    Repository selection is logical and trusted.

    Git command shape is fixed by application code.
    """

    (
        target,
        target_error,
    ) = (
        _resolve_target(
            repository
        )
    )

    if target is None:

        return (
            target_error
            or {
                "ok":
                    False,

                "status":
                    "denied",

                "error":
                    "Repository resolution failed.",
            }
        )

    result = (
        _run_git(
            target=(
                target
            ),

            args=[
                "status",
                "--short",
                "--branch",
                "--untracked-files=all",
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not (
        result.get(
            "ok",
            False,
        )
    ):

        return (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    result
                ),
            )
        )

    raw_stdout = (
        _stdout(
            result
        )
    )

    lines = (
        _stdout_lines(
            result
        )
    )

    branch = None
    upstream = None
    ahead = 0
    behind = 0

    change_lines = (
        lines
    )

    if (
        lines
        and lines[
            0
        ].startswith(
            "## "
        )
    ):

        (
            branch,
            upstream,
            ahead,
            behind,
        ) = (
            _parse_branch_header(
                lines[
                    0
                ]
            )
        )

        change_lines = (
            lines[
                1:
            ]
        )

    changes = (
        _parse_status_changes(
            change_lines
        )
    )

    summary = (
        _working_tree_summary(
            changes
        )
    )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "branch":
            branch,

        "upstream":
            upstream,

        "ahead":
            ahead,

        "behind":
            behind,

        "clean":
            len(
                changes
            )
            == 0,

        "changes":
            changes,

        "change_count":
            len(
                changes
            ),

        **summary,

        "truncated":
            _is_truncated(
                raw_stdout
            ),
    }


# ============================================================
# BRANCHES
# ============================================================


def workspace_git_branches(
    repository: (
        str | None
    ) = None,

    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    List local branches for one configured repository.
    """

    (
        target,
        target_error,
    ) = (
        _resolve_target(
            repository
        )
    )

    if target is None:

        return (
            target_error
            or {
                "ok":
                    False,

                "status":
                    "denied",

                "error":
                    "Repository resolution failed.",
            }
        )

    result = (
        _run_git(
            target=(
                target
            ),

            args=[
                "branch",
                "--list",
                "--no-color",
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not (
        result.get(
            "ok",
            False,
        )
    ):

        return (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    result
                ),
            )
        )

    branches = []

    current_branch = None

    for line in (
        _stdout_lines(
            result
        )
    ):

        stripped = (
            line.strip()
        )

        current = (
            line.startswith(
                "*"
            )
        )

        if current:

            name = (
                stripped[
                    1:
                ]
                .strip()
            )

        else:

            name = (
                stripped
            )

        if not name:

            continue

        branches.append(
            {
                "name":
                    name,

                "current":
                    current,
            }
        )

        if current:

            current_branch = (
                name
            )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "current_branch":
            current_branch,

        "branches":
            branches,

        "count":
            len(
                branches
            ),

        "truncated":
            _is_truncated(
                _stdout(
                    result
                )
            ),
    }


# ============================================================
# LOG
# ============================================================


def workspace_git_log(
    repository: (
        str | None
    ) = None,

    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Return the latest twenty commits from one configured
    repository.

    History size and command shape are trusted constants.
    """

    (
        target,
        target_error,
    ) = (
        _resolve_target(
            repository
        )
    )

    if target is None:

        return (
            target_error
            or {
                "ok":
                    False,

                "status":
                    "denied",

                "error":
                    "Repository resolution failed.",
            }
        )

    result = (
        _run_git(
            target=(
                target
            ),

            args=[
                "log",
                "--oneline",
                "--no-decorate",
                "-n",
                "20",
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not (
        result.get(
            "ok",
            False,
        )
    ):

        return (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    result
                ),
            )
        )

    commits = []

    for line in (
        _stdout_lines(
            result
        )
    ):

        pieces = (
            line.split(
                " ",
                1,
            )
        )

        commit_hash = (
            pieces[
                0
            ]
        )

        message = (
            pieces[
                1
            ]
            if len(
                pieces
            )
            > 1
            else ""
        )

        commits.append(
            {
                "hash":
                    commit_hash,

                "message":
                    message,
            }
        )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "commits":
            commits,

        "count":
            len(
                commits
            ),

        "truncated":
            _is_truncated(
                _stdout(
                    result
                )
            ),
    }


# ============================================================
# DIFF
# ============================================================


def workspace_git_diff(
    repository: (
        str | None
    ) = None,

    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Return the bounded current UNSTAGED diff for one configured
    repository.

    Staged changes are intentionally not included by this
    capability.
    """

    (
        target,
        target_error,
    ) = (
        _resolve_target(
            repository
        )
    )

    if target is None:

        return (
            target_error
            or {
                "ok":
                    False,

                "status":
                    "denied",

                "error":
                    "Repository resolution failed.",
            }
        )

    result = (
        _run_git(
            target=(
                target
            ),

            args=[
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--unified=3",
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not (
        result.get(
            "ok",
            False,
        )
    ):

        return (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    result
                ),
            )
        )

    diff = (
        _stdout(
            result
        )
    )

    truncated = (
        _is_truncated(
            diff
        )
    )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "scope":
            "unstaged",

        "diff":
            diff.strip(),

        "has_changes":
            bool(
                diff.strip()
            ),

        "truncated":
            truncated,
    }


# ============================================================
# CHANGED FILES
# ============================================================


def workspace_git_changed_files(
    repository: (
        str | None
    ) = None,

    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Return all current changed files from one configured repository.

    Includes:
        staged
        unstaged
        untracked
        conflicted

    This capability reports working-tree state rather than only
    `git diff --name-only`.
    """

    (
        target,
        target_error,
    ) = (
        _resolve_target(
            repository
        )
    )

    if target is None:

        return (
            target_error
            or {
                "ok":
                    False,

                "status":
                    "denied",

                "error":
                    "Repository resolution failed.",
            }
        )

    result = (
        _run_git(
            target=(
                target
            ),

            args=[
                "status",
                "--short",
                "--untracked-files=all",
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not (
        result.get(
            "ok",
            False,
        )
    ):

        return (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    result
                ),
            )
        )

    raw_stdout = (
        _stdout(
            result
        )
    )

    changes = (
        _parse_status_changes(
            _stdout_lines(
                result
            )
        )
    )

    summary = (
        _working_tree_summary(
            changes
        )
    )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "scope":
            "working_tree",

        "changes":
            changes,

        **summary,

        "truncated":
            _is_truncated(
                raw_stdout
            ),
    }