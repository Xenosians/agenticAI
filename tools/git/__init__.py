from __future__ import annotations

from typing import (
    Any,
)

from services.git_repositories import (
    GitRepositoryTarget,
    resolve_git_repository,
)

from services.process_runner import (
    run_process,
)


DEFAULT_TIMEOUT_SECONDS = 10

OUTPUT_TRUNCATION_MARKER = (
    "...[output truncated]"
)


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

    return (
        value
    )


def _stdout_lines(
    result: dict[
        str,
        Any,
    ],
) -> list[str]:

    return [
        line

        for line
        in _stdout(
            result
        ).splitlines()

        if line.strip()
    ]


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

    return (
        run_process(
            executable="git",

            args=args,

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
                    ahead = int(
                        component[
                            6:
                        ]
                    )

                except ValueError:
                    ahead = 0

            elif component.startswith(
                "behind "
            ):
                try:
                    behind = int(
                        component[
                            7:
                        ]
                    )

                except ValueError:
                    behind = 0

    return (
        branch or None,
        upstream or None,
        ahead,
        behind,
    )


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
    Inspect branch and working-tree state for one configured
    logical repository.

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
            target=target,

            args=[
                "status",
                "--short",
                "--branch",
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

                result=result,
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

    changes = []

    for line in change_lines:
        if len(
            line
        ) >= 3:
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

        else:
            code = (
                line.strip()
            )

            path = ""

        changes.append(
            {
                "code":
                    code,

                "path":
                    path,
            }
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
    }


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
            target=target,

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

                result=result,
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
    }


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
            target=target,

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

                result=result,
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
    }


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
    Return the bounded current unstaged diff for one configured
    repository.
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
            target=target,

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

                result=result,
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

        "diff":
            diff.strip(),

        "has_changes":
            bool(
                diff.strip()
            ),

        "truncated":
            truncated,
    }


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
    Return unstaged changed filenames from one configured
    repository.
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
            target=target,

            args=[
                "diff",
                "--no-ext-diff",
                "--name-only",
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

                result=result,
            )
        )

    files = (
        _stdout_lines(
            result
        )
    )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "files":
            files,

        "count":
            len(
                files
            ),
    }