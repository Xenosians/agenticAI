from __future__ import annotations

import re

from pathlib import (
    Path,
)

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

MAX_STAGE_PATHS = 50
MAX_GIT_PATH_CHARS = 512

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


WINDOWS_DRIVE_PREFIX = re.compile(
    r"^[A-Za-z]:"
)


# ============================================================
# REPOSITORY RESOLUTION
# ============================================================


def _resolve_target(
    repository: str | None,
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

    lines: list[str] = []

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

    if line.startswith(
        "No commits yet on "
    ):

        branch = (
            line[
                len(
                    "No commits yet on "
                ):
            ]
            .strip()
        )

        return (
            branch or None,
            None,
            0,
            0,
        )

    if line.startswith(
        "Initial commit on "
    ):

        branch = (
            line[
                len(
                    "Initial commit on "
                ):
            ]
            .strip()
        )

        return (
            branch or None,
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
            tracking_part.rstrip(
                "]"
            )
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
        branch or None,
        upstream or None,
        ahead,
        behind,
    )


def _parse_status_change(
    line: str,
) -> dict[
    str,
    Any,
] | None:

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

        if parsed is not None:

            changes.append(
                parsed
            )

    return (
        changes
    )


def _unique_paths(
    changes: list[
        dict[
            str,
            Any,
        ]
    ],
    *,
    flag: str | None = None,
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

    return (
        paths
    )


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
    repository: str | None = None,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:

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

    if not result.get(
        "ok",
        False,
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
    repository: str | None = None,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:

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

    if not result.get(
        "ok",
        False,
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
    repository: str | None = None,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:

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

    if not result.get(
        "ok",
        False,
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

        commits.append(
            {
                "hash":
                    pieces[
                        0
                    ],

                "message": (
                    pieces[
                        1
                    ]
                    if len(
                        pieces
                    )
                    > 1
                    else ""
                ),
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


def _workspace_git_diff(
    *,
    repository: str | None,
    timeout_seconds: int,
    staged: bool,
) -> dict[
    str,
    Any,
]:

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

    if staged:

        args = [
            "diff",
            "--cached",
            "--no-ext-diff",
            "--no-color",
            "--unified=3",
        ]

        scope = (
            "staged"
        )

    else:

        args = [
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--unified=3",
        ]

        scope = (
            "unstaged"
        )

    result = (
        _run_git(
            target=(
                target
            ),

            args=(
                args
            ),

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not result.get(
        "ok",
        False,
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

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "scope":
            scope,

        "diff":
            diff.strip(),

        "has_changes":
            bool(
                diff.strip()
            ),

        "truncated":
            _is_truncated(
                diff
            ),
    }


def workspace_git_diff(
    repository: str | None = None,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:

    return (
        _workspace_git_diff(
            repository=(
                repository
            ),

            timeout_seconds=(
                timeout_seconds
            ),

            staged=False,
        )
    )


def workspace_git_staged_diff(
    repository: str | None = None,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:

    return (
        _workspace_git_diff(
            repository=(
                repository
            ),

            timeout_seconds=(
                timeout_seconds
            ),

            staged=True,
        )
    )


# ============================================================
# CHANGED FILES
# ============================================================


def workspace_git_changed_files(
    repository: str | None = None,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:

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

    if not result.get(
        "ok",
        False,
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

        **_working_tree_summary(
            changes
        ),

        "truncated":
            _is_truncated(
                raw_stdout
            ),
    }


# ============================================================
# STAGE FILE POLICY
# ============================================================


def _validate_stage_paths(
    paths: Any,
) -> tuple[
    list[str] | None,
    str | None,
]:

    if not isinstance(
        paths,
        list,
    ):

        return (
            None,
            "paths must be a list of repository-relative files.",
        )

    if not paths:

        return (
            None,
            "At least one file path must be supplied.",
        )

    if len(
        paths
    ) > MAX_STAGE_PATHS:

        return (
            None,
            (
                "Too many Git paths were requested. "
                f"The maximum is {MAX_STAGE_PATHS}."
            ),
        )

    normalized: list[str] = []

    seen: set[str] = set()

    for path in paths:

        if not isinstance(
            path,
            str,
        ):

            return (
                None,
                "Every Git path must be a string.",
            )

        if (
            not path
            or path
            != path.strip()
        ):

            return (
                None,
                (
                    "Every Git path must be a non-empty "
                    "exact repository-relative path."
                ),
            )

        if len(
            path
        ) > MAX_GIT_PATH_CHARS:

            return (
                None,
                (
                    "A requested Git path exceeds the "
                    "maximum supported length."
                ),
            )

        if any(
            ord(
                character
            ) < 32
            or ord(
                character
            ) == 127

            for character
            in path
        ):

            return (
                None,
                (
                    "Git paths containing control "
                    "characters are not permitted."
                ),
            )

        if (
            path.startswith(
                "/"
            )
            or WINDOWS_DRIVE_PREFIX.match(
                path
            )
        ):

            return (
                None,
                (
                    "Git paths must be repository-relative, "
                    "not absolute."
                ),
            )

        # Repository paths are represented canonically with "/".
        # Backslashes are rejected rather than guessed/translated.
        if "\\" in path:

            return (
                None,
                (
                    "Git paths must use repository-relative "
                    "forward-slash notation."
                ),
            )

        parts = (
            path.split(
                "/"
            )
        )

        if any(
            part
            in {
                "",
                ".",
                "..",
            }

            for part
            in parts
        ):

            return (
                None,
                (
                    "Git paths may not contain empty, '.', "
                    "or '..' path components."
                ),
            )

        if (
            parts[
                0
            ]
            == ".git"
        ):

            return (
                None,
                (
                    "Git internal metadata paths "
                    "cannot be staged."
                ),
            )

        if path in seen:

            return (
                None,
                (
                    "Duplicate Git paths are "
                    "not permitted."
                ),
            )

        seen.add(
            path
        )

        normalized.append(
            path
        )

    return (
        normalized,
        None,
    )


def _validate_stage_request(
    *,
    repository: Any,
    paths: Any,
) -> tuple[
    GitRepositoryTarget | None,
    list[str] | None,
    str | None,
]:

    if not isinstance(
        repository,
        str,
    ):

        return (
            None,
            None,
            (
                "An explicit logical repository "
                "identifier is required."
            ),
        )

    if (
        not repository
        or repository
        != repository.strip()
    ):

        return (
            None,
            None,
            (
                "The repository identifier must "
                "be a non-empty exact value."
            ),
        )

    try:

        target = (
            resolve_git_repository(
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

    (
        normalized_paths,
        path_error,
    ) = (
        _validate_stage_paths(
            paths
        )
    )

    if normalized_paths is None:

        return (
            None,
            None,
            path_error,
        )

    repository_root = (
        target.path.resolve()
    )

    for relative_path in (
        normalized_paths
    ):

        candidate = (
            repository_root
            / Path(
                *relative_path.split(
                    "/"
                )
            )
        ).resolve(
            strict=False
        )

        if (
            candidate
            != repository_root
            and repository_root
            not in candidate.parents
        ):

            return (
                None,
                None,
                (
                    "A requested Git path escapes "
                    "the selected repository."
                ),
            )

        # `git add directory` recursively stages a whole subtree.
        #
        # workspace_git_stage_files deliberately does not have that
        # authority.
        if candidate.is_dir():

            return (
                None,
                None,
                (
                    f"'{relative_path}' is a directory. "
                    "This capability stages explicit files only."
                ),
            )

    return (
        target,
        normalized_paths,
        None,
    )


def evaluate_git_stage_policy(
    repository: Any,
    paths: Any,
) -> dict[
    str,
    Any,
]:
    """
    Trusted pre-approval policy.

    This performs no mutation.

    It ensures the approval is created only for:
        - an explicitly selected configured repository
        - bounded exact repository-relative file paths
    """

    (
        target,
        normalized_paths,
        error,
    ) = (
        _validate_stage_request(
            repository=(
                repository
            ),

            paths=(
                paths
            ),
        )
    )

    if (
        target is None
        or normalized_paths is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "error":
                (
                    error
                    or "Git staging policy denied the request."
                ),
        }

    return {
        "ok":
            True,

        "status":
            "allowed",

        "risk":
            "low",

        "requires_approval":
            True,
    }


# ============================================================
# STAGE FILES
# ============================================================


def workspace_git_stage_files(
    repository: str,
    paths: list[str],
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Stage exactly the approved repository-relative file paths.

    The persisted approval arguments are revalidated here because
    ApprovalManager executes the approved MCP call directly.

    Git pathspec magic is disabled using :(literal).
    """

    (
        target,
        normalized_paths,
        error,
    ) = (
        _validate_stage_request(
            repository=(
                repository
            ),

            paths=(
                paths
            ),
        )
    )

    if (
        target is None
        or normalized_paths is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "repository":
                repository,

            "error":
                (
                    error
                    or "Git staging request was denied."
                ),
        }

    literal_pathspecs = [
        f":(literal){path}"

        for path
        in normalized_paths
    ]

    mutation = (
        _run_git(
            target=(
                target
            ),

            args=[
                "add",
                "--",
                *literal_pathspecs,
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not mutation.get(
        "ok",
        False,
    ):

        failure = (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    mutation
                ),
            )
        )

        failure[
            "requested_paths"
        ] = (
            normalized_paths
        )

        return (
            failure
        )

    # ========================================================
    # TRUSTED POST-MUTATION VERIFICATION
    #
    # If verification itself fails, the mutation still completed
    # successfully because `git add` returned success.
    #
    # Do not falsely report the side effect as failed.
    # ========================================================

    verification = (
        _run_git(
            target=(
                target
            ),

            args=[
                "diff",
                "--cached",
                "--name-only",
                "--",
                *literal_pathspecs,
            ],

            timeout_seconds=(
                timeout_seconds
            ),
        )
    )

    if not verification.get(
        "ok",
        False,
    ):

        verification_failure = (
            _process_failure(
                repository=(
                    target.name
                ),

                result=(
                    verification
                ),
            )
        )

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_paths":
                normalized_paths,

            "files":
                [],

            "staged_paths":
                [],

            "staged_count":
                0,

            "verification_ok":
                False,

            "verification_error":
                verification_failure.get(
                    "error"
                ),
        }

    staged_paths = (
        _stdout_lines(
            verification
        )
    )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "requested_paths":
            normalized_paths,

        # `files` keeps the result compatible with the generic Git
        # learning-context extractor.
        "files":
            staged_paths,

        "staged_paths":
            staged_paths,

        "staged_count":
            len(
                staged_paths
            ),

        "verification_ok":
            True,

        "verification_error":
            None,
    }

# ============================================================
# GOVERNED GIT UNSTAGE
# ============================================================


def _validate_unstage_request(
    repository: Any,
    paths: Any,
):
    """
    Reuse the exact bounded repository/path validation used by
    governed Git staging.

    Staging and unstaging differ in their Git side effect, not in
    the authority granted over repository identity or file paths.
    """

    return (
        _validate_stage_request(
            repository=repository,
            paths=paths,
        )
    )


def evaluate_git_unstage_policy(
    repository: Any,
    paths: Any,
) -> dict[
    str,
    Any,
]:
    """
    Trusted pre-approval policy.

    This performs no mutation.

    Approval is created only for an explicitly selected configured
    repository and bounded exact repository-relative file paths.
    """

    (
        target,
        normalized_paths,
        error,
    ) = (
        _validate_unstage_request(
            repository=repository,
            paths=paths,
        )
    )

    if (
        target is None
        or normalized_paths is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "error":
                (
                    error
                    or "Git unstaging policy denied the request."
                ),
        }

    return {
        "ok":
            True,

        "status":
            "allowed",

        "risk":
            "low",

        "requires_approval":
            True,
    }


def workspace_git_unstage_files(
    repository: str,
    paths: list[str],
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Remove exactly the approved repository-relative file paths
    from the Git index without modifying working-tree contents.

    The persisted approval arguments are revalidated here because
    ApprovalManager executes the approved MCP call directly.

    Trusted command:

        git reset -- :(literal)<path> ...

    This is the path-scoped index-reset form only.

    It does NOT:
        - reset working-tree contents
        - use --hard
        - delete files
        - switch branches
        - create commits
        - contact remotes

    Git pathspec magic is disabled using :(literal).
    """

    (
        target,
        normalized_paths,
        error,
    ) = (
        _validate_unstage_request(
            repository=repository,
            paths=paths,
        )
    )

    if (
        target is None
        or normalized_paths is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "repository":
                repository,

            "error":
                (
                    error
                    or "Git unstaging request was denied."
                ),
        }

    literal_pathspecs = [
        f":(literal){path}"

        for path
        in normalized_paths
    ]

    # ========================================================
    # TRUSTED PRE-MUTATION SNAPSHOT
    #
    # Record which of the explicitly requested paths are actually
    # staged immediately before the mutation.
    #
    # This gives us an exact postcondition comparison and avoids
    # claiming that an already-unstaged path was changed.
    # ========================================================

    before = (
        _run_git(
            target=target,

            args=[
                "diff",
                "--cached",
                "--name-only",
                "--",
                *literal_pathspecs,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not before.get(
        "ok",
        False,
    ):

        failure = (
            _process_failure(
                repository=target.name,
                result=before,
            )
        )

        failure[
            "requested_paths"
        ] = (
            normalized_paths
        )

        return (
            failure
        )

    staged_before_paths = (
        _stdout_lines(
            before
        )
    )

    # Nothing among the exact requested paths is currently staged.
    # This is a successful no-op, not a mutation failure.
    if not staged_before_paths:

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_paths":
                normalized_paths,

            "files":
                [],

            "staged_before_paths":
                [],

            "unstaged_paths":
                [],

            "unstaged_count":
                0,

            "remaining_staged_paths":
                [],

            "mutation_performed":
                False,

            "verification_ok":
                True,

            "verification_error":
                None,
        }

    # ========================================================
    # TRUSTED INDEX-ONLY MUTATION
    # ========================================================

    mutation = (
        _run_git(
            target=target,

            args=[
                "reset",
                "--",
                *literal_pathspecs,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not mutation.get(
        "ok",
        False,
    ):

        failure = (
            _process_failure(
                repository=target.name,
                result=mutation,
            )
        )

        failure[
            "requested_paths"
        ] = (
            normalized_paths
        )

        failure[
            "staged_before_paths"
        ] = (
            staged_before_paths
        )

        return (
            failure
        )

    # ========================================================
    # TRUSTED POST-MUTATION VERIFICATION
    #
    # If this verification fails, the index mutation has already
    # happened because the path-scoped reset returned success.
    #
    # Do not falsely report that no side effect occurred.
    # ========================================================

    verification = (
        _run_git(
            target=target,

            args=[
                "diff",
                "--cached",
                "--name-only",
                "--",
                *literal_pathspecs,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not verification.get(
        "ok",
        False,
    ):

        verification_failure = (
            _process_failure(
                repository=target.name,
                result=verification,
            )
        )

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_paths":
                normalized_paths,

            "files":
                [],

            "staged_before_paths":
                staged_before_paths,

            "unstaged_paths":
                [],

            "unstaged_count":
                0,

            "remaining_staged_paths":
                [],

            "mutation_performed":
                True,

            "verification_ok":
                False,

            "verification_error":
                verification_failure.get(
                    "error"
                ),
        }

    remaining_staged_paths = (
        _stdout_lines(
            verification
        )
    )

    remaining = set(
        remaining_staged_paths
    )

    unstaged_paths = [
        path

        for path
        in staged_before_paths

        if path not in remaining
    ]

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "requested_paths":
            normalized_paths,

        # Keep this compatible with the generic Git learning-context
        # extractor. These are the paths whose staged state actually
        # changed during this execution.
        "files":
            unstaged_paths,

        "staged_before_paths":
            staged_before_paths,

        "unstaged_paths":
            unstaged_paths,

        "unstaged_count":
            len(
                unstaged_paths
            ),

        "remaining_staged_paths":
            remaining_staged_paths,

        "mutation_performed":
            True,

        "verification_ok":
            True,

        "verification_error":
            None,
    }


# ============================================================
# GOVERNED GIT BRANCH CREATION
# ============================================================

MAX_GIT_BRANCH_CHARS = 255


def _validate_create_branch_request(
    repository: Any,
    branch_name: Any,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
):
    """
    Validate one exact local branch-creation request.

    This performs no mutation.

    Branch syntax is delegated to Git's trusted ref validator.
    The user/model cannot select the validation command shape.

    The branch will be created from the repository's current HEAD.
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

        error = (
            target_error.get(
                "error"
            )
            if isinstance(
                target_error,
                dict,
            )
            else None
        )

        return (
            None,
            None,
            None,
            (
                error
                or "Git repository resolution failed."
            ),
        )

    if not isinstance(
        branch_name,
        str,
    ):

        return (
            None,
            None,
            None,
            "Git branch name must be a string.",
        )

    if not branch_name:

        return (
            None,
            None,
            None,
            "Git branch name cannot be empty.",
        )

    # Do not silently normalize user/model input.
    if (
        branch_name
        != branch_name.strip()
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch name must not contain "
                "leading or trailing whitespace."
            ),
        )

    if (
        len(
            branch_name
        )
        > MAX_GIT_BRANCH_CHARS
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch name exceeds the "
                f"{MAX_GIT_BRANCH_CHARS}-character limit."
            ),
        )

    if branch_name.startswith(
        "-"
    ):

        return (
            None,
            None,
            None,
            "Git branch name must not begin with '-'.",
        )

    if "@{" in branch_name:

        return (
            None,
            None,
            None,
            (
                "Git branch name must not contain "
                "reflog-selection syntax."
            ),
        )

    if any(
        ord(character) < 32
        or ord(character) == 127

        for character
        in branch_name
    ):

        return (
            None,
            None,
            None,
            "Git branch name contains control characters.",
        )

    # ========================================================
    # TRUSTED GIT BRANCH-NAME VALIDATION
    #
    # --branch applies Git's real branch-name rules.
    #
    # We additionally require Git to return the exact supplied
    # value so special branch expressions cannot be expanded into
    # some different branch name.
    # ========================================================

    branch_validation = (
        _run_git(
            target=target,

            args=[
                "check-ref-format",
                "--branch",
                branch_name,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not branch_validation.get(
        "ok",
        False,
    ):

        return (
            None,
            None,
            None,
            "Git branch name is not valid.",
        )

    validated_lines = (
        _stdout_lines(
            branch_validation
        )
    )

    if (
        len(
            validated_lines
        )
        != 1
        or validated_lines[
            0
        ]
        != branch_name
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch name must resolve to exactly "
                "the explicitly requested value."
            ),
        )

    # ========================================================
    # CURRENT START POINT
    #
    # Branch creation is intentionally bounded to current HEAD.
    # The model cannot choose an arbitrary commit.
    # ========================================================

    head_result = (
        _run_git(
            target=target,

            args=[
                "rev-parse",
                "--verify",
                "HEAD",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not head_result.get(
        "ok",
        False,
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch creation requires a repository "
                "with a committed HEAD."
            ),
        )

    head_lines = (
        _stdout_lines(
            head_result
        )
    )

    if (
        len(
            head_lines
        )
        != 1
    ):

        return (
            None,
            None,
            None,
            "Could not resolve the current Git HEAD.",
        )

    base_commit = (
        head_lines[
            0
        ].strip()
    )

    if not base_commit:

        return (
            None,
            None,
            None,
            "Could not resolve the current Git HEAD.",
        )

    ref_name = (
        f"refs/heads/{branch_name}"
    )

    # ========================================================
    # EXISTENCE CHECK
    #
    # show-ref:
    #   0 -> ref exists
    #   1 -> ref does not exist
    #
    # Any other result fails closed.
    # ========================================================

    existing = (
        _run_git(
            target=target,

            args=[
                "show-ref",
                "--verify",
                "--quiet",
                ref_name,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if existing.get(
        "ok",
        False,
    ):

        return (
            None,
            None,
            None,
            (
                f"Local Git branch '{branch_name}' "
                "already exists."
            ),
        )

    if (
        existing.get(
            "exit_code"
        )
        != 1
    ):

        return (
            None,
            None,
            None,
            (
                "Could not safely determine whether "
                "the requested Git branch already exists."
            ),
        )

    return (
        target,
        branch_name,
        base_commit,
        None,
    )


def evaluate_git_create_branch_policy(
    repository: Any,
    branch_name: Any,
) -> dict[
    str,
    Any,
]:
    """
    Trusted pre-approval policy.

    No Git reference is created here.
    """

    (
        target,
        normalized_branch,
        base_commit,
        error,
    ) = (
        _validate_create_branch_request(
            repository=repository,
            branch_name=branch_name,
        )
    )

    if (
        target is None
        or normalized_branch is None
        or base_commit is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "error":
                (
                    error
                    or "Git branch creation policy denied the request."
                ),
        }

    return {
        "ok":
            True,

        "status":
            "allowed",

        "risk":
            "low",

        "requires_approval":
            True,
    }


def workspace_git_create_branch(
    repository: str,
    branch_name: str,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Create exactly one approved local Git branch from current HEAD.

    Approval arguments are revalidated immediately before mutation.

    The branch is created without checking it out.

    The model controls only:
        repository logical identifier
        exact branch name

    Trusted code controls:
        Git executable
        validation
        start point
        command shape
        verification
    """

    (
        target,
        normalized_branch,
        base_commit,
        error,
    ) = (
        _validate_create_branch_request(
            repository=repository,
            branch_name=branch_name,
            timeout_seconds=timeout_seconds,
        )
    )

    if (
        target is None
        or normalized_branch is None
        or base_commit is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "repository":
                repository,

            "requested_branch":
                branch_name,

            "error":
                (
                    error
                    or "Git branch creation request was denied."
                ),
        }

    # ========================================================
    # TRUSTED MUTATION
    #
    # Explicitly supply the already captured trusted HEAD commit.
    #
    # The branch name has passed Git's branch-name validator and
    # names beginning with '-' are rejected before this point.
    #
    # `git branch` refuses to overwrite an existing local branch.
    # ========================================================

    mutation = (
        _run_git(
            target=target,

            args=[
                "branch",
                normalized_branch,
                base_commit,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not mutation.get(
        "ok",
        False,
    ):

        failure = (
            _process_failure(
                repository=target.name,
                result=mutation,
            )
        )

        failure[
            "requested_branch"
        ] = (
            normalized_branch
        )

        failure[
            "created_from"
        ] = (
            base_commit
        )

        return (
            failure
        )

    # ========================================================
    # TRUSTED POST-MUTATION VERIFICATION
    #
    # Mutation already happened if `git branch` succeeded.
    # A failed verification must therefore not be reported as
    # "no mutation".
    # ========================================================

    ref_name = (
        f"refs/heads/{normalized_branch}"
    )

    verification = (
        _run_git(
            target=target,

            args=[
                "rev-parse",
                "--verify",
                ref_name,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not verification.get(
        "ok",
        False,
    ):

        verification_failure = (
            _process_failure(
                repository=target.name,
                result=verification,
            )
        )

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_branch":
                normalized_branch,

            "created_branch":
                normalized_branch,

            "created_from":
                base_commit,

            "created_commit":
                None,

            "files":
                [],

            "verification_ok":
                False,

            "verification_error":
                verification_failure.get(
                    "error"
                ),
        }

    verification_lines = (
        _stdout_lines(
            verification
        )
    )

    created_commit = (
        verification_lines[
            0
        ].strip()

        if (
            len(
                verification_lines
            )
            == 1
        )

        else None
    )

    if (
        created_commit
        != base_commit
    ):

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_branch":
                normalized_branch,

            "created_branch":
                normalized_branch,

            "created_from":
                base_commit,

            "created_commit":
                created_commit,

            "files":
                [],

            "verification_ok":
                False,

            "verification_error":
                (
                    "Created branch does not point to the "
                    "trusted pre-mutation HEAD commit."
                ),
        }

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "requested_branch":
            normalized_branch,

        "created_branch":
            normalized_branch,

        "created_from":
            base_commit,

        "created_commit":
            created_commit,

        # Keep generic Git learning extraction compatible.
        "files":
            [],

        "verification_ok":
            True,

        "verification_error":
            None,
    }


# ============================================================
# GOVERNED GIT BRANCH SWITCHING
# ============================================================


def _validate_switch_branch_request(
    repository: Any,
    branch_name: Any,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
):
    """
    Validate switching to one exact existing local Git branch.

    This performs no mutation.

    Safety contract:
        - exact configured repository
        - exact Git-valid branch name
        - branch must already exist locally
        - working tree must be completely clean
        - no remote branch guessing
        - no branch creation
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

        error = (
            target_error.get(
                "error"
            )
            if isinstance(
                target_error,
                dict,
            )
            else None
        )

        return (
            None,
            None,
            None,
            (
                error
                or "Git repository resolution failed."
            ),
        )

    if not isinstance(
        branch_name,
        str,
    ):

        return (
            None,
            None,
            None,
            "Git branch name must be a string.",
        )

    if not branch_name:

        return (
            None,
            None,
            None,
            "Git branch name cannot be empty.",
        )

    if (
        branch_name
        != branch_name.strip()
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch name must not contain "
                "leading or trailing whitespace."
            ),
        )

    if (
        len(
            branch_name
        )
        > MAX_GIT_BRANCH_CHARS
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch name exceeds the "
                f"{MAX_GIT_BRANCH_CHARS}-character limit."
            ),
        )

    if branch_name.startswith(
        "-"
    ):

        return (
            None,
            None,
            None,
            "Git branch name must not begin with '-'.",
        )

    if "@{" in branch_name:

        return (
            None,
            None,
            None,
            (
                "Git branch name must not contain "
                "reflog-selection syntax."
            ),
        )

    if any(
        ord(character) < 32
        or ord(character) == 127

        for character
        in branch_name
    ):

        return (
            None,
            None,
            None,
            "Git branch name contains control characters.",
        )

    # ========================================================
    # TRUSTED BRANCH-NAME VALIDATION
    # ========================================================

    validation = (
        _run_git(
            target=target,

            args=[
                "check-ref-format",
                "--branch",
                branch_name,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not validation.get(
        "ok",
        False,
    ):

        return (
            None,
            None,
            None,
            "Git branch name is not valid.",
        )

    validated_lines = (
        _stdout_lines(
            validation
        )
    )

    if (
        len(
            validated_lines
        )
        != 1
        or validated_lines[
            0
        ]
        != branch_name
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch name must resolve to exactly "
                "the explicitly requested value."
            ),
        )

    # ========================================================
    # REQUIRE EXISTING LOCAL BRANCH
    # ========================================================

    local_ref = (
        f"refs/heads/{branch_name}"
    )

    exists = (
        _run_git(
            target=target,

            args=[
                "show-ref",
                "--verify",
                "--quiet",
                local_ref,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not exists.get(
        "ok",
        False,
    ):

        if (
            exists.get(
                "exit_code"
            )
            == 1
        ):

            return (
                None,
                None,
                None,
                (
                    f"Local Git branch '{branch_name}' "
                    "does not exist."
                ),
            )

        return (
            None,
            None,
            None,
            (
                "Could not safely verify the requested "
                "local Git branch."
            ),
        )

    # ========================================================
    # CURRENT BRANCH
    # ========================================================

    current = (
        _run_git(
            target=target,

            args=[
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    current_branch = None

    if current.get(
        "ok",
        False,
    ):

        current_lines = (
            _stdout_lines(
                current
            )
        )

        if (
            len(
                current_lines
            )
            == 1
        ):

            current_branch = (
                current_lines[
                    0
                ].strip()
            )

    elif (
        current.get(
            "exit_code"
        )
        != 1
    ):

        return (
            None,
            None,
            None,
            (
                "Could not safely determine the current "
                "Git branch state."
            ),
        )

    # ========================================================
    # CLEAN-WORKTREE PRECONDITION
    #
    # Includes tracked modifications and untracked files.
    #
    # We intentionally do not allow Git to carry dirty state
    # across branches in this capability.
    # ========================================================

    working_tree = (
        _run_git(
            target=target,

            args=[
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not working_tree.get(
        "ok",
        False,
    ):

        return (
            None,
            None,
            None,
            (
                "Could not safely inspect the Git "
                "working tree before switching branches."
            ),
        )

    if (
        _stdout_lines(
            working_tree
        )
    ):

        return (
            None,
            None,
            None,
            (
                "Git branch switching requires a completely "
                "clean working tree. Commit, stash, or otherwise "
                "resolve staged, unstaged, and untracked changes "
                "before switching branches."
            ),
        )

    return (
        target,
        branch_name,
        current_branch,
        None,
    )


def evaluate_git_switch_branch_policy(
    repository: Any,
    branch_name: Any,
) -> dict[
    str,
    Any,
]:
    """
    Trusted pre-approval policy.

    This performs no branch switch.
    """

    (
        target,
        normalized_branch,
        current_branch,
        error,
    ) = (
        _validate_switch_branch_request(
            repository=repository,
            branch_name=branch_name,
        )
    )

    if (
        target is None
        or normalized_branch is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "error":
                (
                    error
                    or "Git branch-switch policy denied the request."
                ),
        }

    return {
        "ok":
            True,

        "status":
            "allowed",

        "risk":
            "low",

        "requires_approval":
            True,
    }


def workspace_git_switch_branch(
    repository: str,
    branch_name: str,
    timeout_seconds: int = (
        DEFAULT_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    Any,
]:
    """
    Switch to exactly one approved existing local Git branch.

    Approval arguments and the clean-worktree precondition are
    revalidated immediately before mutation.

    Trusted command:

        git switch --no-guess <exact-local-branch>

    This capability does NOT:
        - create branches
        - guess remote tracking branches
        - discard working-tree changes
        - stash changes
        - commit changes
        - fetch, pull, or push
    """

    (
        target,
        normalized_branch,
        previous_branch,
        error,
    ) = (
        _validate_switch_branch_request(
            repository=repository,
            branch_name=branch_name,
            timeout_seconds=timeout_seconds,
        )
    )

    if (
        target is None
        or normalized_branch is None
    ):

        return {
            "ok":
                False,

            "status":
                "denied",

            "repository":
                repository,

            "requested_branch":
                branch_name,

            "error":
                (
                    error
                    or "Git branch-switch request was denied."
                ),
        }

    # Already on the requested branch.
    #
    # Approval may have been created earlier, but execution is a
    # truthful no-op rather than issuing an unnecessary mutation.
    if (
        previous_branch
        == normalized_branch
    ):

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_branch":
                normalized_branch,

            "previous_branch":
                previous_branch,

            "current_branch":
                previous_branch,

            "switched_branch":
                normalized_branch,

            "switched_to_commit":
                None,

            "mutation_performed":
                False,

            "files":
                [],

            "verification_ok":
                True,

            "verification_error":
                None,
        }

    # ========================================================
    # TRUSTED MUTATION
    #
    # --no-guess prevents Git from creating a local tracking
    # branch from a similarly named remote branch.
    # ========================================================

    mutation = (
        _run_git(
            target=target,

            args=[
                "switch",
                "--no-guess",
                normalized_branch,
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not mutation.get(
        "ok",
        False,
    ):

        failure = (
            _process_failure(
                repository=target.name,
                result=mutation,
            )
        )

        failure[
            "requested_branch"
        ] = (
            normalized_branch
        )

        failure[
            "previous_branch"
        ] = (
            previous_branch
        )

        return (
            failure
        )

    # ========================================================
    # TRUSTED POST-MUTATION VERIFICATION
    # ========================================================

    current = (
        _run_git(
            target=target,

            args=[
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    if not current.get(
        "ok",
        False,
    ):

        verification_failure = (
            _process_failure(
                repository=target.name,
                result=current,
            )
        )

        return {
            "ok":
                True,

            "status":
                "success",

            "repository":
                target.name,

            "requested_branch":
                normalized_branch,

            "previous_branch":
                previous_branch,

            "current_branch":
                None,

            "switched_branch":
                normalized_branch,

            "switched_to_commit":
                None,

            "mutation_performed":
                True,

            "files":
                [],

            "verification_ok":
                False,

            "verification_error":
                verification_failure.get(
                    "error"
                ),
        }

    current_lines = (
        _stdout_lines(
            current
        )
    )

    current_branch = (
        current_lines[
            0
        ].strip()

        if (
            len(
                current_lines
            )
            == 1
        )

        else None
    )

    head_result = (
        _run_git(
            target=target,

            args=[
                "rev-parse",
                "--verify",
                "HEAD",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    ref_result = (
        _run_git(
            target=target,

            args=[
                "rev-parse",
                "--verify",
                f"refs/heads/{normalized_branch}",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    final_status = (
        _run_git(
            target=target,

            args=[
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],

            timeout_seconds=timeout_seconds,
        )
    )

    head_lines = (
        _stdout_lines(
            head_result
        )
        if head_result.get(
            "ok",
            False,
        )
        else []
    )

    ref_lines = (
        _stdout_lines(
            ref_result
        )
        if ref_result.get(
            "ok",
            False,
        )
        else []
    )

    final_changes = (
        _stdout_lines(
            final_status
        )
        if final_status.get(
            "ok",
            False,
        )
        else [
            "<verification unavailable>"
        ]
    )

    head_commit = (
        head_lines[
            0
        ].strip()

        if (
            len(
                head_lines
            )
            == 1
        )

        else None
    )

    branch_commit = (
        ref_lines[
            0
        ].strip()

        if (
            len(
                ref_lines
            )
            == 1
        )

        else None
    )

    verification_ok = (
        current_branch
        == normalized_branch

        and head_commit is not None

        and branch_commit is not None

        and head_commit
        == branch_commit

        and not final_changes
    )

    verification_error = None

    if not verification_ok:

        problems = []

        if (
            current_branch
            != normalized_branch
        ):

            problems.append(
                "current branch does not match the requested branch"
            )

        if (
            head_commit is None
            or branch_commit is None
            or head_commit
            != branch_commit
        ):

            problems.append(
                "HEAD does not match the requested local branch ref"
            )

        if final_changes:

            problems.append(
                "working tree is no longer clean after the switch"
            )

        verification_error = (
            "; ".join(
                problems
            )
            or "Git branch-switch verification failed."
        )

    return {
        "ok":
            True,

        "status":
            "success",

        "repository":
            target.name,

        "requested_branch":
            normalized_branch,

        "previous_branch":
            previous_branch,

        "current_branch":
            current_branch,

        "switched_branch":
            normalized_branch,

        "switched_to_commit":
            head_commit,

        "mutation_performed":
            True,

        "files":
            [],

        "verification_ok":
            verification_ok,

        "verification_error":
            verification_error,
    }
