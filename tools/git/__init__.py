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
