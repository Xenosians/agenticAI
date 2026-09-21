from __future__ import annotations

from typing import (
    Any,
)

from tools.presentation.result_cards import (
    build_result_card,
    list_section,
    result_field,
    text_section,
)


def _repository_name(
    result: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        result.get(
            "repository"
        )
    )

    if (
        isinstance(
            repository,
            str,
        )
        and repository.strip()
    ):

        return (
            repository.strip()
        )

    return "repository"


# ============================================================
# STATUS
# ============================================================


def format_git_status_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        _repository_name(
            result
        )
    )

    branch = (
        result.get(
            "branch"
        )
    )

    branch_text = (
        branch
        if isinstance(
            branch,
            str,
        )
        else "unknown"
    )

    if (
        result.get(
            "clean"
        )
        is True
    ):

        return (
            f"{repository} Git status:\n"
            f"- Branch: {branch_text}\n"
            "- Working tree: clean"
        )

    return (
        f"{repository} Git status:\n"
        f"- Branch: {branch_text}\n"
        f"- Changed files: {result.get('count', 0)}\n"
        f"- Staged: {result.get('staged_count', 0)}\n"
        f"- Unstaged: {result.get('unstaged_count', 0)}\n"
        f"- Untracked: {result.get('untracked_count', 0)}\n"
        f"- Conflicted: {result.get('conflicted_count', 0)}"
    )


def build_git_status_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    repository = (
        _repository_name(
            result
        )
    )

    changes = (
        result.get(
            "changes"
        )
    )

    if not isinstance(
        changes,
        list,
    ):

        changes = []

    items: list[str] = []

    for change in changes:

        if not isinstance(
            change,
            dict,
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

        code = (
            change.get(
                "code"
            )
        )

        if (
            isinstance(
                code,
                str,
            )
            and code.strip()
        ):

            items.append(
                f"{code} {path}"
            )

        else:

            items.append(
                path
            )

    sections = []

    if items:

        sections.append(
            list_section(
                title="Working tree changes",
                items=items,
            )
        )

    return (
        build_result_card(
            kind="git-status",

            title=(
                f"{repository} Git status"
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "unknown",
                    )
                )
            ),

            fields=[
                result_field(
                    "Repository",
                    repository,
                ),

                result_field(
                    "Branch",
                    result.get(
                        "branch"
                    ),
                ),

                result_field(
                    "Upstream",
                    result.get(
                        "upstream"
                    ),
                ),

                result_field(
                    "Ahead",
                    result.get(
                        "ahead"
                    ),
                ),

                result_field(
                    "Behind",
                    result.get(
                        "behind"
                    ),
                ),

                result_field(
                    "Clean",
                    result.get(
                        "clean"
                    ),
                ),

                result_field(
                    "Changed files",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "Staged",
                    result.get(
                        "staged_count"
                    ),
                ),

                result_field(
                    "Unstaged",
                    result.get(
                        "unstaged_count"
                    ),
                ),

                result_field(
                    "Untracked",
                    result.get(
                        "untracked_count"
                    ),
                ),

                result_field(
                    "Conflicted",
                    result.get(
                        "conflicted_count"
                    ),
                ),

                result_field(
                    "Truncated",
                    result.get(
                        "truncated"
                    ),
                ),
            ],

            sections=(
                sections
            ),
        )
    )


# ============================================================
# BRANCHES
# ============================================================


def format_git_branches_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        _repository_name(
            result
        )
    )

    branches = (
        result.get(
            "branches"
        )
    )

    if not isinstance(
        branches,
        list,
    ):

        return (
            f"No valid local Git branches were "
            f"returned for {repository}."
        )

    lines: list[str] = []

    for branch in branches:

        if not isinstance(
            branch,
            dict,
        ):

            continue

        name = (
            branch.get(
                "name"
            )
        )

        if not isinstance(
            name,
            str,
        ):

            continue

        suffix = (
            " (current)"
            if branch.get(
                "current"
            )
            is True
            else ""
        )

        lines.append(
            f"- {name}{suffix}"
        )

    if not lines:

        return (
            f"No local Git branches were "
            f"returned for {repository}."
        )

    return (
        f"{repository} local Git branches:\n"
        + "\n".join(
            lines
        )
    )


def build_git_branches_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    repository = (
        _repository_name(
            result
        )
    )

    branches = (
        result.get(
            "branches"
        )
    )

    if not isinstance(
        branches,
        list,
    ):

        branches = []

    items: list[str] = []

    for branch in branches:

        if not isinstance(
            branch,
            dict,
        ):

            continue

        name = (
            branch.get(
                "name"
            )
        )

        if not isinstance(
            name,
            str,
        ):

            continue

        if (
            branch.get(
                "current"
            )
            is True
        ):

            name = (
                f"{name} (current)"
            )

        items.append(
            name
        )

    sections = []

    if items:

        sections.append(
            list_section(
                title="Branches",
                items=items,
            )
        )

    return (
        build_result_card(
            kind="git-branches",

            title=(
                f"{repository} Git branches"
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "unknown",
                    )
                )
            ),

            fields=[
                result_field(
                    "Repository",
                    repository,
                ),

                result_field(
                    "Current branch",
                    result.get(
                        "current_branch"
                    ),
                ),

                result_field(
                    "Branches",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "Truncated",
                    result.get(
                        "truncated"
                    ),
                ),
            ],

            sections=(
                sections
            ),
        )
    )


# ============================================================
# LOG
# ============================================================


def format_git_log_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        _repository_name(
            result
        )
    )

    commits = (
        result.get(
            "commits"
        )
    )

    if not isinstance(
        commits,
        list,
    ):

        return (
            f"No valid Git history was "
            f"returned for {repository}."
        )

    lines: list[str] = []

    for commit in commits:

        if not isinstance(
            commit,
            dict,
        ):

            continue

        commit_hash = (
            commit.get(
                "hash"
            )
        )

        if not isinstance(
            commit_hash,
            str,
        ):

            continue

        message = (
            commit.get(
                "message"
            )
        )

        if isinstance(
            message,
            str,
        ):

            lines.append(
                f"- {commit_hash} {message}"
            )

        else:

            lines.append(
                f"- {commit_hash}"
            )

    if not lines:

        return (
            f"No Git history was returned "
            f"for {repository}."
        )

    return (
        f"Recent commits in {repository}:\n"
        + "\n".join(
            lines
        )
    )


def build_git_log_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    repository = (
        _repository_name(
            result
        )
    )

    commits = (
        result.get(
            "commits"
        )
    )

    if not isinstance(
        commits,
        list,
    ):

        commits = []

    items: list[str] = []

    for commit in commits:

        if not isinstance(
            commit,
            dict,
        ):

            continue

        commit_hash = (
            commit.get(
                "hash"
            )
        )

        if not isinstance(
            commit_hash,
            str,
        ):

            continue

        message = (
            commit.get(
                "message"
            )
        )

        if (
            isinstance(
                message,
                str,
            )
            and message.strip()
        ):

            items.append(
                f"{commit_hash} — {message.strip()}"
            )

        else:

            items.append(
                commit_hash
            )

    sections = []

    if items:

        sections.append(
            list_section(
                title="Recent commits",
                items=items,
            )
        )

    return (
        build_result_card(
            kind="git-log",

            title=(
                f"{repository} Git history"
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "unknown",
                    )
                )
            ),

            fields=[
                result_field(
                    "Repository",
                    repository,
                ),

                result_field(
                    "Commits",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "Truncated",
                    result.get(
                        "truncated"
                    ),
                ),
            ],

            sections=(
                sections
            ),
        )
    )


# ============================================================
# DIFF HELPERS
# ============================================================


def _format_diff_result(
    result: dict[
        str,
        Any,
    ],
    *,
    scope: str,
) -> str:

    repository = (
        _repository_name(
            result
        )
    )

    diff = (
        result.get(
            "diff"
        )
    )

    if not isinstance(
        diff,
        str,
    ):

        return (
            f"{scope.capitalize()} Git diff completed for "
            f"{repository}, but no valid diff text was returned."
        )

    if not diff.strip():

        return (
            f"There are no {scope} Git changes "
            f"in {repository}."
        )

    return (
        f"Current {scope} Git diff for "
        f"{repository}:\n\n"
        f"{diff}"
    )


def _build_diff_card(
    result: dict[
        str,
        Any,
    ],
    *,
    kind: str,
    title_suffix: str,
    section_title: str,
) -> dict[
    str,
    Any,
]:

    repository = (
        _repository_name(
            result
        )
    )

    diff = (
        result.get(
            "diff"
        )
    )

    sections = []

    if (
        isinstance(
            diff,
            str,
        )
        and diff.strip()
    ):

        sections.append(
            text_section(
                title=(
                    section_title
                ),

                content=(
                    diff
                ),

                kind="preformatted",
            )
        )

    return (
        build_result_card(
            kind=(
                kind
            ),

            title=(
                f"{repository} {title_suffix}"
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "unknown",
                    )
                )
            ),

            fields=[
                result_field(
                    "Repository",
                    repository,
                ),

                result_field(
                    "Scope",
                    result.get(
                        "scope"
                    ),
                ),

                result_field(
                    "Has changes",
                    result.get(
                        "has_changes"
                    ),
                ),

                result_field(
                    "Truncated",
                    result.get(
                        "truncated"
                    ),
                ),
            ],

            sections=(
                sections
            ),
        )
    )


# ============================================================
# UNSTAGED DIFF
# ============================================================


def format_git_diff_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _format_diff_result(
            result,
            scope="unstaged",
        )
    )


def build_git_diff_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    return (
        _build_diff_card(
            result,

            kind="git-diff",

            title_suffix="Git diff",

            section_title="Diff",
        )
    )


# ============================================================
# STAGED DIFF
# ============================================================


def format_git_staged_diff_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _format_diff_result(
            result,
            scope="staged",
        )
    )


def build_git_staged_diff_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    return (
        _build_diff_card(
            result,

            kind="git-staged-diff",

            title_suffix="staged Git diff",

            section_title="Staged diff",
        )
    )


# ============================================================
# CHANGED FILES
# ============================================================


def format_git_changed_files_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        _repository_name(
            result
        )
    )

    files = (
        result.get(
            "files"
        )
    )

    if not isinstance(
        files,
        list,
    ):

        return (
            f"No valid changed-file list was "
            f"returned for {repository}."
        )

    valid_files = [
        path

        for path
        in files

        if isinstance(
            path,
            str,
        )
    ]

    if not valid_files:

        return (
            f"There are no current changed files "
            f"in {repository}."
        )

    lines = [
        f"Changed files in {repository}:",
        *[
            f"- {path}"

            for path
            in valid_files
        ],
        "",
        (
            "Summary: "
            f"staged={result.get('staged_count', 0)}, "
            f"unstaged={result.get('unstaged_count', 0)}, "
            f"untracked={result.get('untracked_count', 0)}, "
            f"conflicted={result.get('conflicted_count', 0)}"
        ),
    ]

    return (
        "\n".join(
            lines
        )
    )


def build_git_changed_files_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:

    repository = (
        _repository_name(
            result
        )
    )

    files = (
        result.get(
            "files"
        )
    )

    if not isinstance(
        files,
        list,
    ):

        files = []

    items = [
        path

        for path
        in files

        if (
            isinstance(
                path,
                str,
            )
            and path.strip()
        )
    ]

    sections = []

    if items:

        sections.append(
            list_section(
                title="Changed files",
                items=items,
            )
        )

    return (
        build_result_card(
            kind="git-files",

            title=(
                f"{repository} changed files"
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "unknown",
                    )
                )
            ),

            fields=[
                result_field(
                    "Repository",
                    repository,
                ),

                result_field(
                    "Scope",
                    result.get(
                        "scope"
                    ),
                ),

                result_field(
                    "Files",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "Staged",
                    result.get(
                        "staged_count"
                    ),
                ),

                result_field(
                    "Unstaged",
                    result.get(
                        "unstaged_count"
                    ),
                ),

                result_field(
                    "Untracked",
                    result.get(
                        "untracked_count"
                    ),
                ),

                result_field(
                    "Conflicted",
                    result.get(
                        "conflicted_count"
                    ),
                ),

                result_field(
                    "Truncated",
                    result.get(
                        "truncated"
                    ),
                ),
            ],

            sections=(
                sections
            ),
        )
    )

# ============================================================
# UNSTAGE FILES
# ============================================================


def format_git_unstage_files_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        _repository_name(
            result
        )
    )

    if (
        result.get(
            "verification_ok"
        )
        is False
    ):

        return (
            f"Git unstaging completed for {repository}, "
            "but the final staged-state verification failed."
        )

    unstaged_paths = (
        result.get(
            "unstaged_paths"
        )
    )

    if not isinstance(
        unstaged_paths,
        list,
    ):

        unstaged_paths = []

    if not unstaged_paths:

        return (
            f"None of the explicitly requested files "
            f"were staged in {repository}; no index "
            "mutation was required."
        )

    lines = [
        (
            f"Unstaged {len(unstaged_paths)} "
            f"explicitly requested file(s) in "
            f"{repository}:"
        ),
    ]

    for path in unstaged_paths:

        if isinstance(
            path,
            str,
        ):

            lines.append(
                f"- {path}"
            )

    return (
        "\n".join(
            lines
        )
    )


def format_git_unstage_files_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    repository = (
        arguments.get(
            "repository"
        )
    )

    paths = (
        arguments.get(
            "paths"
        )
    )

    repository_text = (
        repository

        if (
            isinstance(
                repository,
                str,
            )
            and repository.strip()
        )

        else "repository"
    )

    valid_paths = [
        path

        for path
        in (
            paths
            if isinstance(
                paths,
                list,
            )
            else []
        )

        if isinstance(
            path,
            str,
        )
    ]

    rendered_paths = (
        ", ".join(
            valid_paths
        )
        if valid_paths
        else "no valid paths"
    )

    return (
        f"Unstaging {len(valid_paths)} explicitly requested "
        f"file(s) in {repository_text} requires approval: "
        f"{rendered_paths}."
    )
