from __future__ import annotations

from typing import (
    Any,
)

from tools.result_cards import (
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

    return (
        "repository"
    )


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

    clean = (
        result.get(
            "clean"
        )
    )

    change_count = (
        result.get(
            "change_count",
            0,
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

    if clean is True:
        return (
            f"{repository} Git status:\n"
            f"- Branch: {branch_text}\n"
            "- Working tree: clean"
        )

    return (
        f"{repository} Git status:\n"
        f"- Branch: {branch_text}\n"
        f"- Working tree changes: {change_count}"
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

    raw_changes = (
        result.get(
            "changes"
        )
    )

    changes = (
        raw_changes
        if isinstance(
            raw_changes,
            list,
        )
        else []
    )

    items = []

    for change in changes:
        if not isinstance(
            change,
            dict,
        ):
            continue

        code = (
            change.get(
                "code"
            )
        )

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

        rendered = (
            path
        )

        if (
            isinstance(
                code,
                str,
            )
            and code.strip()
        ):
            rendered = (
                f"{code} {path}"
            )

        items.append(
            rendered
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
                    "Changes",
                    result.get(
                        "change_count"
                    ),
                ),
            ],

            sections=sections,
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

    if not branches:
        return (
            f"No local Git branches were "
            f"returned for {repository}."
        )

    lines = []

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
            lines.append(
                f"- {name} (current)"
            )

        else:
            lines.append(
                f"- {name}"
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

    items = []

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
            items.append(
                f"{name} (current)"
            )

        else:
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
            ],

            sections=sections,
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

    if not commits:
        return (
            f"No Git history was returned "
            f"for {repository}."
        )

    lines = []

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

        message = (
            commit.get(
                "message"
            )
        )

        if not isinstance(
            commit_hash,
            str,
        ):
            continue

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

    items = []

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

        message = (
            commit.get(
                "message"
            )
        )

        if not isinstance(
            commit_hash,
            str,
        ):
            continue

        rendered = (
            commit_hash
        )

        if (
            isinstance(
                message,
                str,
            )
            and message.strip()
        ):
            rendered += (
                f" — {message.strip()}"
            )

        items.append(
            rendered
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
            ],

            sections=sections,
        )
    )


# ============================================================
# DIFF
# ============================================================


def format_git_diff_result(
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
            f"Git diff completed for {repository}, "
            "but no valid diff text was returned."
        )

    if not diff.strip():
        return (
            f"There are no unstaged Git changes "
            f"in {repository}."
        )

    return (
        f"Current unstaged Git diff for "
        f"{repository}:\n\n"
        f"{diff}"
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
                title="Diff",
                content=(
                    diff
                ),
                kind="preformatted",
            )
        )

    return (
        build_result_card(
            kind="git-diff",

            title=(
                f"{repository} Git diff"
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

            sections=sections,
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

    if not files:
        return (
            f"There are no unstaged changed files "
            f"in {repository}."
        )

    return (
        f"Changed files in {repository}:\n"
        + "\n".join(
            f"- {path}"
            for path in files
            if isinstance(
                path,
                str,
            )
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

        if isinstance(
            path,
            str,
        )
        and path.strip()
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
                    "Files",
                    result.get(
                        "count"
                    ),
                ),
            ],

            sections=sections,
        )
    )