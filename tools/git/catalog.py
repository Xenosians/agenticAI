from __future__ import annotations

from tools.git import (
    evaluate_git_unstage_policy,
)

from tools.git.presentation import (
    format_git_unstage_files_approval,
    format_git_unstage_files_result,
)

from typing import (
    Any,
)

from services.git_repositories import (
    build_git_repository_registry,
)

from tools.git import (
    evaluate_git_stage_policy,
)

from tools.git.presentation import (
    build_git_branches_card,
    build_git_changed_files_card,
    build_git_diff_card,
    build_git_log_card,
    build_git_staged_diff_card,
    build_git_status_card,
    format_git_branches_result,
    format_git_changed_files_result,
    format_git_diff_result,
    format_git_log_result,
    format_git_staged_diff_result,
    format_git_status_result,
)


def resolve_git_argument_values(
) -> dict[
    str,
    list[str],
]:

    registry = (
        build_git_repository_registry()
    )

    return {
        "repository":
            registry.available(),
    }


REPOSITORY_PARAMETER = {
    "type":
        "str",

    "description": (
        "Configured logical repository identifier. "
        "Use the exact trusted enum value corresponding "
        "to the repository explicitly requested by the user."
    ),
}


def format_git_stage_files_approval(
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

    if not isinstance(
        repository,
        str,
    ):

        repository = (
            "the repository"
        )

    if not isinstance(
        paths,
        list,
    ):

        return (
            f"Staging files in {repository} "
            "requires approval."
        )

    valid_paths = [
        path

        for path
        in paths

        if isinstance(
            path,
            str,
        )
    ]

    if not valid_paths:

        return (
            f"Staging files in {repository} "
            "requires approval."
        )

    preview = (
        ", ".join(
            valid_paths[
                :5
            ]
        )
    )

    if len(
        valid_paths
    ) > 5:

        preview += (
            f", +{len(valid_paths) - 5} more"
        )

    return (
        f"Staging {len(valid_paths)} explicitly requested "
        f"file(s) in {repository} requires approval: "
        f"{preview}."
    )


def format_git_stage_files_result(
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

    requested_paths = (
        result.get(
            "requested_paths"
        )
    )

    staged_paths = (
        result.get(
            "staged_paths"
        )
    )

    verification_ok = (
        result.get(
            "verification_ok"
        )
    )

    if not isinstance(
        repository,
        str,
    ):

        repository = (
            "repository"
        )

    requested_count = (
        len(
            requested_paths
        )
        if isinstance(
            requested_paths,
            list,
        )
        else 0
    )

    staged_count = (
        len(
            staged_paths
        )
        if isinstance(
            staged_paths,
            list,
        )
        else 0
    )

    if verification_ok is False:

        return (
            f"Git staging completed in {repository} for "
            f"{requested_count} requested file(s), but "
            "post-staging verification could not be completed."
        )

    return (
        f"Git staging completed in {repository}. "
        f"Requested files: {requested_count}. "
        f"Files currently present in the staged diff: "
        f"{staged_count}."
    )


GIT_TOOLS = {
    "workspace_git_status": {
        "description": (
            "Inspect the current branch and complete working-tree "
            "state of one configured logical Git repository, "
            "including staged, unstaged, untracked, and conflicted "
            "files."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository":
                REPOSITORY_PARAMETER,
        },

        "argument_values_resolver":
            resolve_git_argument_values,

        "result_formatter":
            format_git_status_result,

        "presentation_builder":
            build_git_status_card,
    },

    "workspace_git_branches": {
        "description": (
            "List local Git branches and identify the current "
            "branch for one configured logical repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository":
                REPOSITORY_PARAMETER,
        },

        "argument_values_resolver":
            resolve_git_argument_values,

        "result_formatter":
            format_git_branches_result,

        "presentation_builder":
            build_git_branches_card,
    },

    "workspace_git_log": {
        "description": (
            "Retrieve the bounded recent commit history for "
            "one configured logical Git repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository":
                REPOSITORY_PARAMETER,
        },

        "argument_values_resolver":
            resolve_git_argument_values,

        "result_formatter":
            format_git_log_result,

        "presentation_builder":
            build_git_log_card,
    },

    "workspace_git_diff": {
        "description": (
            "Retrieve the bounded current UNSTAGED Git diff for "
            "one configured logical repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository":
                REPOSITORY_PARAMETER,
        },

        "argument_values_resolver":
            resolve_git_argument_values,

        "result_formatter":
            format_git_diff_result,

        "presentation_builder":
            build_git_diff_card,
    },

    "workspace_git_staged_diff": {
        "description": (
            "Retrieve the bounded current STAGED Git diff for "
            "one configured logical repository. Use this for "
            "changes currently ready to commit."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository":
                REPOSITORY_PARAMETER,
        },

        "argument_values_resolver":
            resolve_git_argument_values,

        "result_formatter":
            format_git_staged_diff_result,

        "presentation_builder":
            build_git_staged_diff_card,
    },

    "workspace_git_changed_files": {
        "description": (
            "List all currently changed files in one configured "
            "logical Git repository, including staged, unstaged, "
            "untracked, and conflicted files."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
        ],

        "parameters": {
            "repository":
                REPOSITORY_PARAMETER,
        },

        "argument_values_resolver":
            resolve_git_argument_values,

        "result_formatter":
            format_git_changed_files_result,

        "presentation_builder":
            build_git_changed_files_card,
    },

    # ========================================================
    # MUTATION
    # ========================================================

    "workspace_git_stage_files": {
    "description": (
        "GIT STAGE explicitly named files in one configured Git "
        "repository, equivalent to a governed `git add` of those "
        "exact files. Use this when the user asks to stage a file, "
        "stage files, add files to the Git index, or prepare "
        "specific files for commit. "
        "This modifies only the Git index and requires approval. "
        "It does NOT create directories or folders, create files, "
        "commit changes, switch branches, pull, push, or modify "
        "remote state."
    ),

    "risk":
        "low",

    "requires_approval":
        True,

    "policy_resolver":
        evaluate_git_stage_policy,

    "grounded_arguments": [
        "repository",
        "paths",
    ],

    "parameters": {
        "repository":
            REPOSITORY_PARAMETER,

        "paths": {
            "type":
                "list[str]",

            "description": (
                "Exact repository-relative file paths explicitly "
                "supplied by the user. Return every requested file "
                "and do not invent, omit, broaden, glob, normalize, "
                "or rewrite paths."
            ),
        },
    },

    "argument_values_resolver":
        resolve_git_argument_values,

    "result_formatter":
        format_git_stage_files_result,

    "approval_formatter":
        format_git_stage_files_approval,
    },
}

# ============================================================
# GOVERNED GIT MUTATION — UNSTAGE
# ============================================================

GIT_TOOLS[
    "workspace_git_unstage_files"
] = {
    "description": (
        "GIT UNSTAGE explicitly named files from the Git index "
        "in one configured Git repository. Use this when the user "
        "asks to unstage a file, unstage files, remove specific "
        "files from the staging area, or move specific staged "
        "changes back to the working tree. This modifies only the "
        "Git index and requires approval. It does NOT delete files, "
        "discard working-tree changes, revert file contents, create "
        "directories, commit changes, switch branches, pull, push, "
        "or modify remote state."
    ),

    "risk":
        "low",

    "requires_approval":
        True,

    "policy_resolver":
        evaluate_git_unstage_policy,

    "grounded_arguments": [
        "repository",
        "paths",
    ],

    "parameters": {
        "repository": {
            "type":
                "str",

            "description": (
                "Configured logical repository identifier. "
                "Use the exact trusted enum value corresponding "
                "to the repository explicitly requested by "
                "the user."
            ),
        },

        "paths": {
            "type":
                "list[str]",

            "description": (
                "Exact repository-relative file paths explicitly "
                "supplied by the user. Return every requested "
                "file and do not invent, omit, broaden, glob, "
                "normalize, or rewrite paths."
            ),
        },
    },

    "argument_values_resolver":
        resolve_git_argument_values,

    "result_formatter":
        format_git_unstage_files_result,

    "approval_formatter":
        format_git_unstage_files_approval,
}
