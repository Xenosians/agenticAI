from __future__ import annotations

from services.git_repositories import (
    build_git_repository_registry,
)

from tools.git.presentation import (
    build_git_branches_card,
    build_git_changed_files_card,
    build_git_diff_card,
    build_git_log_card,
    build_git_status_card,
    format_git_branches_result,
    format_git_changed_files_result,
    format_git_diff_result,
    format_git_log_result,
    format_git_status_result,
)


# ============================================================
# MODEL-FACING BOUNDED ARGUMENT VALUES
# ============================================================


def resolve_git_argument_values(
) -> dict[
    str,
    list[str],
]:
    """
    Expose configured logical Git repository identifiers to the
    capability catalog.

    Values come from trusted runtime configuration.

    Physical paths are never exposed.
    """

    registry = (
        build_git_repository_registry()
    )

    return {
        "repository":
            registry.available(),
    }


# ============================================================
# SHARED REPOSITORY ARGUMENT
# ============================================================


REPOSITORY_PARAMETER = {
    "type":
        "str",

    "description": (
        "Optional configured logical repository identifier. "
        "When enum values are present, return one exact enum "
        "value corresponding to the repository requested by "
        "the user. Omit this argument when the user does not "
        "identify a repository so trusted configuration may "
        "select the default."
    ),
}


# ============================================================
# GIT CAPABILITY CATALOG
# ============================================================


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
            "one configured logical repository. This capability "
            "does not include staged changes."
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
}