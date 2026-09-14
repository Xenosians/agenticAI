from tools.git_presentation import (
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


REPOSITORY_PARAMETER = {
    "type":
        "str",

    "description": (
        "Optional configured logical repository identifier "
        "explicitly supplied by the user, such as ai, backend, "
        "or frontend. Omit this argument when the user does not "
        "name a repository so trusted configuration can use the "
        "default repository."
    ),
}


GIT_TOOLS = {
    "workspace_git_status": {
        "description": (
            "Inspect the current branch and working-tree state "
            "of one configured logical Git repository."
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

        "result_formatter":
            format_git_log_result,

        "presentation_builder":
            build_git_log_card,
    },

    "workspace_git_diff": {
        "description": (
            "Retrieve the bounded current unstaged Git diff for "
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

        "result_formatter":
            format_git_diff_result,

        "presentation_builder":
            build_git_diff_card,
    },

    "workspace_git_changed_files": {
        "description": (
            "List files with current unstaged Git changes in "
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

        "result_formatter":
            format_git_changed_files_result,

        "presentation_builder":
            build_git_changed_files_card,
    },
}