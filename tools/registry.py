from typing import (
    Any,
    Callable,
)

from services.process_runner import (
    evaluate_process_policy,
)

from tools.presentation import (
    format_access_check_result,
    format_account_status_result,
    format_generic_approval,
    format_git_branches_result,
    format_git_changed_files_result,
    format_git_diff_result,
    format_git_log_result,
    format_git_status_result,
    format_password_reset_approval,
    format_process_result,
    format_unlock_approval,
    format_workspace_read_result,
)


TOOLS = {
    # ============================================================
    # DIRECTORY / IDENTITY
    # ============================================================

    "account_status": {
        "description": (
            "Check whether an account is enabled or locked."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "user_id",
        ],

        "parameters": {
            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier to check."
                ),
            },
        },

        "result_formatter": (
            format_account_status_result
        ),
    },

    "check_access": {
        "description": (
            "Check whether a user can access a resource."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "user_id",
            "resource",
        ],

        "parameters": {
            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier to check."
                ),
            },

            "resource": {
                "type":
                    "str",

                "description": (
                    "Exact resource identifier or name."
                ),
            },
        },

        "result_formatter": (
            format_access_check_result
        ),
    },

    "unlock_user": {
        "description":
            "Unlock an account.",

        "risk":
            "low",

        "requires_approval":
            True,

        "grounded_arguments": [
            "user_id",
        ],

        "parameters": {
            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier to unlock."
                ),
            },
        },

        "approval_formatter": (
            format_unlock_approval
        ),
    },

    "reset_password": {
        "description":
            "Reset an account password.",

        "risk":
            "high",

        "requires_approval":
            True,

        "grounded_arguments": [
            "user_id",
        ],

        "parameters": {
            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier."
                ),
            },
        },

        "approval_formatter": (
            format_password_reset_approval
        ),
    },

    # ============================================================
    # GENERIC GOVERNED PROCESS
    # ============================================================

    "process_exec": {
        "description": (
            "Execute an approved native process "
            "inside the configured workspace."
        ),

        "risk":
            "low",

        "requires_approval":
            True,

        "policy_resolver": (
            evaluate_process_policy
        ),

        "grounded_arguments":
            [],

        "parameters": {
            "executable": {
                "type":
                    "str",

                "description": (
                    "Approved native executable."
                ),
            },

            "args": {
                "type":
                    "list[str]",

                "description": (
                    "Structured executable arguments."
                ),
            },

            "cwd": {
                "type":
                    "str",

                "description": (
                    "Optional working directory inside "
                    "the approved workspace."
                ),
            },

            "timeout_seconds": {
                "type":
                    "int",

                "description": (
                    "Execution timeout from 1 to 30 seconds."
                ),
            },
        },

        "result_formatter": (
            format_process_result
        ),

        "approval_formatter": (
            format_generic_approval
        ),
    },

    # ============================================================
    # WORKSPACE
    # ============================================================

    "workspace_mkdir": {
        "description": (
            "Create exactly one direct-child directory "
            "inside the approved developer workspace."
        ),

        "risk":
            "low",

        "requires_approval":
            True,

        "grounded_arguments": [
            "directory_name",
        ],

        "parameters": {
            "directory_name": {
                "type":
                    "str",

                "description": (
                    "Exact directory name explicitly "
                    "requested by the user."
                ),
            },

            "cwd": {
                "type":
                    "str",

                "description": (
                    "Optional working directory inside "
                    "the approved workspace."
                ),
            },

            "timeout_seconds": {
                "type":
                    "int",

                "description": (
                    "Execution timeout from 1 to 30 seconds."
                ),
            },
        },

        "approval_formatter": (
            format_generic_approval
        ),
    },

    "workspace_read_text": {
        "description": (
            "Read one approved UTF-8 text or source file "
            "from the developer workspace."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "relative_path",
        ],

        "parameters": {
            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Exact workspace-relative file path "
                    "explicitly supplied by the user."
                ),
            },
        },

        "result_formatter": (
            format_workspace_read_result
        ),
    },

    # ============================================================
    # GIT — READ-ONLY CAPABILITY ADAPTERS
    #
    # Git arguments are not supplied by the model.
    # Each tool owns one deterministic Git operation.
    # ============================================================

    "workspace_git_status": {
        "description": (
            "Inspect the current Git branch and "
            "working-tree status."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters":
            {},

        "result_formatter": (
            format_git_status_result
        ),
    },

    "workspace_git_branches": {
        "description": (
            "List local Git branches and identify "
            "the current branch."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters":
            {},

        "result_formatter": (
            format_git_branches_result
        ),
    },

    "workspace_git_log": {
        "description": (
            "Inspect the most recent Git commit history "
            "for the current repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters":
            {},

        "result_formatter": (
            format_git_log_result
        ),
    },

    "workspace_git_diff": {
        "description": (
            "Inspect the current unstaged Git diff."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters":
            {},

        "result_formatter": (
            format_git_diff_result
        ),
    },

    "workspace_git_changed_files": {
        "description": (
            "List files with unstaged Git changes "
            "in the current repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments":
            [],

        "parameters":
            {},

        "result_formatter": (
            format_git_changed_files_result
        ),
    },
}


def list_tools(
) -> list[str]:
    """
    Return registered trusted tool names.
    """

    return list(
        TOOLS.keys()
    )


def get_tool(
    name: str,
) -> dict | None:
    """
    Return one trusted tool definition.
    """

    return (
        TOOLS.get(
            name
        )
    )


def format_tool_result(
    name: str,
    result: dict[
        str,
        Any,
    ],
) -> str:
    """
    Convert trusted structured tool output into text suitable
    for specialist/Main synthesis.
    """

    tool = (
        get_tool(
            name
        )
    )

    if tool is None:
        return (
            f"The {name} operation "
            "completed successfully."
        )

    formatter: (
        Callable | None
    ) = (
        tool.get(
            "result_formatter"
        )
    )

    if formatter is None:
        return (
            f"The {name} operation "
            "completed successfully."
        )

    return formatter(
        result
    )


def format_approval_required(
    name: str,
    arguments: dict[
        str,
        Any,
    ],
    approval_id: str | None,
) -> str:
    """
    Format deterministic approval-required responses.
    """

    tool = (
        get_tool(
            name
        )
    )

    formatter: (
        Callable | None
    ) = (
        tool.get(
            "approval_formatter"
        )
        if tool is not None
        else None
    )

    if formatter is None:
        message = (
            "This action requires approval."
        )

    else:
        message = (
            formatter(
                arguments
            )
        )

    if approval_id:
        return (
            f"{message} "
            f"Approval ID: {approval_id}."
        )

    return message