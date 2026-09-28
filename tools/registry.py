from typing import (
    Any,
    Callable,
)

from services.process_runner import (
    evaluate_process_policy,
)

from tools.account.catalog import (
    ACCOUNT_TOOLS,
)

from tools.access.catalog import (
    ACCESS_TOOLS,
)

from tools.atlassian.catalog import (
    ATLASSIAN_TOOLS,
)

from tools.assets.catalog import (
    ASSET_TOOLS,
)

from tools.developer.catalog import (
    DEVELOPER_TOOLS,
)

from tools.git.catalog import (
    GIT_TOOLS,
)

from tools.jira.catalog import (
    JIRA_TOOLS,
)


from tools.knowledge.catalog import (
    KNOWLEDGE_TOOLS,
)

from tools.presentation import (
    format_access_check_result,
    format_account_status_result,
    format_generic_approval,
    format_password_reset_approval,
    format_process_result,
    format_unlock_approval,
    format_workspace_read_result,
)

from tools.ticketing.catalog import (
    TICKETING_TOOLS,
)

from tools.workspace.catalog import (
    WORKSPACE_REPOSITORY_PARAMETER,
    resolve_workspace_argument_values,
)

from tools.workspace.presentation import (
    format_workspace_file_info_result,
    format_workspace_list_result,
    format_workspace_search_result,
)


TOOLS = {
    # ============================================================
    # DIRECTORY / IDENTITY — READ
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

        "condition_fields": [
            "enabled",
            "locked",
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

        "result_formatter":
            format_account_status_result,
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

        "condition_fields": [
            "has_access",
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

        "result_formatter":
            format_access_check_result,
    },

    # ============================================================
    # DIRECTORY / IDENTITY — EXISTING MUTATIONS
    # ============================================================

    "unlock_user": {
        "description": (
            "Unlock one explicitly identified locked account. "
            "This clears account lockout state but does not enable "
            "a disabled account."
        ),

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

        "approval_formatter":
            format_unlock_approval,
    },

    "reset_password": {
        "description": (
            "Reset the password for one explicitly identified "
            "account."
        ),

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

        "approval_formatter":
            format_password_reset_approval,
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

        "policy_resolver":
            evaluate_process_policy,

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

        "result_formatter":
            format_process_result,

        "approval_formatter":
            format_generic_approval,
    },

    # ============================================================
    # WORKSPACE
    # ============================================================

    "workspace_mkdir": {
    "description": (
        "Create exactly one new direct-child DIRECTORY or FOLDER "
        "inside the approved developer workspace. "
        "Use this only when the user explicitly asks to create, "
        "make, or add a directory/folder. "
        "This capability does NOT perform Git staging, Git add, "
        "file staging, source-control staging, or any other Git "
        "operation."
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
                "Exact new directory/folder name explicitly "
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

    "approval_formatter":
        format_generic_approval,
    },

    "workspace_read_text": {
        "description": (
            "Read one approved UTF-8 text or source file from one "
            "configured logical developer repository. Use "
            "repository for repository scope and relative_path "
            "only for a path inside that repository."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
            "relative_path",
        ],

        "parameters": {
            "repository":
                WORKSPACE_REPOSITORY_PARAMETER,

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Exact path inside the selected repository. "
                    "Do not place a repository identifier here."
                ),
            },
        },

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "result_formatter":
            format_workspace_read_result,
    },

    "workspace_list": {
        "description": (
            "List direct children of a safe directory in one "
            "configured logical developer repository. Use "
            "repository for repository scope. Omit relative_path "
            "to list the selected repository root."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
            "relative_path",
        ],

        "parameters": {
            "repository":
                WORKSPACE_REPOSITORY_PARAMETER,

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional path inside the selected repository. "
                    "Defaults to that repository root. Do not put "
                    "the repository identifier in this field."
                ),
            },
        },

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "result_formatter":
            format_workspace_list_result,
    },

    "workspace_search": {
        "description": (
            "Search approved source/text files recursively inside "
            "one configured logical developer repository. Use "
            "repository for repository scope and relative_path "
            "only for an optional directory inside it."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
            "query",
            "relative_path",
        ],

        "parameters": {
            "repository":
                WORKSPACE_REPOSITORY_PARAMETER,

            "query": {
                "type":
                    "str",

                "description": (
                    "Exact literal source/text phrase supplied "
                    "by the user."
                ),
            },

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Optional path inside the selected repository."
                ),
            },
        },

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "result_formatter":
            format_workspace_search_result,
    },

    "workspace_file_info": {
        "description": (
            "Inspect safe metadata for one file or directory in "
            "a configured logical developer repository without "
            "reading file contents."
        ),

        "risk":
            "read",

        "requires_approval":
            False,

        "grounded_arguments": [
            "repository",
            "relative_path",
        ],

        "parameters": {
            "repository":
                WORKSPACE_REPOSITORY_PARAMETER,

            "relative_path": {
                "type":
                    "str",

                "description": (
                    "Exact path inside the selected repository."
                ),
            },
        },

        "argument_values_resolver":
            resolve_workspace_argument_values,

        "result_formatter":
            format_workspace_file_info_result,
    },
}


TOOLS.update(
    ACCOUNT_TOOLS
)

TOOLS.update(
    ACCESS_TOOLS
)

TOOLS.update(
    ATLASSIAN_TOOLS
)

TOOLS.update(
    ASSET_TOOLS
)

TOOLS.update(
    DEVELOPER_TOOLS
)

TOOLS.update(
    GIT_TOOLS
)

TOOLS.update(
    KNOWLEDGE_TOOLS
)


TOOLS.update(
    JIRA_TOOLS
)

TOOLS.update(
    TICKETING_TOOLS
)


def list_tools(
) -> list[str]:

    return (
        list(
            TOOLS.keys()
        )
    )


def get_tool(
    name: str,
) -> dict | None:

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

    return (
        formatter(
            result
        )
    )


def format_approval_required(
    name: str,
    arguments: dict[
        str,
        Any,
    ],
    approval_id: str | None,
) -> str:

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

    return (
        message
    )
