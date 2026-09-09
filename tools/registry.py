from services.process_runner import (
    evaluate_process_policy,
)

from .account import (
    account_status,
    reset_password,
    unlock_user,
)

from .access import check_access
from .process import process_exec
from .workspace import (
    workspace_git_status,
    workspace_mkdir,
    workspace_read_text,
)


TOOLS = {
    "account_status": {
        "function": account_status,

        "description": (
            "Check whether an account is "
            "enabled or locked."
        ),

        "risk": "read",
        "requires_approval": False,

        "grounded_arguments": [
            "user_id",
        ],

        "parameters": {
            "user_id": {
                "type": "str",

                "description": (
                    "Exact user identifier "
                    "to check."
                ),
            },
        },
    },

    "check_access": {
        "function": check_access,

        "description": (
            "Check whether a user can "
            "access a resource."
        ),

        "risk": "read",
        "requires_approval": False,

        "grounded_arguments": [
            "user_id",
            "resource",
        ],

        "parameters": {
            "user_id": {
                "type": "str",

                "description": (
                    "Exact user identifier "
                    "to check."
                ),
            },

            "resource": {
                "type": "str",

                "description": (
                    "Exact resource identifier "
                    "or name."
                ),
            },
        },
    },

    "unlock_user": {
        "function": unlock_user,

        "description": (
            "Unlock an account."
        ),

        "risk": "low",
        "requires_approval": True,

        "grounded_arguments": [
            "user_id",
        ],

        "parameters": {
            "user_id": {
                "type": "str",

                "description": (
                    "Exact user identifier "
                    "to unlock."
                ),
            },
        },
    },

    "reset_password": {
        "function": reset_password,

        "description": (
            "Reset an account password."
        ),

        "risk": "high",
        "requires_approval": True,

        "grounded_arguments": [
            "user_id",
        ],

        "parameters": {
            "user_id": {
                "type": "str",

                "description": (
                    "Exact user identifier."
                ),
            },
        },
    },

    "process_exec": {
        "function": process_exec,

        "description": (
            "Execute an approved native process "
            "inside the configured workspace."
        ),

        "risk": "low",
        "requires_approval": True,

        "policy_resolver": (
            evaluate_process_policy
        ),

        # Not grounded here because requests such as
        # "What is the current working directory?"
        # legitimately map to executable="pwd" even though
        # the literal word "pwd" was not supplied by the user.
        "grounded_arguments": [],

        "parameters": {
            "executable": {
                "type": "str",

                "description": (
                    "Approved native executable."
                ),
            },

            "args": {
                "type": "list[str]",

                "description": (
                    "Structured executable "
                    "arguments."
                ),
            },

            "cwd": {
                "type": "str",

                "description": (
                    "Optional working directory "
                    "inside the approved workspace."
                ),
            },

            "timeout_seconds": {
                "type": "int",

                "description": (
                    "Execution timeout from "
                    "1 to 30 seconds."
                ),
            },
        },
    },

    "workspace_mkdir": {
        "function": workspace_mkdir,

        "description": (
            "Create exactly one direct-child "
            "directory inside the approved "
            "developer workspace."
        ),

        "risk": "low",
        "requires_approval": True,

        "grounded_arguments": [
            "directory_name",
        ],

        "parameters": {
            "directory_name": {
                "type": "str",

                "description": (
                    "Exact directory name "
                    "explicitly requested by "
                    "the user."
                ),
            },

            "cwd": {
                "type": "str",

                "description": (
                    "Optional working directory "
                    "inside the approved workspace."
                ),
            },

            "timeout_seconds": {
                "type": "int",

                "description": (
                    "Execution timeout from "
                    "1 to 30 seconds."
                ),
            },
        },
    },

    "workspace_read_text": {
        "function": workspace_read_text,

        "description": (
            "Read one approved UTF-8 text or source "
            "file from the developer workspace."
        ),

        "risk": "read",
        "requires_approval": False,

        # A model must not invent a file path.
        # The requested relative path must appear literally
        # in the original user request.
        "grounded_arguments": [
            "relative_path",
        ],

        "parameters": {
            "relative_path": {
                "type": "str",

                "description": (
                    "Exact workspace-relative file path "
                    "explicitly supplied by the user."
                ),
            },
        },
    },

    "workspace_git_status": {
        "function": workspace_git_status,

        "description": (
            "Inspect the current Git branch and "
            "working-tree status of the approved "
            "developer workspace."
        ),

        "risk": "read",
        "requires_approval": False,

        "grounded_arguments": [],

        # V1 deliberately exposes no Git arguments or cwd.
        # Trusted code owns the exact command:
        # git status --short --branch
        "parameters": {},
    },
}


def list_tools() -> list[str]:
    return list(
        TOOLS.keys()
    )


def get_tool(
    name: str,
) -> dict | None:
    return TOOLS.get(
        name
    )


def execute_tool(
    name: str,
    arguments: dict,
    allow_mutation: bool = False,
) -> dict:
    tool = get_tool(
        name
    )

    if tool is None:
        return {
            "ok": False,
            "status": "error",

            "error": (
                f"Unknown tool: {name}"
            ),
        }

    requires_approval = tool[
        "requires_approval"
    ]

    policy_resolver = tool.get(
        "policy_resolver"
    )

    if policy_resolver is not None:
        try:
            policy_result = (
                policy_resolver(
                    **arguments
                )
            )

        except TypeError as exc:
            return {
                "ok": False,
                "status": "denied",

                "error": (
                    "Invalid policy arguments "
                    f"for '{name}': {exc}"
                ),
            }

        except Exception as exc:
            return {
                "ok": False,
                "status": "error",

                "error": (
                    f"Policy evaluation for "
                    f"'{name}' failed: {exc}"
                ),
            }

        if not isinstance(
            policy_result,
            dict,
        ):
            return {
                "ok": False,
                "status": "error",

                "error": (
                    f"Policy for '{name}' returned "
                    "an invalid result."
                ),
            }

        if not policy_result.get(
            "ok",
            False,
        ):
            return policy_result

        dynamic_requires_approval = (
            policy_result.get(
                "requires_approval"
            )
        )

        if not isinstance(
            dynamic_requires_approval,
            bool,
        ):
            return {
                "ok": False,
                "status": "error",

                "error": (
                    f"Policy for '{name}' did not "
                    "return a valid approval decision."
                ),
            }

        requires_approval = (
            dynamic_requires_approval
        )

    if (
        requires_approval
        and not allow_mutation
    ):
        return {
            "ok": False,
            "status": "blocked",

            "error": (
                f"Tool '{name}' requires approval."
            ),
        }

    try:
        result = tool[
            "function"
        ](
            **arguments
        )

    except TypeError as exc:
        return {
            "ok": False,
            "status": "error",

            "error": (
                "Invalid arguments for "
                f"'{name}': {exc}"
            ),
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": "error",

            "error": (
                f"Tool '{name}' failed: "
                f"{exc}"
            ),
        }

    if not isinstance(
        result,
        dict,
    ):
        return {
            "ok": True,
            "result": result,
        }

    return result