from .account import (
    account_status,
    reset_password,
    unlock_user,
)
from .access import check_access
from .process import process_exec


TOOLS = {
    "account_status": {
        "function": account_status,
        "description": (
            "Check whether an account is enabled or locked."
        ),
        "risk": "read",
        "requires_approval": False,
        "parameters": {
            "user_id": {
                "type": "str",
                "description": (
                    "Exact user identifier to check."
                ),
            },
        },
    },

    "check_access": {
        "function": check_access,
        "description": (
            "Check whether a user can access a resource."
        ),
        "risk": "read",
        "requires_approval": False,
        "parameters": {
            "user_id": {
                "type": "str",
                "description": (
                    "Exact user identifier to check."
                ),
            },
            "resource": {
                "type": "str",
                "description": (
                    "Exact resource identifier or name."
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
        "parameters": {
            "user_id": {
                "type": "str",
                "description": (
                    "Exact user identifier to unlock."
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
            "Execute an approved native process inside the "
            "configured workspace. The executable must pass "
            "the deterministic process-runner policy."
        ),
        "risk": "read",
        "requires_approval": False,
        "parameters": {
            "executable": {
                "type": "str",
                "description": (
                    "Executable name. Currently only 'pwd' "
                    "is allowed by policy."
                ),
            },
            "args": {
                "type": "list[str]",
                "description": (
                    "Structured executable arguments. "
                    "Currently pwd accepts no arguments."
                ),
            },
            "cwd": {
                "type": "str",
                "description": (
                    "Optional working directory inside the "
                    "approved workspace."
                ),
            },
            "timeout_seconds": {
                "type": "int",
                "description": (
                    "Execution timeout from 1 to 30 seconds."
                ),
            },
        },
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

    if (
        tool["requires_approval"]
        and not allow_mutation
    ):
        return {
            "ok": False,
            "status": "blocked",
            "error": (
                f"Tool '{name}' is mutating "
                "and requires approval."
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
                f"Invalid arguments for "
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