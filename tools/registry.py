from .account import (
    account_status,
    reset_password,
    unlock_user,
)
from .access import check_access


TOOLS = {
    "account_status": {
        "function": account_status,
        "description": "Check whether an account is enabled or locked.",
        "risk": "read",
        "requires_approval": False,
        "parameters": {
            "user_id": {
                "type": "str",
                "description": "Exact user identifier to check.",
            },
        },
    },

    "check_access": {
        "function": check_access,
        "description": "Check whether a user can access a resource.",
        "risk": "read",
        "requires_approval": False,
        "parameters": {
            "user_id": {
                "type": "str",
                "description": "Exact user identifier to check.",
            },
            "resource": {
                "type": "str",
                "description": "Exact resource identifier or name.",
            },
        },
    },

    "unlock_user": {
        "function": unlock_user,
        "description": "Unlock an account.",
        "risk": "low",
        "requires_approval": True,
        "parameters": {
            "user_id": {
                "type": "str",
                "description": "Exact user identifier to unlock.",
            },
        },
    },

    "reset_password": {
        "function": reset_password,
        "description": "Reset an account password.",
        "risk": "high",
        "requires_approval": True,
        "parameters": {
            "user_id": {
                "type": "str",
                "description": "Exact user identifier.",
            },
        },
    },
}


def list_tools() -> list[str]:
    return list(TOOLS.keys())


def get_tool(name: str) -> dict | None:
    return TOOLS.get(name)


def execute_tool(
    name: str,
    arguments: dict,
    allow_mutation: bool = False,
) -> dict:
    tool = get_tool(name)

    if tool is None:
        return {
            "ok": False,
            "status": "error",
            "error": f"Unknown tool: {name}",
        }

    if tool["requires_approval"] and not allow_mutation:
        return {
            "ok": False,
            "status": "blocked",
            "error": (
                f"Tool '{name}' is mutating and requires approval."
            ),
        }

    try:
        result = tool["function"](**arguments)

    except TypeError as exc:
        return {
            "ok": False,
            "status": "error",
            "error": f"Invalid arguments for '{name}': {exc}",
        }

    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "error": f"Tool '{name}' failed: {exc}",
        }

    if not isinstance(result, dict):
        return {
            "ok": True,
            "result": result,
        }

    return result