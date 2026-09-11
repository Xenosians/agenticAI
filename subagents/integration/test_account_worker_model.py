import json

import pytest

from config import (
    get_settings,
)

from subagents.llm.qwen_funcall import (
    QwenFuncCallBackend,
)


ACCOUNT_TOOLS = [
    {
        "name":
            "account_status",

        "description": (
            "Check whether a user account exists, "
            "is enabled, and whether it is locked."
        ),

        "parameters": {
            "user_id": {
                "description":
                    "Exact user identifier to check.",

                "type":
                    "str",
            }
        },
    },

    {
        "name":
            "unlock_user",

        "description":
            "Unlock a locked user account.",

        "parameters": {
            "user_id": {
                "description":
                    "Exact user identifier to unlock.",

                "type":
                    "str",
            }
        },
    },

    {
        "name":
            "reset_password",

        "description": (
            "Request a password reset "
            "for a user account."
        ),

        "parameters": {
            "user_id": {
                "description":
                    "Exact user identifier.",

                "type":
                    "str",
            }
        },
    },
]


def build_system_prompt() -> str:
    return (
        "You are a function-calling assistant. "
        "Given a user query and a list of available tools, "
        "respond with ONLY a JSON array of the function call(s) "
        "needed to fulfill the query. "
        "Each item must have 'name' and 'arguments' keys. "
        "Do not include explanation, markdown, or any text "
        "outside the raw JSON array.\n\n"
        "Available tools:\n"
        f"{json.dumps(ACCOUNT_TOOLS, indent=2)}"
    )


@pytest.fixture(
    scope="module"
)
def backend():
    settings = get_settings()

    model_path = (
        settings.require_path(
            settings.account_model_path,
            "ACCOUNT_MODEL_PATH",
        )
    )

    return QwenFuncCallBackend(
        model_path
    )


def generate_tool_call(
    backend,
    user_request: str,
) -> list[dict]:
    messages = [
        {
            "role":
                "system",

            "content":
                build_system_prompt(),
        },

        {
            "role":
                "user",

            "content":
                user_request,
        },
    ]

    response = (
        backend.generate(
            messages,
            max_new_tokens=128,
        )
    )

    print(
        f"\nREQUEST: {user_request}"
        f"\nRAW MODEL OUTPUT: {response}"
    )

    parsed = json.loads(
        response
    )

    assert isinstance(
        parsed,
        list,
    )

    assert (
        len(parsed)
        >= 1
    )

    return parsed


def test_account_status_selection(
    backend,
):
    calls = generate_tool_call(
        backend,
        "Is jdoe locked?",
    )

    call = calls[0]

    assert (
        call["name"]
        == "account_status"
    )

    assert (
        call["arguments"][
            "user_id"
        ]
        == "jdoe"
    )


def test_unlock_selection(
    backend,
):
    calls = generate_tool_call(
        backend,
        "Unlock jdoe",
    )

    call = calls[0]

    assert (
        call["name"]
        == "unlock_user"
    )

    assert (
        call["arguments"][
            "user_id"
        ]
        == "jdoe"
    )


def test_password_reset_selection(
    backend,
):
    calls = generate_tool_call(
        backend,
        "Reset the password for jdoe",
    )

    call = calls[0]

    assert (
        call["name"]
        == "reset_password"
    )

    assert (
        call["arguments"][
            "user_id"
        ]
        == "jdoe"
    )