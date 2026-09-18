from typing import (
    Any,
)


def format_account_lifecycle_result(
    result: dict[
        str,
        Any,
    ],
) -> str:

    user_id = (
        result.get(
            "user_id"
        )
    )

    enabled = (
        result.get(
            "enabled"
        )
    )

    changed = (
        result.get(
            "changed"
        )
    )

    if (
        isinstance(
            user_id,
            str,
        )
        and user_id.strip()
    ):

        if enabled is True:

            if changed:

                return (
                    f"{user_id} is now enabled."
                )

            return (
                f"{user_id} was already enabled."
            )

        if enabled is False:

            if changed:

                return (
                    f"{user_id} is now disabled."
                )

            return (
                f"{user_id} was already disabled."
            )

    message = (
        result.get(
            "message"
        )
    )

    if (
        isinstance(
            message,
            str,
        )
        and message.strip()
    ):

        return (
            message.strip()
        )

    return (
        "The account lifecycle operation "
        "completed successfully."
    )


def format_enable_user_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    user_id = (
        arguments.get(
            "user_id"
        )
    )

    if (
        isinstance(
            user_id,
            str,
        )
        and user_id.strip()
    ):

        return (
            f"Enabling {user_id}'s account "
            "requires approval."
        )

    return (
        "Enabling the account requires approval."
    )


def format_disable_user_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    user_id = (
        arguments.get(
            "user_id"
        )
    )

    if (
        isinstance(
            user_id,
            str,
        )
        and user_id.strip()
    ):

        return (
            f"Disabling {user_id}'s account "
            "requires approval."
        )

    return (
        "Disabling the account requires approval."
    )


ACCOUNT_TOOLS = {
    "enable_user": {
        "description": (
            "Enable one explicitly identified user account. "
            "This changes account lifecycle state and requires "
            "approval. Use only when the user explicitly asks "
            "to enable, activate, or re-enable the account. "
            "This capability does not unlock a locked account."
        ),

        "risk":
            "medium",

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
                    "Exact account identifier supplied by "
                    "the user."
                ),
            },
        },

        "result_formatter":
            format_account_lifecycle_result,

        "approval_formatter":
            format_enable_user_approval,
    },

    "disable_user": {
        "description": (
            "Disable one explicitly identified user account. "
            "This changes account lifecycle state and requires "
            "approval. Use only when the user explicitly asks "
            "to disable, deactivate, or suspend the account."
        ),

        "risk":
            "medium",

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
                    "Exact account identifier supplied by "
                    "the user."
                ),
            },
        },

        "result_formatter":
            format_account_lifecycle_result,

        "approval_formatter":
            format_disable_user_approval,
    },
}
