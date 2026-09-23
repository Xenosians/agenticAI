from tools.account.policy import (
    evaluate_account_create_policy,
)

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

def format_account_create_result(
    result: dict[str, Any],
) -> str:
    user_id = result.get("user_id")
    email = result.get("email")
    credential_id = result.get("credential_id")

    if result.get("ok") is True and isinstance(user_id, str):
        message = f"Corporate account {user_id} was created"
        if isinstance(email, str):
            message += f" with email {email}"
        if isinstance(credential_id, str):
            message += ". The temporary credential is encrypted in SurrealDB"
        return message + "."

    error = result.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    return "The account creation operation did not complete successfully."


def format_account_create_approval(
    arguments: dict[str, Any],
) -> str:
    username = arguments.get("expected_username")
    email = arguments.get("expected_email")
    if isinstance(username, str) and isinstance(email, str):
        return (
            f"Creating corporate account {username} ({email}), generating a temporary "
            "credential, and persisting the encrypted credential requires approval."
        )
    return "Creating the corporate account requires approval."



ACCOUNT_TOOLS = {
    "account_create": {
        "description": (
            "Create one corporate directory account from explicitly supplied business "
            "identity fields. Trusted policy derives the username, corporate email, "
            "directory container, and identity-policy version. The temporary password "
            "is generated only after approval by trusted code and is never returned "
            "to the model. Phoenix encrypts and persists the credential in SurrealDB."
        ),
        "risk": "high",
        "requires_approval": True,
        "policy_resolver": evaluate_account_create_policy,
        "trusted_policy_arguments": [
            "expected_username",
            "expected_email",
            "expected_container_dn",
            "identity_policy_version",
        ],
        "grounded_arguments": [
            "given_name",
            "family_name",
            "department",
            "role",
        ],
        "parameters": {
            "given_name": {
                "type": "str",
                "description": "Exact given name supplied by the user.",
            },
            "family_name": {
                "type": "str",
                "description": "Exact family name supplied by the user.",
            },
            "department": {
                "type": "str",
                "description": "Optional exact department supplied by the user.",
            },
            "role": {
                "type": "str",
                "description": "Optional exact job role supplied by the user.",
            },
        },
        "result_formatter": format_account_create_result,
        "approval_formatter": format_account_create_approval,
    },

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
