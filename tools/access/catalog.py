from typing import (
    Any,
)


def format_access_mutation_result(
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

    resource = (
        result.get(
            "resource"
        )
    )

    has_access = (
        result.get(
            "has_access"
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
        and isinstance(
            resource,
            str,
        )
    ):

        if (
            has_access is True
        ):

            if changed:

                return (
                    f"{user_id} now has access "
                    f"to {resource}."
                )

            return (
                f"{user_id} already has access "
                f"to {resource}."
            )

        if (
            has_access is False
        ):

            if changed:

                return (
                    f"{user_id}'s access to "
                    f"{resource} was revoked."
                )

            return (
                f"{user_id} already does not have "
                f"access to {resource}."
            )

    message = (
        result.get(
            "message"
        )
    )

    if isinstance(
        message,
        str,
    ) and message.strip():

        return (
            message.strip()
        )

    return (
        "The access-management operation "
        "completed successfully."
    )


def _approval(
    arguments: dict[
        str,
        Any,
    ],
    *,
    operation: str,
) -> str:

    user_id = (
        arguments.get(
            "user_id"
        )
    )

    resource = (
        arguments.get(
            "resource"
        )
    )

    if (
        isinstance(
            user_id,
            str,
        )
        and user_id.strip()
        and isinstance(
            resource,
            str,
        )
        and resource.strip()
    ):

        return (
            f"{operation} {user_id}'s access "
            f"to {resource} requires approval."
        )

    return (
        f"{operation} access requires approval."
    )


def format_grant_access_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _approval(
            arguments,
            operation="Granting",
        )
    )


def format_revoke_access_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:

    return (
        _approval(
            arguments,
            operation="Revoking",
        )
    )


ACCESS_TOOLS = {
    "grant_access": {
        "description": (
            "Grant one explicitly identified user access to one "
            "explicitly identified resource. This changes access "
            "state and requires approval. Use only when the user "
            "explicitly requests granting or adding access."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments": [
            "user_id",
            "resource",
        ],

        "parameters": {
            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier supplied by the user."
                ),
            },

            "resource": {
                "type":
                    "str",

                "description": (
                    "Exact logical resource name or identifier "
                    "supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_access_mutation_result,

        "approval_formatter":
            format_grant_access_approval,
    },

    "revoke_access": {
        "description": (
            "Revoke one explicitly identified user's access to one "
            "explicitly identified resource. This changes access "
            "state and requires approval. Use only when the user "
            "explicitly requests revoking or removing access."
        ),

        "risk":
            "medium",

        "requires_approval":
            True,

        "grounded_arguments": [
            "user_id",
            "resource",
        ],

        "parameters": {
            "user_id": {
                "type":
                    "str",

                "description": (
                    "Exact user identifier supplied by the user."
                ),
            },

            "resource": {
                "type":
                    "str",

                "description": (
                    "Exact logical resource name or identifier "
                    "supplied by the user."
                ),
            },
        },

        "result_formatter":
            format_access_mutation_result,

        "approval_formatter":
            format_revoke_access_approval,
    },
}
