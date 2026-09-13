from typing import (
    Any,
)


def format_account_status_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    user_id = (
        result.get(
            "user_id",
            "The account",
        )
    )

    enabled = (
        result.get(
            "enabled"
        )
    )

    locked = (
        result.get(
            "locked"
        )
    )

    if (
        enabled is True
        and locked is True
    ):
        return (
            f"{user_id} is enabled, "
            "but the account is currently locked."
        )

    if (
        enabled is True
        and locked is False
    ):
        return (
            f"{user_id} is enabled "
            "and is not locked."
        )

    if (
        enabled is False
        and locked is True
    ):
        return (
            f"{user_id} is disabled "
            "and is currently locked."
        )

    if (
        enabled is False
        and locked is False
    ):
        return (
            f"{user_id} is disabled "
            "and is not locked."
        )

    return (
        "I retrieved the account status for "
        f"{user_id}, but the directory did not "
        "return a complete enabled/locked state."
    )


def format_access_check_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    user_id = (
        result.get(
            "user_id",
            "The user",
        )
    )

    resource = (
        result.get(
            "resource",
            "the requested resource",
        )
    )

    has_access = (
        result.get(
            "has_access"
        )
    )

    if (
        has_access is True
    ):
        return (
            f"{user_id} has access to "
            f"{resource}."
        )

    if (
        has_access is False
    ):
        return (
            f"{user_id} does not have access to "
            f"{resource}."
        )

    return (
        f"I checked {user_id}'s access to "
        f"{resource}, but the directory did not "
        "return a definitive access state."
    )


def format_process_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    executable = (
        result.get(
            "executable"
        )
    )

    stdout = (
        result.get(
            "stdout"
        )
    )

    if (
        executable == "pwd"
        and isinstance(
            stdout,
            str,
        )
    ):
        working_directory = (
            stdout.strip()
        )

        if working_directory:
            return (
                "The current working directory is "
                f"{working_directory}."
            )

    if (
        executable == "ls"
        and isinstance(
            stdout,
            str,
        )
    ):
        entries = [
            line.strip()

            for line
            in stdout.splitlines()

            if line.strip()
        ]

        if not entries:
            return (
                "The current workspace is empty."
            )

        formatted_entries = (
            "\n".join(
                f"- {entry}"

                for entry
                in entries
            )
        )

        return (
            "The current workspace contains:\n"
            f"{formatted_entries}"
        )

    return (
        "The approved local process "
        "completed successfully."
    )


def format_workspace_read_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    path = (
        result.get(
            "path"
        )
    )

    content = (
        result.get(
            "content"
        )
    )

    truncated = (
        result.get(
            "truncated",
            False,
        )
    )

    if (
        isinstance(
            path,
            str,
        )
        and isinstance(
            content,
            str,
        )
    ):
        if truncated:
            return (
                f"Contents of {path} "
                "(truncated to the allowed "
                "read limit):\n\n"
                f"{content}"
            )

        return (
            f"Contents of {path}:\n\n"
            f"{content}"
        )

    return (
        "The workspace file was read successfully, "
        "but the result did not contain valid text."
    )


def format_git_status_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    stdout = (
        result.get(
            "stdout"
        )
    )

    if not isinstance(
        stdout,
        str,
    ):
        return (
            "Git status completed successfully, "
            "but no readable status output was returned."
        )

    lines = [
        line

        for line
        in stdout.splitlines()

        if line.strip()
    ]

    if not lines:
        return (
            "Git status completed successfully, "
            "but no branch information was returned."
        )

    branch_line = (
        lines[0]
    )

    if branch_line.startswith(
        "## "
    ):
        branch_status = (
            branch_line[3:]
            .strip()
        )

    else:
        branch_status = (
            branch_line
            .strip()
        )

    changes = (
        lines[1:]
    )

    if not changes:
        return (
            "Git status:\n"
            f"- Branch: {branch_status}\n"
            "- Working tree: clean"
        )

    formatted_changes = (
        "\n".join(
            f"- {change}"

            for change
            in changes
        )
    )

    return (
        "Git status:\n"
        f"- Branch: {branch_status}\n"
        "- Working tree changes:\n"
        f"{formatted_changes}"
    )


def format_unlock_approval(
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

    if user_id:
        return (
            f"Unlocking {user_id} "
            "requires approval."
        )

    return (
        "The account unlock "
        "requires approval."
    )


def format_password_reset_approval(
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

    if user_id:
        return (
            f"Resetting {user_id}'s "
            "password requires approval."
        )

    return (
        "The password reset "
        "requires approval."
    )


def format_generic_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    return (
        "This action requires approval."
    )