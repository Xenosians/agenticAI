from typing import Any


def format_workspace_list_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    path = result.get(
        "path",
        ".",
    )

    entries = result.get(
        "entries"
    )

    truncated = result.get(
        "truncated",
        False,
    )

    if not isinstance(
        entries,
        list,
    ):
        return (
            "Workspace listing completed, "
            "but no valid entry list was returned."
        )

    if not entries:
        return (
            f"Workspace directory {path} is empty."
        )

    lines = []

    for entry in entries:
        if not isinstance(
            entry,
            dict,
        ):
            continue

        name = entry.get(
            "name"
        )

        entry_type = entry.get(
            "type"
        )

        if not isinstance(
            name,
            str,
        ):
            continue

        if entry_type == "directory":
            label = (
                "directory"
            )

        elif entry_type == "file":
            label = (
                "file"
            )

        elif entry_type == "symlink":
            label = (
                "symlink"
            )

        else:
            label = (
                "other"
            )

        lines.append(
            f"- {name} [{label}]"
        )

    if truncated:
        lines.append(
            "- ... additional entries omitted"
        )

    return (
        f"Workspace directory {path}:\n"
        + "\n".join(
            lines
        )
    )


def format_workspace_search_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    query = result.get(
        "query"
    )

    matches = result.get(
        "matches"
    )

    truncated = result.get(
        "truncated",
        False,
    )

    if not isinstance(
        matches,
        list,
    ):
        return (
            "Workspace search completed, "
            "but no valid match list was returned."
        )

    if not matches:
        return (
            f"No workspace source/text matches were found "
            f"for '{query}'."
        )

    lines = []

    for match in matches:
        if not isinstance(
            match,
            dict,
        ):
            continue

        path = match.get(
            "path"
        )

        line_number = match.get(
            "line_number"
        )

        line = match.get(
            "line"
        )

        if (
            not isinstance(
                path,
                str,
            )
            or not isinstance(
                line_number,
                int,
            )
            or not isinstance(
                line,
                str,
            )
        ):
            continue

        lines.append(
            f"- {path}:{line_number}: {line}"
        )

    if truncated:
        lines.append(
            "- ... additional matches omitted"
        )

    return (
        f"Workspace matches for '{query}':\n"
        + "\n".join(
            lines
        )
    )


def format_workspace_file_info_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    path = result.get(
        "path"
    )

    entry_type = result.get(
        "type"
    )

    size_bytes = result.get(
        "size_bytes"
    )

    readable_text = result.get(
        "readable_text"
    )

    if not isinstance(
        path,
        str,
    ):
        return (
            "Workspace metadata was retrieved, "
            "but no valid path was returned."
        )

    lines = [
        f"Workspace path: {path}",
        f"- Type: {entry_type}",
        f"- Size: {size_bytes} bytes",
    ]

    if entry_type == "file":
        lines.append(
            "- Readable as approved text: "
            + (
                "yes"
                if readable_text is True
                else "no"
            )
        )

    return "\n".join(
        lines
    )