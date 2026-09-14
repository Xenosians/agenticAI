from typing import Any


def format_project_info_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    project_type = (
        result.get(
            "project_type"
        )
    )

    path = (
        result.get(
            "path",
            ".",
        )
    )

    markers = (
        result.get(
            "markers",
            [],
        )
    )

    if not isinstance(
        project_type,
        str,
    ):
        return (
            "Project detection completed, "
            "but no supported project type "
            "was returned."
        )

    marker_lines = "\n".join(
        f"- {marker}"
        for marker
        in markers
        if isinstance(
            marker,
            str,
        )
    )

    if not marker_lines:
        marker_lines = (
            "- no marker details returned"
        )

    return (
        f"Detected {project_type} project "
        f"at {path}.\n"
        "Project markers:\n"
        f"{marker_lines}\n"
        "Supported governed operations:\n"
        "- tests\n"
        "- build"
    )


def format_developer_execution_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    operation = (
        result.get(
            "operation"
        )
    )

    project_type = (
        result.get(
            "project_type"
        )
    )

    project_path = (
        result.get(
            "project_path",
            ".",
        )
    )

    exit_code = (
        result.get(
            "exit_code"
        )
    )

    duration_ms = (
        result.get(
            "duration_ms"
        )
    )

    stdout = (
        result.get(
            "stdout"
        )
    )

    stderr = (
        result.get(
            "stderr"
        )
    )

    lines = [
        (
            f"Developer {operation} completed "
            f"for the {project_type} project "
            f"at {project_path}."
        ),
        f"- Exit code: {exit_code}",
        f"- Duration: {duration_ms} ms",
    ]

    if (
        isinstance(
            stdout,
            str,
        )
        and stdout.strip()
    ):
        lines.extend(
            [
                "",
                "Output:",
                stdout.strip(),
            ]
        )

    if (
        isinstance(
            stderr,
            str,
        )
        and stderr.strip()
    ):
        lines.extend(
            [
                "",
                "Error output:",
                stderr.strip(),
            ]
        )

    return "\n".join(
        lines
    )


def format_developer_execution_approval(
    arguments: dict[
        str,
        Any,
    ],
) -> str:
    relative_path = (
        arguments.get(
            "relative_path",
            ".",
        )
    )

    return (
        "Executing repository test/build code "
        f"for workspace project '{relative_path}' "
        "requires approval."
    )