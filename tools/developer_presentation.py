from typing import Any

from tools.result_cards import (
    build_result_card,
    list_section,
    result_field,
    text_section,
)


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


def build_project_info_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    markers = (
        result.get(
            "markers"
        )
    )

    marker_items = []

    if isinstance(
        markers,
        list,
    ):
        marker_items = [
            marker

            for marker in markers

            if isinstance(
                marker,
                str,
            )
            and marker.strip()
        ]

    operations = (
        result.get(
            "supported_operations"
        )
    )

    operation_items = []

    if isinstance(
        operations,
        list,
    ):
        operation_items = [
            operation

            for operation
            in operations

            if isinstance(
                operation,
                str,
            )
            and operation.strip()
        ]

    project_type = (
        result.get(
            "project_type"
        )
    )

    title = (
        f"{project_type} project"
        if isinstance(
            project_type,
            str,
        )
        and project_type.strip()
        else "Workspace project"
    )

    return (
        build_result_card(
            kind=(
                "developer_project"
            ),

            title=(
                title
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "success",
                    )
                )
            ),

            fields=[
                result_field(
                    "Project type",
                    project_type,
                ),

                result_field(
                    "Path",
                    result.get(
                        "path"
                    ),
                ),
            ],

            sections=[
                list_section(
                    title=(
                        "Project markers"
                    ),
                    items=(
                        marker_items
                    ),
                ),

                list_section(
                    title=(
                        "Supported operations"
                    ),
                    items=(
                        operation_items
                    ),
                ),
            ],
        )
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


def build_developer_execution_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    operation = (
        result.get(
            "operation"
        )
    )

    if operation == "tests":
        title = "Tests"

    elif operation == "build":
        title = "Build"

    else:
        title = (
            "Developer execution"
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

    return (
        build_result_card(
            kind=(
                "developer_execution"
            ),

            title=(
                title
            ),

            status=(
                str(
                    result.get(
                        "status",
                        "unknown",
                    )
                )
            ),

            fields=[
                result_field(
                    "Project",
                    result.get(
                        "project_type"
                    ),
                ),

                result_field(
                    "Path",
                    result.get(
                        "project_path"
                    ),
                ),

                result_field(
                    "Exit code",
                    result.get(
                        "exit_code"
                    ),
                ),

                result_field(
                    "Duration",
                    (
                        f"{result.get('duration_ms')} ms"
                        if result.get(
                            "duration_ms"
                        )
                        is not None
                        else None
                    ),
                ),

                result_field(
                    "Timed out",
                    result.get(
                        "timed_out"
                    ),
                ),
            ],

            sections=[
                text_section(
                    title="Output",
                    content=stdout,
                    kind=(
                        "preformatted"
                    ),
                )
                if isinstance(
                    stdout,
                    str,
                )
                else None,

                text_section(
                    title=(
                        "Error output"
                    ),
                    content=stderr,
                    kind=(
                        "preformatted"
                    ),
                )
                if isinstance(
                    stderr,
                    str,
                )
                else None,
            ],
        )
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