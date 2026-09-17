from typing import Any

from tools.presentation.result_cards import (
    build_result_card,
    list_section,
    result_field,
    text_section,
)


MAX_PRESENTED_PROCESSES = 50


def _process_lines(
    processes: list,
) -> list[str]:
    lines = []

    for process in (
        processes[
            :MAX_PRESENTED_PROCESSES
        ]
    ):
        if not isinstance(
            process,
            dict,
        ):
            continue

        lines.append(
            (
                f"PID {process.get('pid')}: "
                f"{process.get('executable')} "
                f"[state={process.get('state')}, "
                f"elapsed="
                f"{process.get('elapsed_seconds')}s]"
            )
        )

    return lines


def format_process_snapshot_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    processes = (
        result.get(
            "processes"
        )
    )

    if not isinstance(
        processes,
        list,
    ):
        return (
            "Process inspection completed, "
            "but no valid process list was returned."
        )

    if not processes:
        return (
            "No host processes were returned."
        )

    lines = [
        f"- {line}"
        for line in (
            _process_lines(
                processes
            )
        )
    ]

    total_count = result.get(
        "count",
        len(processes),
    )

    truncated = result.get(
        "truncated",
        False,
    )

    footer = ""

    if (
        truncated
        or len(processes)
        > MAX_PRESENTED_PROCESSES
    ):
        footer = (
            "\n- ... additional processes omitted"
        )

    return (
        f"Host process snapshot "
        f"({total_count} returned):\n"
        + "\n".join(
            lines
        )
        + footer
    )


def build_process_snapshot_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    processes = (
        result.get(
            "processes"
        )
    )

    process_items = (
        _process_lines(
            processes
        )
        if isinstance(
            processes,
            list,
        )
        else []
    )

    return (
        build_result_card(
            kind=(
                "process_snapshot"
            ),

            title=(
                "Host processes"
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
                    "Provider",
                    result.get(
                        "provider"
                    ),
                ),

                result_field(
                    "Processes",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "Truncated",
                    result.get(
                        "truncated"
                    ),
                ),
            ],

            sections=[
                list_section(
                    title=(
                        "Processes"
                    ),
                    items=(
                        process_items
                    ),
                ),
            ],
        )
    )


def _service_lines(
    services: list,
) -> list[str]:
    lines = []

    for service in services:
        if not isinstance(
            service,
            dict,
        ):
            continue

        details = [
            (
                "state="
                f"{service.get('state')}"
            )
        ]

        health = (
            service.get(
                "health"
            )
        )

        status = (
            service.get(
                "status"
            )
        )

        if health:
            details.append(
                f"health={health}"
            )

        if status:
            details.append(
                f"status={status}"
            )

        lines.append(
            (
                f"{service.get('service')}: "
                + ", ".join(
                    details
                )
            )
        )

    return lines


def format_service_status_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    services = (
        result.get(
            "services"
        )
    )

    if not isinstance(
        services,
        list,
    ):
        return (
            "Service inspection completed, "
            "but no valid service list was returned."
        )

    if not services:
        return (
            "No workspace services were returned."
        )

    lines = [
        f"- {line}"

        for line
        in _service_lines(
            services
        )
    ]

    return (
        "Workspace service status:\n"
        + "\n".join(
            lines
        )
    )


def build_service_status_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    services = (
        result.get(
            "services"
        )
    )

    service_items = (
        _service_lines(
            services
        )
        if isinstance(
            services,
            list,
        )
        else []
    )

    return (
        build_result_card(
            kind=(
                "service_status"
            ),

            title=(
                "Workspace services"
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
                    "Provider",
                    result.get(
                        "provider"
                    ),
                ),

                result_field(
                    "Services",
                    result.get(
                        "service_count"
                    ),
                ),
            ],

            sections=[
                list_section(
                    title=(
                        "Services"
                    ),
                    items=(
                        service_items
                    ),
                ),
            ],
        )
    )


def format_service_logs_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    service_name = (
        result.get(
            "service_name"
        )
    )

    logs = (
        result.get(
            "logs"
        )
    )

    tail_lines = (
        result.get(
            "tail_lines"
        )
    )

    redaction_applied = (
        result.get(
            "redaction_applied",
            False,
        )
    )

    if not isinstance(
        logs,
        str,
    ):
        return (
            "Service logs were retrieved, "
            "but no valid log text was returned."
        )

    if not logs.strip():
        return (
            f"No recent logs were returned "
            f"for service {service_name}."
        )

    redaction_note = ""

    if redaction_applied:
        redaction_note = (
            "\n\n"
            "Common credential/token patterns "
            "were redacted from this output."
        )

    return (
        f"Last {tail_lines} log lines "
        f"for service {service_name}:\n\n"
        f"{logs.strip()}"
        f"{redaction_note}"
    )


def build_service_logs_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    logs = (
        result.get(
            "logs"
        )
    )

    return (
        build_result_card(
            kind=(
                "service_logs"
            ),

            title=(
                (
                    "Service logs · "
                    f"{result.get('service_name')}"
                )
                if result.get(
                    "service_name"
                )
                else "Service logs"
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
                    "Service",
                    result.get(
                        "service_name"
                    ),
                ),

                result_field(
                    "Provider",
                    result.get(
                        "provider"
                    ),
                ),

                result_field(
                    "Tail lines",
                    result.get(
                        "tail_lines"
                    ),
                ),

                result_field(
                    "Redaction applied",
                    result.get(
                        "redaction_applied"
                    ),
                ),
            ],

            sections=[
                text_section(
                    title=(
                        "Recent logs"
                    ),
                    content=(
                        logs
                    ),
                    kind=(
                        "preformatted"
                    ),
                )
                if isinstance(
                    logs,
                    str,
                )
                else None,
            ],
        )
    )