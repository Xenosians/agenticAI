from typing import Any


MAX_PRESENTED_PROCESSES = 50


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

        pid = process.get(
            "pid"
        )

        executable = (
            process.get(
                "executable"
            )
        )

        state = process.get(
            "state"
        )

        elapsed = (
            process.get(
                "elapsed_seconds"
            )
        )

        lines.append(
            (
                f"- PID {pid}: "
                f"{executable} "
                f"[state={state}, "
                f"elapsed={elapsed}s]"
            )
        )

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

    lines = []

    for service in services:
        if not isinstance(
            service,
            dict,
        ):
            continue

        service_name = (
            service.get(
                "service"
            )
        )

        state = (
            service.get(
                "state"
            )
        )

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

        details = [
            f"state={state}"
        ]

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
                f"- {service_name}: "
                + ", ".join(
                    details
                )
            )
        )

    return (
        "Workspace service status:\n"
        + "\n".join(
            lines
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
