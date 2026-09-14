from typing import (
    Any,
)

from tools.result_cards import (
    build_result_card,
    result_field,
)


def format_ticket_get_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    ticket = (
        result.get(
            "ticket"
        )
    )

    if not isinstance(
        ticket,
        dict,
    ):
        return (
            "Ticket lookup completed, "
            "but no valid ticket was returned."
        )

    key = (
        ticket.get(
            "key"
        )
    )

    summary = (
        ticket.get(
            "summary"
        )
    )

    status = (
        ticket.get(
            "status"
        )
    )

    if not isinstance(
        key,
        str,
    ):
        key = "Ticket"

    if not isinstance(
        summary,
        str,
    ):
        summary = (
            "No summary returned."
        )

    if not isinstance(
        status,
        str,
    ):
        status = (
            "Unknown"
        )

    return (
        f"{key}: {summary}\n"
        f"Status: {status}"
    )


def build_ticket_get_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    ticket = (
        result.get(
            "ticket"
        )
    )

    if not isinstance(
        ticket,
        dict,
    ):
        ticket = {}

    key = (
        ticket.get(
            "key"
        )
    )

    title = (
        key.strip()
        if isinstance(
            key,
            str,
        )
        and key.strip()
        else "Ticket"
    )

    return (
        build_result_card(
            kind=(
                "ticket"
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
                    "Summary",
                    ticket.get(
                        "summary"
                    ),
                ),

                result_field(
                    "Status",
                    ticket.get(
                        "status"
                    ),
                ),

                result_field(
                    "Type",
                    ticket.get(
                        "ticket_type"
                    ),
                ),

                result_field(
                    "Priority",
                    ticket.get(
                        "priority"
                    ),
                ),

                result_field(
                    "Assignee",
                    ticket.get(
                        "assignee"
                    ),
                ),

                result_field(
                    "Reporter",
                    ticket.get(
                        "reporter"
                    ),
                ),

                result_field(
                    "Project",
                    ticket.get(
                        "project_name"
                    ),
                ),

                result_field(
                    "Project key",
                    ticket.get(
                        "project_key"
                    ),
                ),

                result_field(
                    "Created",
                    ticket.get(
                        "created_at"
                    ),
                ),

                result_field(
                    "Updated",
                    ticket.get(
                        "updated_at"
                    ),
                ),

                result_field(
                    "Provider",
                    ticket.get(
                        "provider"
                    ),
                ),
            ],

            sections=[],
        )
    )