from typing import (
    Any,
)

from tools.result_cards import (
    build_result_card,
    list_section,
    result_field,
)


# ============================================================
# SINGLE TICKET
# ============================================================


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
            kind="ticket",

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


# ============================================================
# TICKET SEARCH
# ============================================================


def format_ticket_search_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    tickets = (
        result.get(
            "tickets"
        )
    )

    if not isinstance(
        tickets,
        list,
    ):
        return (
            "Ticket search completed, "
            "but no valid result list was returned."
        )

    if not tickets:
        return (
            "No tickets matched the requested filters."
        )

    lines = []

    for ticket in tickets:
        if not isinstance(
            ticket,
            dict,
        ):
            continue

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
            continue

        line = (
            key
        )

        if isinstance(
            summary,
            str,
        ):
            line += (
                f": {summary}"
            )

        if isinstance(
            status,
            str,
        ):
            line += (
                f" [{status}]"
            )

        lines.append(
            line
        )

    if not lines:
        return (
            "Ticket search completed, "
            "but no valid tickets were returned."
        )

    response = (
        "Matching tickets:\n"
        + "\n".join(
            lines
        )
    )

    if (
        result.get(
            "truncated"
        )
        is True
    ):
        response += (
            "\nAdditional matching tickets "
            "were not returned because the "
            "search result is bounded."
        )

    return (
        response
    )


def build_ticket_search_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    raw_tickets = (
        result.get(
            "tickets"
        )
    )

    tickets = (
        raw_tickets
        if isinstance(
            raw_tickets,
            list,
        )
        else []
    )

    items = []
    providers = set()

    for ticket in tickets:
        if not isinstance(
            ticket,
            dict,
        ):
            continue

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

        priority = (
            ticket.get(
                "priority"
            )
        )

        provider = (
            ticket.get(
                "provider"
            )
        )

        if (
            isinstance(
                provider,
                str,
            )
            and provider.strip()
        ):
            providers.add(
                provider.strip()
            )

        if not isinstance(
            key,
            str,
        ):
            continue

        item = (
            key.strip()
        )

        if (
            isinstance(
                summary,
                str,
            )
            and summary.strip()
        ):
            item += (
                f" — {summary.strip()}"
            )

        details = []

        if (
            isinstance(
                status,
                str,
            )
            and status.strip()
        ):
            details.append(
                status.strip()
            )

        if (
            isinstance(
                priority,
                str,
            )
            and priority.strip()
        ):
            details.append(
                priority.strip()
            )

        if details:
            item += (
                " · "
                + " · ".join(
                    details
                )
            )

        items.append(
            item
        )

    provider_value = None

    if len(
        providers
    ) == 1:
        provider_value = (
            next(
                iter(
                    providers
                )
            )
        )

    return (
        build_result_card(
            kind="ticket-list",

            title="Ticket search",

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
                    "Returned",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "More results",
                    result.get(
                        "truncated"
                    ),
                ),

                result_field(
                    "Provider",
                    provider_value,
                ),
            ],

            sections=[
                list_section(
                    title="Tickets",
                    items=items,
                )
            ],
        )
    )


# ============================================================
# TICKET HISTORY
# ============================================================


def _render_change(
    change: dict[
        str,
        Any,
    ],
) -> (
    str | None
):
    field = (
        change.get(
            "field"
        )
    )

    if not isinstance(
        field,
        str,
    ):
        return None

    before = (
        change.get(
            "from_value"
        )
    )

    after = (
        change.get(
            "to_value"
        )
    )

    before_text = (
        str(
            before
        )
        if before is not None
        else "—"
    )

    after_text = (
        str(
            after
        )
        if after is not None
        else "—"
    )

    return (
        f"{field}: "
        f"{before_text} → {after_text}"
    )


def format_ticket_history_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    ticket_key = (
        result.get(
            "ticket_key"
        )
    )

    history = (
        result.get(
            "history"
        )
    )

    if not isinstance(
        history,
        list,
    ):
        return (
            "Ticket history lookup completed, "
            "but no valid history was returned."
        )

    if not history:
        return (
            f"No history entries were returned "
            f"for {ticket_key or 'the ticket'}."
        )

    return (
        f"Retrieved {len(history)} history "
        f"entries for {ticket_key or 'the ticket'}."
    )


def build_ticket_history_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    ticket_key = (
        result.get(
            "ticket_key"
        )
    )

    history = (
        result.get(
            "history"
        )
    )

    if not isinstance(
        history,
        list,
    ):
        history = []

    items = []

    for entry in history:
        if not isinstance(
            entry,
            dict,
        ):
            continue

        created_at = (
            entry.get(
                "created_at"
            )
        )

        author = (
            entry.get(
                "author"
            )
        )

        changes = (
            entry.get(
                "changes"
            )
        )

        rendered_changes = []

        if isinstance(
            changes,
            list,
        ):
            for change in changes:
                if not isinstance(
                    change,
                    dict,
                ):
                    continue

                rendered = (
                    _render_change(
                        change
                    )
                )

                if rendered:
                    rendered_changes.append(
                        rendered
                    )

        prefix = []

        if (
            isinstance(
                created_at,
                str,
            )
            and created_at.strip()
        ):
            prefix.append(
                created_at.strip()
            )

        if (
            isinstance(
                author,
                str,
            )
            and author.strip()
        ):
            prefix.append(
                author.strip()
            )

        item = (
            " · ".join(
                prefix
            )
        )

        if rendered_changes:
            if item:
                item += " — "

            item += (
                "; ".join(
                    rendered_changes
                )
            )

        if item:
            items.append(
                item
            )

    return (
        build_result_card(
            kind="ticket-history",

            title=(
                f"{ticket_key} history"
                if isinstance(
                    ticket_key,
                    str,
                )
                and ticket_key.strip()
                else "Ticket history"
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
                    "Returned",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "More results",
                    result.get(
                        "truncated"
                    ),
                ),

                result_field(
                    "Provider",
                    result.get(
                        "provider"
                    ),
                ),
            ],

            sections=[
                list_section(
                    title="Changes",
                    items=items,
                )
            ],
        )
    )


# ============================================================
# TICKET COMMENTS
# ============================================================


def format_ticket_comments_result(
    result: dict[
        str,
        Any,
    ],
) -> str:
    ticket_key = (
        result.get(
            "ticket_key"
        )
    )

    comments = (
        result.get(
            "comments"
        )
    )

    if not isinstance(
        comments,
        list,
    ):
        return (
            "Ticket comment lookup completed, "
            "but no valid comments were returned."
        )

    if not comments:
        return (
            f"No comments were returned for "
            f"{ticket_key or 'the ticket'}."
        )

    return (
        f"Retrieved {len(comments)} comments "
        f"for {ticket_key or 'the ticket'}."
    )


def build_ticket_comments_card(
    result: dict[
        str,
        Any,
    ],
) -> dict[
    str,
    Any,
]:
    ticket_key = (
        result.get(
            "ticket_key"
        )
    )

    comments = (
        result.get(
            "comments"
        )
    )

    if not isinstance(
        comments,
        list,
    ):
        comments = []

    items = []

    for comment in comments:
        if not isinstance(
            comment,
            dict,
        ):
            continue

        author = (
            comment.get(
                "author"
            )
        )

        created_at = (
            comment.get(
                "created_at"
            )
        )

        body = (
            comment.get(
                "body"
            )
        )

        prefix = []

        if (
            isinstance(
                author,
                str,
            )
            and author.strip()
        ):
            prefix.append(
                author.strip()
            )

        if (
            isinstance(
                created_at,
                str,
            )
            and created_at.strip()
        ):
            prefix.append(
                created_at.strip()
            )

        item = (
            " · ".join(
                prefix
            )
        )

        if (
            isinstance(
                body,
                str,
            )
            and body.strip()
        ):
            if item:
                item += " — "

            item += (
                body.strip()
            )

        if item:
            items.append(
                item
            )

    return (
        build_result_card(
            kind="ticket-comments",

            title=(
                f"{ticket_key} comments"
                if isinstance(
                    ticket_key,
                    str,
                )
                and ticket_key.strip()
                else "Ticket comments"
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
                    "Returned",
                    result.get(
                        "count"
                    ),
                ),

                result_field(
                    "More results",
                    result.get(
                        "truncated"
                    ),
                ),

                result_field(
                    "Provider",
                    result.get(
                        "provider"
                    ),
                ),
            ],

            sections=[
                list_section(
                    title="Comments",
                    items=items,
                )
            ],
        )
    )