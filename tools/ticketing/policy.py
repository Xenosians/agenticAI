from __future__ import annotations

from typing import (
    Any,
)

from config import (
    Settings,
)

from services.ticketing import (
    JiraTicketService,
    build_ticket_service,
)

from services.ticketing.jira_create_policy import (
    prepare_jira_ticket_create,
)


def _provider_neutral_create_policy(
    *,
    project_key: str,
    summary: str,
    ticket_type: (
        str
        | None
    ) = None,
) -> dict[
    str,
    Any,
]:
    """
    Conservative provider-neutral fallback.

    Non-Jira providers do not currently expose a provider-owned
    create snapshot equivalent to Jira.

    The exact user/model arguments remain immutable and creation
    remains approval-gated at medium risk.
    """

    execution_arguments: dict[
        str,
        Any,
    ] = {
        "project_key":
            project_key,

        "summary":
            summary,
    }

    if (
        ticket_type
        is not None
    ):
        execution_arguments[
            "ticket_type"
        ] = (
            ticket_type
        )

    return {
        "ok":
            True,

        "status":
            "ready",

        "risk":
            "medium",

        "requires_approval":
            True,

        "execution_arguments":
            execution_arguments,

        "error":
            None,
    }


def evaluate_ticket_create_policy(
    project_key: str,
    summary: str,
    ticket_type: (
        str
        | None
    ) = None,
) -> dict[
    str,
    Any,
]:
    """
    Trusted ticket-create policy entry point.

    Provider selection is deployment configuration.

    Jira:
        perform read-only provider preparation and freeze the
        exact project / issue-type provider snapshot.

    Other configured ticket providers:
        preserve exact arguments and keep the mutation
        approval-gated at medium risk.

    No ticket mutation occurs here.
    """

    settings = (
        Settings()
    )

    if (
        settings.ticketing_backend
        != "jira"
    ):
        return (
            _provider_neutral_create_policy(
                project_key=(
                    project_key
                ),

                summary=(
                    summary
                ),

                ticket_type=(
                    ticket_type
                ),
            )
        )

    service = None

    try:
        service = (
            build_ticket_service(
                settings
            )
        )

        if not isinstance(
            service,
            JiraTicketService,
        ):
            return {
                "ok":
                    False,

                "status":
                    "error",

                "risk":
                    "medium",

                "requires_approval":
                    True,

                "error": (
                    "Configured Jira ticket provider did not "
                    "produce the trusted Jira service."
                ),
            }

        return (
            prepare_jira_ticket_create(
                service,
                project_key=(
                    project_key
                ),

                summary=(
                    summary
                ),

                ticket_type=(
                    ticket_type
                ),
            )
        )

    except Exception:
        # Do not surface provider credentials, URLs, or arbitrary
        # initialization exception details through policy output.
        return {
            "ok":
                False,

            "status":
                "error",

            "risk":
                "medium",

            "requires_approval":
                True,

            "error": (
                "Trusted ticket-create policy could not "
                "initialize the configured provider."
            ),
        }

    finally:
        if (
            service
            is not None
        ):
            close = (
                getattr(
                    service,
                    "close",
                    None,
                )
            )

            if callable(
                close
            ):
                close()
