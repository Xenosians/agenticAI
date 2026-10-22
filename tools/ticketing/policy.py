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

from services.ticketing.jira_assignment_policy import (
    prepare_jira_ticket_assignment,
)

from services.ticketing.jira_comment_policy import (
    prepare_jira_ticket_comment,
)

from services.ticketing.jira_transition_policy import (
    prepare_jira_ticket_transition,
)

from services.ticketing.jira_create_policy import (
    prepare_jira_ticket_create,
)

from services.ticketing.jira_link_policy import (
    prepare_jira_ticket_link,
)

from services.ticketing.jira_update_policy import (
    prepare_jira_ticket_update,
)

from services.ticketing.jira_worklog_policy import (
    prepare_jira_worklog_add,
)

from services.ticketing.validation import (
    validate_update_request,
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


def _provider_neutral_add_comment_policy(
    *,
    ticket_key: str,
    comment: str,
) -> dict[
    str,
    Any,
]:
    return {
        "ok": True,
        "status": "ready",
        "risk": "medium",
        "requires_approval": True,
        "execution_arguments": {
            "ticket_key": ticket_key,
            "comment": comment,
        },
        "error": None,
    }


def evaluate_ticket_add_comment_policy(
    ticket_key: str,
    comment: str,
) -> dict[
    str,
    Any,
]:
    """Trusted ticket-comment policy entry point; performs no mutation."""

    settings = Settings()

    if settings.ticketing_backend != "jira":
        return _provider_neutral_add_comment_policy(
            ticket_key=ticket_key,
            comment=comment,
        )

    service = None

    try:
        service = build_ticket_service(
            settings
        )

        if not isinstance(
            service,
            JiraTicketService,
        ):
            return {
                "ok": False,
                "status": "error",
                "risk": "medium",
                "requires_approval": True,
                "error": (
                    "Configured Jira ticket provider did not produce the "
                    "trusted Jira service."
                ),
            }

        return prepare_jira_ticket_comment(
            service,
            ticket_key=ticket_key,
            comment=comment,
        )

    except Exception:
        return {
            "ok": False,
            "status": "error",
            "risk": "medium",
            "requires_approval": True,
            "error": (
                "Trusted ticket-comment policy could not initialize the "
                "configured provider."
            ),
        }

    finally:
        if service is not None:
            close = getattr(
                service,
                "close",
                None,
            )
            if callable(close):
                close()


def _provider_neutral_assignment_policy(
    *,
    ticket_key: str,
    assignee: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ready",
        "risk": "medium",
        "requires_approval": True,
        "execution_arguments": {
            "ticket_key": ticket_key,
            "assignee": assignee,
        },
        "error": None,
    }


def evaluate_ticket_assign_policy(
    ticket_key: str,
    assignee: str,
) -> dict[str, Any]:
    settings = Settings()

    if settings.ticketing_backend != "jira":
        return _provider_neutral_assignment_policy(
            ticket_key=ticket_key,
            assignee=assignee,
        )

    service = None
    try:
        service = build_ticket_service(settings)
        if not isinstance(service, JiraTicketService):
            return {
                "ok": False,
                "status": "error",
                "risk": "medium",
                "requires_approval": True,
                "error": (
                    "Configured Jira ticket provider did not produce the trusted Jira service."
                ),
            }

        return prepare_jira_ticket_assignment(
            service,
            ticket_key=ticket_key,
            assignee=assignee,
        )
    except Exception:
        return {
            "ok": False,
            "status": "error",
            "risk": "medium",
            "requires_approval": True,
            "error": (
                "Trusted ticket-assignment policy could not initialize the configured provider."
            ),
        }
    finally:
        if service is not None:
            close = getattr(service, "close", None)
            if callable(close):
                close()


def _provider_neutral_transition_policy(
    *,
    ticket_key: str,
    status: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "ready",
        "risk": "medium",
        "requires_approval": True,
        "execution_arguments": {
            "ticket_key": ticket_key,
            "status": status,
        },
        "error": None,
    }


def evaluate_ticket_transition_policy(
    ticket_key: str,
    status: str,
) -> dict[str, Any]:
    settings = Settings()

    if settings.ticketing_backend != "jira":
        return _provider_neutral_transition_policy(
            ticket_key=ticket_key,
            status=status,
        )

    service = None
    try:
        service = build_ticket_service(settings)
        if not isinstance(service, JiraTicketService):
            return {
                "ok": False,
                "status": "error",
                "risk": "medium",
                "requires_approval": True,
                "error": (
                    "Configured Jira ticket provider did not produce the trusted Jira service."
                ),
            }

        return prepare_jira_ticket_transition(
            service,
            ticket_key=ticket_key,
            status=status,
        )
    except Exception:
        return {
            "ok": False,
            "status": "error",
            "risk": "medium",
            "requires_approval": True,
            "error": (
                "Trusted ticket-transition policy could not initialize the configured provider."
            ),
        }
    finally:
        if service is not None:
            close = getattr(service, "close", None)
            if callable(close):
                close()


def _prepare_with_jira(
    prepare,
    *,
    action: str,
    provider_neutral_arguments: dict[str, Any],
    **arguments: Any,
) -> dict[str, Any]:
    """
    Shared trusted policy entry point for the Jira depth mutations.

    Jira: read-only preparation freezes the provider snapshot. Other
    providers keep the exact arguments, approval-gated at medium risk.
    """

    settings = Settings()

    if settings.ticketing_backend != "jira":
        return {
            "ok": True,
            "status": "ready",
            "risk": "medium",
            "requires_approval": True,
            "execution_arguments": provider_neutral_arguments,
            "error": None,
        }

    service = None
    try:
        service = build_ticket_service(settings)
        if not isinstance(service, JiraTicketService):
            return {
                "ok": False,
                "status": "error",
                "risk": "medium",
                "requires_approval": True,
                "error": (
                    "Configured Jira ticket provider did not produce the trusted Jira service."
                ),
            }

        return prepare(service, **arguments)
    except Exception:
        return {
            "ok": False,
            "status": "error",
            "risk": "medium",
            "requires_approval": True,
            "error": (
                f"Trusted {action} policy could not initialize the configured provider."
            ),
        }
    finally:
        if service is not None:
            close = getattr(service, "close", None)
            if callable(close):
                close()


def _present(**arguments: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in arguments.items()
        if value is not None
    }


def evaluate_ticket_update_policy(
    ticket_key: str,
    priority: str | None = None,
    add_labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
    due_date: str | None = None,
) -> dict[str, Any]:
    try:
        validate_update_request(
            priority=priority,
            add_labels=add_labels,
            remove_labels=remove_labels,
            due_date=due_date,
        )
    except ValueError as exc:
        return {
            "ok": False,
            "status": "denied",
            "risk": "medium",
            "requires_approval": True,
            "error": str(exc),
        }

    return _prepare_with_jira(
        prepare_jira_ticket_update,
        action="ticket-update",
        provider_neutral_arguments=_present(
            ticket_key=ticket_key,
            priority=priority,
            add_labels=add_labels,
            remove_labels=remove_labels,
            due_date=due_date,
        ),
        ticket_key=ticket_key,
        priority=priority,
        add_labels=add_labels,
        remove_labels=remove_labels,
        due_date=due_date,
    )


def evaluate_ticket_link_policy(
    ticket_key: str,
    other_ticket_key: str,
    relationship: str,
) -> dict[str, Any]:
    return _prepare_with_jira(
        prepare_jira_ticket_link,
        action="ticket-link",
        provider_neutral_arguments={
            "ticket_key": ticket_key,
            "other_ticket_key": other_ticket_key,
            "relationship": relationship,
        },
        ticket_key=ticket_key,
        other_ticket_key=other_ticket_key,
        relationship=relationship,
    )


def evaluate_ticket_worklog_add_policy(
    ticket_key: str,
    time_spent: str,
    comment: str | None = None,
) -> dict[str, Any]:
    return _prepare_with_jira(
        prepare_jira_worklog_add,
        action="ticket-worklog",
        provider_neutral_arguments=_present(
            ticket_key=ticket_key,
            time_spent=time_spent,
            comment=comment,
        ),
        ticket_key=ticket_key,
        time_spent=time_spent,
        comment=comment,
    )
