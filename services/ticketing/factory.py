from config import (
    Settings,
)

from .base import (
    TicketService,
)

from .jira import (
    JiraTicketService,
)

from .mock import (
    MockTicketService,
)


def build_ticket_service(
    settings: Settings,
) -> TicketService:
    """
    Build one ticket-service implementation from validated
    application configuration.

    Provider choice is deployment configuration.

    Agents, ToolGateway, and model-facing tool definitions depend
    only on the logical TicketService boundary.
    """

    backend = (
        settings
        .ticketing_backend
    )

    if backend == "mock":
        return (
            MockTicketService()
        )

    if backend == "jira":
        return (
            JiraTicketService(
                base_url=(
                    settings
                    .require_jira_base_url()
                ),

                email=(
                    settings
                    .require_jira_email()
                ),

                api_token=(
                    settings
                    .require_jira_api_token()
                ),

                timeout_seconds=(
                    settings
                    .jira_http_timeout_seconds
                ),
            )
        )

    raise RuntimeError(
        "Unsupported ticketing backend: "
        f"{backend}"
    )