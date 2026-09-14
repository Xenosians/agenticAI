import httpx

from services.ticketing import (
    JiraTicketService,
)


def test_jira_get_ticket_maps_issue_fields():
    seen_requests = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        seen_requests.append(
            request
        )

        return (
            httpx.Response(
                200,

                json={
                    "id": "10001",

                    "key":
                        "ITSM-101",

                    "fields": {
                        "summary":
                            "VPN access unavailable",

                        "status": {
                            "name":
                                "In Progress",
                        },

                        "issuetype": {
                            "name":
                                "Task",
                        },

                        "priority": {
                            "name":
                                "High",
                        },

                        "assignee": {
                            "displayName":
                                "Service Desk",
                        },

                        "reporter": {
                            "displayName":
                                "Jane Doe",
                        },

                        "project": {
                            "key":
                                "ITSM",

                            "name": (
                                "IT Service "
                                "Management"
                            ),
                        },

                        "created": (
                            "2026-09-01T09:"
                            "00:00.000+0000"
                        ),

                        "updated": (
                            "2026-09-14T10:"
                            "30:00.000+0000"
                        ),
                    },
                },
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=(
                "test-token"
            ),

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )

    try:
        result = (
            service.get_ticket(
                "itsm-101"
            )
        )

    finally:
        service.close()

    assert result.ok is True

    assert (
        result.status
        == "success"
    )

    assert (
        result.ticket
        is not None
    )

    assert (
        result.ticket.provider
        == "jira"
    )

    assert (
        result.ticket.key
        == "ITSM-101"
    )

    assert (
        result.ticket.summary
        == "VPN access unavailable"
    )

    assert (
        result.ticket.status
        == "In Progress"
    )

    assert (
        result.ticket.ticket_type
        == "Task"
    )

    assert (
        result.ticket.priority
        == "High"
    )

    assert (
        result.ticket.assignee
        == "Service Desk"
    )

    assert (
        result.ticket.project_key
        == "ITSM"
    )

    assert len(
        seen_requests
    ) == 1

    request = (
        seen_requests[
            0
        ]
    )

    assert (
        request.url.path
        == (
            "/rest/api/3/"
            "issue/ITSM-101"
        )
    )

    assert (
        request.url.params[
            "fields"
        ]
        is not None
    )

    assert (
        "authorization"
        in request.headers
    )


def test_jira_get_ticket_returns_not_found():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return (
            httpx.Response(
                404,

                json={
                    "errorMessages": [
                        "Issue does not exist."
                    ]
                },
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=(
                "test-token"
            ),

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )

    try:
        result = (
            service.get_ticket(
                "ITSM-999"
            )
        )

    finally:
        service.close()

    assert result.ok is False

    assert (
        result.status
        == "not_found"
    )

    assert (
        result.ticket
        is None
    )

    assert (
        "ITSM-999"
        in result.error
    )


def test_jira_get_ticket_rejects_invalid_key_without_http():
    called = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal called

        called = True

        return (
            httpx.Response(
                500
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=(
                "test-token"
            ),

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )

    try:
        result = (
            service.get_ticket(
                "../../etc/passwd"
            )
        )

    finally:
        service.close()

    assert result.ok is False

    assert (
        result.status
        == "denied"
    )

    assert called is False


def test_jira_auth_error_does_not_expose_credentials():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return (
            httpx.Response(
                401,

                json={
                    "message":
                        "Unauthorized"
                },
            )
        )

    secret = (
        "super-secret-api-token"
    )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=secret,

            transport=(
                httpx.MockTransport(
                    handler
                )
            ),
        )
    )

    try:
        result = (
            service.get_ticket(
                "ITSM-101"
            )
        )

    finally:
        service.close()

    assert result.ok is False

    assert (
        result.status
        == "error"
    )

    assert secret not in (
        result.error
        or ""
    )


def test_jira_rejects_base_url_with_credentials():

    try:
        JiraTicketService(
            base_url=(
                "https://user:pass@"
                "example.atlassian.net"
            ),

            email=(
                "service@example.com"
            ),

            api_token=(
                "test-token"
            ),
        )

    except ValueError as exc:

        assert (
            "credentials"
            in str(
                exc
            )
        )

    else:
        raise AssertionError(
            "Expected invalid Jira "
            "base URL to be rejected."
        )
