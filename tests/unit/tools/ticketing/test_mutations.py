import json

import httpx

from services.ticketing import (
    JiraTicketService,
    MockTicketService,
    build_ticket_mutation_service,
)

from tools.registry import (
    get_tool,
)


def test_ticket_add_comment_catalog_requires_approval():

    tool = (
        get_tool(
            "ticket_add_comment"
        )
    )

    assert (
        tool
        is not None
    )

    assert (
        tool[
            "risk"
        ]
        == "low"
    )

    assert (
        tool[
            "requires_approval"
        ]
        is True
    )

    assert (
        tool[
            "grounded_arguments"
        ]
        == [
            "ticket_key",
            "comment",
        ]
    )


def test_mock_comment_mutation_is_visible_to_reads():

    service = (
        MockTicketService()
    )

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    before = (
        service
        .get_ticket_comments(
            "ITSM-101",
            limit=20,
        )
    )

    assert (
        before.count
        == 1
    )

    result = (
        mutations
        .add_comment(
            "ITSM-101",
            "VPN access was verified.",
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.changed is True
    assert result.provider == "mock"
    assert result.ticket_key == "ITSM-101"

    after = (
        service
        .get_ticket_comments(
            "ITSM-101",
            limit=20,
        )
    )

    assert (
        after.count
        == 2
    )

    assert (
        after.comments[
            -1
        ].body
        == "VPN access was verified."
    )


def test_mock_comment_mutation_rejects_empty_comment():

    service = (
        MockTicketService()
    )

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    result = (
        mutations
        .add_comment(
            "ITSM-101",
            "   ",
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.changed is False


def test_mock_comment_mutation_unknown_ticket_is_not_found():

    service = (
        MockTicketService()
    )

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    result = (
        mutations
        .add_comment(
            "ITSM-999",
            "Investigation started.",
        )
    )

    assert result.ok is False
    assert result.status == "not_found"
    assert result.changed is False


def test_jira_comment_mutation_uses_adf_and_exact_ticket():

    observed = {}

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        observed[
            "method"
        ] = (
            request.method
        )

        observed[
            "path"
        ] = (
            request.url.path
        )

        observed[
            "payload"
        ] = (
            json.loads(
                request
                .content
                .decode(
                    "utf-8"
                )
            )
        )

        return (
            httpx.Response(
                201,

                json={
                    "id":
                        "2001",
                },
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "agent@example.com"
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

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    result = (
        mutations
        .add_comment(
            "itsm-101",
            "VPN access was verified.",
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.changed is True
    assert result.provider == "jira"
    assert result.ticket_key == "ITSM-101"
    assert result.comment_id == "2001"

    assert (
        observed[
            "method"
        ]
        == "POST"
    )

    assert (
        observed[
            "path"
        ]
        == (
            "/rest/api/3/"
            "issue/ITSM-101/comment"
        )
    )

    payload = (
        observed[
            "payload"
        ]
    )

    assert (
        payload[
            "body"
        ][
            "type"
        ]
        == "doc"
    )

    assert (
        payload[
            "body"
        ][
            "version"
        ]
        == 1
    )

    assert (
        payload[
            "body"
        ][
            "content"
        ][
            0
        ][
            "content"
        ][
            0
        ][
            "text"
        ]
        == "VPN access was verified."
    )

    service.close()


def test_jira_comment_5xx_is_reported_as_unknown_state():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        return (
            httpx.Response(
                503,

                json={
                    "error":
                        "unavailable",
                },
            )
        )

    service = (
        JiraTicketService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "agent@example.com"
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

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    result = (
        mutations
        .add_comment(
            "ITSM-101",
            "Investigation started.",
        )
    )

    assert result.ok is False
    assert result.status == "unknown"
    assert result.changed is False

    service.close()
