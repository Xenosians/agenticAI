import httpx

from services.ticketing.jira import (
    JiraTicketService,
)

from services.ticketing.jira_create_policy import (
    prepare_jira_ticket_create,
)


def build_service(
    handler,
) -> JiraTicketService:

    return (
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


def test_prepare_ticket_create_binds_exact_provider_snapshot():

    observed = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        observed.append(
            (
                request.method,
                request.url.path,
                dict(
                    request.url.params
                ),
            )
        )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/KAN"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "issueTypes": [
                            {
                                "id":
                                    "10010",

                                "name":
                                    "Task",

                                "subtask":
                                    False,
                            },

                            {
                                "id":
                                    "10011",

                                "name":
                                    "Bug",

                                "subtask":
                                    False,
                            },
                        ],

                        "startAt":
                            0,

                        "maxResults":
                            50,

                        "total":
                            2,
                    },
                )
            )

        raise AssertionError(
            (
                "Unexpected request: "
                f"{request.method} "
                f"{request.url}"
            )
        )

    service = (
        build_service(
            handler
        )
    )

    result = (
        prepare_jira_ticket_create(
            service,
            project_key="KAN",
            summary="VPN login failure",
        )
    )

    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["risk"] == "medium"
    assert result["requires_approval"] is True

    arguments = (
        result[
            "execution_arguments"
        ]
    )

    assert arguments == {
        "project_key":
            "KAN",

        "summary":
            "VPN login failure",

        "expected_project_id":
            "10001",

        "expected_project_key":
            "KAN",

        "expected_project_name":
            "Kanban",

        "expected_ticket_type_id":
            "10010",

        "expected_ticket_type_name":
            "Task",
    }

    # No optional model-facing ticket_type was invented.
    assert (
        "ticket_type"
        not in arguments
    )

    assert observed == [
        (
            "GET",
            "/rest/api/3/project/KAN",
            {},
        ),
        (
            "GET",
            (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            ),
            {
                "startAt":
                    "0",

                "maxResults":
                    "50",
            },
        ),
    ]

    service.close()


def test_prepare_ticket_create_binds_explicit_issue_type():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.url.path
            == "/rest/api/3/project/KAN"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },
                )
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "issueTypes": [
                            {
                                "id":
                                    "10011",

                                "name":
                                    "Bug",

                                "subtask":
                                    False,
                            },
                        ],

                        "total":
                            1,
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: {request.url}"
        )

    service = (
        build_service(
            handler
        )
    )

    result = (
        prepare_jira_ticket_create(
            service,
            project_key="KAN",
            summary="Broken dashboard",
            ticket_type="Bug",
        )
    )

    assert result["ok"] is True

    arguments = (
        result[
            "execution_arguments"
        ]
    )

    assert (
        arguments[
            "ticket_type"
        ]
        == "Bug"
    )

    assert (
        arguments[
            "expected_ticket_type_id"
        ]
        == "10011"
    )

    assert (
        arguments[
            "expected_ticket_type_name"
        ]
        == "Bug"
    )

    service.close()


def test_prepare_ticket_create_rejects_project_key_rewrite():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        assert (
            request.url.path
            == "/rest/api/3/project/kan"
        )

        return (
            httpx.Response(
                200,
                json={
                    "id":
                        "10001",

                    "key":
                        "KAN",

                    "name":
                        "Kanban",
                },
            )
        )

    service = (
        build_service(
            handler
        )
    )

    result = (
        prepare_jira_ticket_create(
            service,
            project_key="kan",
            summary="VPN login failure",
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"

    assert (
        "silently rewrite"
        in result["error"]
    )

    service.close()


def test_prepare_ticket_create_rejects_unavailable_issue_type():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.url.path
            == "/rest/api/3/project/KAN"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },
                )
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "issueTypes": [
                            {
                                "id":
                                    "10010",

                                "name":
                                    "Task",

                                "subtask":
                                    False,
                            },
                        ],

                        "total":
                            1,
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: {request.url}"
        )

    service = (
        build_service(
            handler
        )
    )

    result = (
        prepare_jira_ticket_create(
            service,
            project_key="KAN",
            summary="Broken dashboard",
            ticket_type="Incident",
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"

    assert (
        "not available"
        in result["error"]
    )

    service.close()


def test_prepare_ticket_create_provider_failure_is_error():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.url.path
            == "/rest/api/3/project/KAN"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },
                )
            )

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
        build_service(
            handler
        )
    )

    result = (
        prepare_jira_ticket_create(
            service,
            project_key="KAN",
            summary="VPN login failure",
        )
    )

    assert result["ok"] is False
    assert result["status"] == "error"

    service.close()
