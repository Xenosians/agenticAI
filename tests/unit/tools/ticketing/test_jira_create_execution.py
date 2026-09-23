import json

import httpx

from services.ticketing import (
    JiraTicketService,
    build_ticket_mutation_service,
)


APPROVED = {
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


def build_service(
    handler,
):

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


def project_response(
    *,
    name="Kanban",
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
                    name,
            },
        )
    )


def issue_types_response(
    *,
    type_id="10010",
    type_name="Task",
):

    return (
        httpx.Response(
            200,
            json={
                "issueTypes": [
                    {
                        "id":
                            type_id,

                        "name":
                            type_name,

                        "subtask":
                            False,
                    },
                ],

                "total":
                    1,
            },
        )
    )


def created_issue_response():

    return (
        httpx.Response(
            200,
            json={
                "key":
                    "KAN-4",

                "fields": {
                    "summary":
                        "VPN login failure",

                    "project": {
                        "id":
                            "10001",

                        "key":
                            "KAN",

                        "name":
                            "Kanban",
                    },

                    "issuetype": {
                        "id":
                            "10010",

                        "name":
                            "Task",
                    },
                },
            },
        )
    )


def test_jira_create_revalidates_snapshot_uses_ids_and_verifies():

    requests = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        requests.append(
            (
                request.method,
                request.url.path,
            )
        )

        if (
            request.method == "GET"
            and request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                project_response()
            )

        if (
            request.method == "GET"
            and request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                issue_types_response()
            )

        if (
            request.method == "POST"
            and request.url.path
            == "/rest/api/3/issue"
        ):

            body = (
                json.loads(
                    request
                    .content
                    .decode(
                        "utf-8"
                    )
                )
            )

            assert body == {
                "fields": {
                    "project": {
                        "id":
                            "10001",
                    },

                    "summary":
                        "VPN login failure",

                    "issuetype": {
                        "id":
                            "10010",
                    },
                }
            }

            return (
                httpx.Response(
                    201,
                    json={
                        "id":
                            "20001",

                        "key":
                            "KAN-4",
                    },
                )
            )

        if (
            request.method == "GET"
            and request.url.path
            == "/rest/api/3/issue/KAN-4"
        ):
            assert (
                request.url.params[
                    "fields"
                ]
                == (
                    "summary,"
                    "project,"
                    "issuetype"
                )
            )

            return (
                created_issue_response()
            )

        raise AssertionError(
            f"Unexpected request: {request.method} {request.url}"
        )

    service = (
        build_service(
            handler
        )
    )

    mutations = (
        build_ticket_mutation_service(
            service
        )
    )

    result = (
        mutations
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is True
    assert result.status == "success"
    assert result.ticket_key == "KAN-4"
    assert result.changed is True
    assert result.mutation_performed is True
    assert result.verification_ok is True
    assert result.reconciled is False

    assert requests == [
        (
            "GET",
            "/rest/api/3/project/10001",
        ),
        (
            "GET",
            (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            ),
        ),
        (
            "POST",
            "/rest/api/3/issue",
        ),
        (
            "GET",
            "/rest/api/3/issue/KAN-4",
        ),
    ]

    service.close()


def test_jira_create_rejects_stale_project_before_post():

    methods = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        methods.append(
            request.method
        )

        assert (
            request.url.path
            == "/rest/api/3/project/10001"
        )

        return (
            project_response(
                name=(
                    "Renamed Kanban"
                )
            )
        )

    service = (
        build_service(
            handler
        )
    )

    result = (
        build_ticket_mutation_service(
            service
        )
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.mutation_performed is False
    assert result.verification_ok is False

    assert methods == [
        "GET",
    ]

    service.close()


def test_jira_create_rejects_stale_issue_type_before_post():

    methods = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        methods.append(
            request.method
        )

        if (
            request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                project_response()
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                issue_types_response(
                    type_id=(
                        "99999"
                    )
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
        build_ticket_mutation_service(
            service
        )
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.mutation_performed is False

    assert "POST" not in methods

    service.close()


def test_jira_create_explicit_400_is_known_failure():

    post_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal post_count

        if (
            request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                project_response()
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                issue_types_response()
            )

        if (
            request.method == "POST"
            and request.url.path
            == "/rest/api/3/issue"
        ):
            post_count += 1

            return (
                httpx.Response(
                    400,
                    json={
                        "errorMessages": [
                            "invalid"
                        ],
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
        build_ticket_mutation_service(
            service
        )
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is False
    assert result.status == "denied"
    assert result.mutation_performed is False
    assert result.verification_ok is False
    assert post_count == 1

    service.close()


def test_jira_create_transport_failure_is_outcome_unknown_no_retry():

    post_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal post_count

        if (
            request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                project_response()
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                issue_types_response()
            )

        if (
            request.method == "POST"
            and request.url.path
            == "/rest/api/3/issue"
        ):
            post_count += 1

            raise httpx.ReadTimeout(
                "simulated ambiguous timeout"
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
        build_ticket_mutation_service(
            service
        )
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is False
    assert (
        result.status
        == "outcome_unknown"
    )
    assert (
        result.mutation_performed
        is None
    )
    assert result.verification_ok is False

    # Critical: no blind retry.
    assert post_count == 1

    service.close()


def test_jira_create_5xx_is_outcome_unknown_no_retry():

    post_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal post_count

        if (
            request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                project_response()
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                issue_types_response()
            )

        if (
            request.method == "POST"
            and request.url.path
            == "/rest/api/3/issue"
        ):
            post_count += 1

            return (
                httpx.Response(
                    503,
                    json={
                        "error":
                            "unavailable",
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
        build_ticket_mutation_service(
            service
        )
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is False
    assert (
        result.status
        == "outcome_unknown"
    )
    assert (
        result.mutation_performed
        is None
    )
    assert post_count == 1

    service.close()


def test_jira_create_201_without_verifiable_readback_is_not_success():

    post_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal post_count

        if (
            request.url.path
            == "/rest/api/3/project/10001"
        ):
            return (
                project_response()
            )

        if (
            request.url.path
            == (
                "/rest/api/3/issue/"
                "createmeta/10001/issuetypes"
            )
        ):
            return (
                issue_types_response()
            )

        if (
            request.method == "POST"
            and request.url.path
            == "/rest/api/3/issue"
        ):
            post_count += 1

            return (
                httpx.Response(
                    201,
                    json={
                        "id":
                            "20001",

                        "key":
                            "KAN-4",
                    },
                )
            )

        if (
            request.method == "GET"
            and request.url.path
            == "/rest/api/3/issue/KAN-4"
        ):
            return (
                httpx.Response(
                    503,
                    json={
                        "error":
                            "unavailable",
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
        build_ticket_mutation_service(
            service
        )
        .create_ticket(
            **APPROVED
        )
    )

    assert result.ok is False
    assert (
        result.status
        == "outcome_unknown"
    )

    # 201 proves the provider accepted creation,
    # but exact resulting state is not verified.
    assert (
        result.mutation_performed
        is True
    )

    assert result.verification_ok is False
    assert result.ticket_key == "KAN-4"

    assert post_count == 1

    service.close()
