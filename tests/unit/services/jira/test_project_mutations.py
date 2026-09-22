import json

import httpx

from services.jira.project_mutations import (
    JiraProjectMutationService,
    PROJECT_TEMPLATE_PROFILES,
)


def _service(
    handler,
) -> JiraProjectMutationService:

    return (
        JiraProjectMutationService(
            base_url=(
                "https://example.atlassian.net"
            ),

            email=(
                "admin@example.com"
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


def _common_get(
    request: httpx.Request,
):

    path = (
        request.url.path
    )

    if (
        path
        == (
            "/rest/api/3/projectvalidate/"
            "validProjectKey"
        )
    ):
        return (
            httpx.Response(
                200,
                content=(
                    request.url.params[
                        "key"
                    ]
                    .encode(
                        "utf-8"
                    )
                ),
                headers={
                    "Content-Type":
                        "application/json;charset=UTF-8",
                },
            )
        )

    if (
        path
        == (
            "/rest/api/3/projectvalidate/"
            "validProjectName"
        )
    ):
        return (
            httpx.Response(
                200,
                content=(
                    request.url.params[
                        "name"
                    ]
                    .encode(
                        "utf-8"
                    )
                ),
                headers={
                    "Content-Type":
                        "application/json;charset=UTF-8",
                },
            )
        )

    if (
        path
        == (
            "/rest/api/3/project/type/"
            "software/accessible"
        )
    ):
        return (
            httpx.Response(
                200,
                json={
                    "key":
                        "software",
                },
            )
        )

    if (
        path
        == "/rest/api/3/myself"
    ):
        return (
            httpx.Response(
                200,
                json={
                    "accountId":
                        "account-1",
                },
            )
        )

    return None


def test_prepare_create_binds_trusted_provider_state():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        response = (
            _common_get(
                request
            )
        )

        if (
            response
            is not None
        ):
            return (
                response
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url.path}"
        )

    result = (
        _service(
            handler
        )
        .prepare_create_project(
            project_key="NEW",
            project_name="New Project",
            template="software-kanban",
        )
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "requires_approval"
        ]
        is True
    )

    assert (
        result[
            "risk"
        ]
        == "medium"
    )

    execution = (
        result[
            "execution_arguments"
        ]
    )

    assert (
        execution[
            "project_key"
        ]
        == "NEW"
    )

    assert (
        execution[
            "project_name"
        ]
        == "New Project"
    )

    assert (
        execution[
            "template"
        ]
        == "software-kanban"
    )

    assert (
        execution[
            "expected_project_type_key"
        ]
        == "software"
    )

    assert (
        execution[
            "expected_project_template_key"
        ]
        == (
            PROJECT_TEMPLATE_PROFILES[
                "software-kanban"
            ][
                "project_template_key"
            ]
        )
    )

    assert (
        execution[
            "expected_lead_account_id"
        ]
        == "account-1"
    )


def test_prepare_create_refuses_provider_key_rewrite():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        path = (
            request.url.path
        )

        if (
            path
            == (
                "/rest/api/3/project/type/"
                "software/accessible"
            )
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "key":
                            "software",
                    },
                )
            )

        if (
            path
            == (
                "/rest/api/3/projectvalidate/"
                "validProjectKey"
            )
        ):
            return (
                httpx.Response(
                    200,
                    content=b"NEW2",
                    headers={
                        "Content-Type":
                            "application/json;charset=UTF-8",
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: {path}"
        )

    result = (
        _service(
            handler
        )
        .prepare_create_project(
            project_key="NEW",
            project_name="New Project",
            template="software-kanban",
        )
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "status"
        ]
        == "denied"
    )

    assert (
        "will not rewrite"
        in result[
            "error"
        ]
    )


def test_create_project_executes_exact_approved_snapshot_and_verifies():

    post_calls = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        response = (
            _common_get(
                request
            )
        )

        if (
            response
            is not None
        ):
            return (
                response
            )

        if (
            request.method
            == "POST"
            and request.url.path
            == "/rest/api/3/project"
        ):

            body = (
                json.loads(
                    request.content
                )
            )

            post_calls.append(
                body
            )

            assert (
                body
                == {
                    "key":
                        "NEW",

                    "name":
                        "New Project",

                    "projectTypeKey":
                        "software",

                    "projectTemplateKey":
                        (
                            PROJECT_TEMPLATE_PROFILES[
                                "software-kanban"
                            ][
                                "project_template_key"
                            ]
                        ),

                    "leadAccountId":
                        "account-1",

                    "assigneeType":
                        "PROJECT_LEAD",
                }
            )

            return (
                httpx.Response(
                    201,
                    json={
                        "id":
                            10002,

                        "key":
                            "NEW",
                    },
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/NEW"
        ):

            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10002",

                        "key":
                            "NEW",

                        "name":
                            "New Project",

                        "projectTypeKey":
                            "software",
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url.path}"
        )

    result = (
        _service(
            handler
        )
        .create_project(
            project_key="NEW",
            project_name="New Project",
            template="software-kanban",
            expected_project_type_key="software",
            expected_project_template_key=(
                PROJECT_TEMPLATE_PROFILES[
                    "software-kanban"
                ][
                    "project_template_key"
                ]
            ),
            expected_lead_account_id="account-1",
        )
    )

    assert (
        len(
            post_calls
        )
        == 1
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "status"
        ]
        == "success"
    )

    assert (
        result[
            "project_id"
        ]
        == "10002"
    )

    assert (
        result[
            "mutation_performed"
        ]
        is True
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )


def test_create_fails_closed_when_approved_identity_changes():

    post_called = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal post_called

        if (
            request.url.path
            == "/rest/api/3/myself"
        ):
            return (
                httpx.Response(
                    200,
                    json={
                        "accountId":
                            "different-account",
                    },
                )
            )

        if request.method == "POST":
            post_called = True

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url.path}"
        )

    result = (
        _service(
            handler
        )
        .create_project(
            project_key="NEW",
            project_name="New Project",
            template="software-kanban",
            expected_project_type_key="software",
            expected_project_template_key=(
                PROJECT_TEMPLATE_PROFILES[
                    "software-kanban"
                ][
                    "project_template_key"
                ]
            ),
            expected_lead_account_id="account-1",
        )
    )

    assert (
        post_called
        is False
    )

    assert (
        result[
            "ok"
        ]
        is False
    )

    assert (
        result[
            "status"
        ]
        == "denied"
    )


def test_timeout_reconciles_verified_created_project():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        response = (
            _common_get(
                request
            )
        )

        if (
            response
            is not None
        ):
            return (
                response
            )

        if (
            request.method
            == "POST"
            and request.url.path
            == "/rest/api/3/project"
        ):

            raise (
                httpx.ReadTimeout(
                    "simulated timeout",
                    request=(
                        request
                    ),
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/NEW"
        ):

            return (
                httpx.Response(
                    200,
                    json={
                        "id":
                            "10002",

                        "key":
                            "NEW",

                        "name":
                            "New Project",

                        "projectTypeKey":
                            "software",
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url.path}"
        )

    result = (
        _service(
            handler
        )
        .create_project(
            project_key="NEW",
            project_name="New Project",
            template="software-kanban",
            expected_project_type_key="software",
            expected_project_template_key=(
                PROJECT_TEMPLATE_PROFILES[
                    "software-kanban"
                ][
                    "project_template_key"
                ]
            ),
            expected_lead_account_id="account-1",
        )
    )

    assert (
        result[
            "ok"
        ]
        is True
    )

    assert (
        result[
            "verification_ok"
        ]
        is True
    )

    assert (
        result[
            "reconciled"
        ]
        is True
    )
