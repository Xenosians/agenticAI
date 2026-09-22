import httpx

from services.jira import (
    JiraProjectReadService,
)


def _service(
    handler,
) -> JiraProjectReadService:

    return (
        JiraProjectReadService(
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


def test_project_list_uses_bounded_project_search():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        assert (
            request.method
            == "GET"
        )

        assert (
            request.url.path
            == "/rest/api/3/project/search"
        )

        assert (
            request.url.params[
                "startAt"
            ]
            == "0"
        )

        assert (
            request.url.params[
                "maxResults"
            ]
            == "2"
        )

        assert (
            request.url.params[
                "query"
            ]
            == "platform"
        )

        return (
            httpx.Response(
                200,
                json={
                    "startAt":
                        0,

                    "maxResults":
                        2,

                    "total":
                        3,

                    "isLast":
                        False,

                    "values": [
                        {
                            "id":
                                "10000",

                            "key":
                                "PLAT",

                            "name":
                                "Platform",

                            "projectTypeKey":
                                "software",

                            "style":
                                "next-gen",

                            "simplified":
                                True,
                        },

                        {
                            "id":
                                "10001",

                            "key":
                                "OPS",

                            "name":
                                "Operations",

                            "projectTypeKey":
                                "service_desk",

                            "style":
                                "classic",

                            "simplified":
                                False,
                        },
                    ],
                },
            )
        )

    result = (
        _service(
            handler
        )
        .list_projects(
            query="platform",
            limit=2,
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
            "status"
        ]
        == "success"
    )

    assert (
        result[
            "query"
        ]
        == "platform"
    )

    assert (
        result[
            "count"
        ]
        == 2
    )

    assert (
        result[
            "total"
        ]
        == 3
    )

    assert (
        result[
            "truncated"
        ]
        is True
    )

    assert (
        result[
            "projects"
        ][
            0
        ][
            "key"
        ]
        == "PLAT"
    )


def test_project_get_uses_exact_quoted_reference():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        assert (
            request.method
            == "GET"
        )

        assert (
            request.url.path
            == "/rest/api/3/project/OPS"
        )

        return (
            httpx.Response(
                200,
                json={
                    "id":
                        "10001",

                    "key":
                        "OPS",

                    "name":
                        "Operations",

                    "projectTypeKey":
                        "service_desk",

                    "style":
                        "classic",

                    "simplified":
                        False,

                    "archived":
                        False,

                    "projectCategory": {
                        "id":
                            "20000",

                        "name":
                            "IT Operations",
                    },
                },
            )
        )

    result = (
        _service(
            handler
        )
        .get_project(
            "OPS"
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
            "project_id_or_key"
        ]
        == "OPS"
    )

    assert (
        result[
            "project"
        ][
            "key"
        ]
        == "OPS"
    )

    assert (
        result[
            "project"
        ][
            "category_name"
        ]
        == "IT Operations"
    )


def test_project_get_rejects_surrounding_whitespace():

    result = (
        _service(
            lambda request: (
                httpx.Response(
                    500
                )
            )
        )
        .get_project(
            " OPS "
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


def test_project_get_maps_403_to_denied():

    service = (
        _service(
            lambda request: (
                httpx.Response(
                    403
                )
            )
        )
    )

    result = (
        service.get_project(
            "OPS"
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
        "authorization"
        in result[
            "error"
        ].lower()
    )


def test_project_get_maps_404_without_exposing_provider_body():

    provider_secret = (
        "do-not-leak-me"
    )

    service = (
        _service(
            lambda request: (
                httpx.Response(
                    404,
                    json={
                        "errorMessages": [
                            provider_secret,
                        ],
                    },
                )
            )
        )
    )

    result = (
        service.get_project(
            "MISSING"
        )
    )

    assert (
        result[
            "status"
        ]
        == "not_found"
    )

    assert (
        provider_secret
        not in repr(
            result
        )
    )


def test_project_list_rejects_invalid_limit():

    result = (
        _service(
            lambda request: (
                httpx.Response(
                    500
                )
            )
        )
        .list_projects(
            limit=1000
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
