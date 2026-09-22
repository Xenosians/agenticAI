import json

import httpx

from services.jira.project_mutations import (
    JiraProjectMutationService,
)


OLD_NAME = (
    "J2B Create Preflight"
)

NEW_NAME = (
    "J2C Rename Proof"
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


def _project(
    name: str,
):
    return {
        "id":
            "10033",

        "key":
            "J2BTST",

        "name":
            name,

        "projectTypeKey":
            "software",
    }


def _validation_response(
    request: httpx.Request,
):

    if (
        request.url.path
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

    return None


def test_prepare_update_binds_exact_current_project():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/J2BTST"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            OLD_NAME
                        )
                    ),
                )
            )

        validation = (
            _validation_response(
                request
            )
        )

        if validation is not None:
            return (
                validation
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .prepare_update_project(
            project_id_or_key="J2BTST",
            new_name=NEW_NAME,
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
            "risk"
        ]
        == "medium"
    )

    assert (
        result[
            "requires_approval"
        ]
        is True
    )

    assert (
        result[
            "execution_arguments"
        ]
        == {
            "project_id_or_key":
                "J2BTST",

            "new_name":
                NEW_NAME,

            "expected_project_id":
                "10033",

            "expected_project_key":
                "J2BTST",

            "expected_project_name":
                OLD_NAME,
        }
    )


def test_prepare_update_refuses_name_rewrite():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.url.path
            == "/rest/api/3/project/J2BTST"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            OLD_NAME
                        )
                    ),
                )
            )

        if (
            request.url.path
            == (
                "/rest/api/3/projectvalidate/"
                "validProjectName"
            )
        ):
            return (
                httpx.Response(
                    200,
                    content=b"J2C Rename Proof 2",
                    headers={
                        "Content-Type":
                            "application/json;charset=UTF-8",
                    },
                )
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .prepare_update_project(
            project_id_or_key="J2BTST",
            new_name=NEW_NAME,
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
        "will not silently rename"
        in result[
            "error"
        ]
    )


def test_update_revalidates_snapshot_then_renames_exactly():

    project_get_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal project_get_count

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10033"
        ):
            project_get_count += 1

            name = (
                OLD_NAME
                if project_get_count == 1
                else NEW_NAME
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            name
                        )
                    ),
                )
            )

        validation = (
            _validation_response(
                request
            )
        )

        if validation is not None:
            return (
                validation
            )

        if (
            request.method
            == "PUT"
            and request.url.path
            == "/rest/api/3/project/10033"
        ):
            assert (
                json.loads(
                    request.content
                )
                == {
                    "name":
                        NEW_NAME,
                }
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            NEW_NAME
                        )
                    ),
                )
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .update_project(
            project_id_or_key="J2BTST",
            new_name=NEW_NAME,
            expected_project_id="10033",
            expected_project_key="J2BTST",
            expected_project_name=OLD_NAME,
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

    assert (
        result[
            "previous_project_name"
        ]
        == OLD_NAME
    )

    assert (
        result[
            "new_project_name"
        ]
        == NEW_NAME
    )


def test_update_fails_closed_if_project_changed_after_approval():

    put_called = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal put_called

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10033"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            "Someone Else Renamed It"
                        )
                    ),
                )
            )

        if request.method == "PUT":
            put_called = True

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .update_project(
            project_id_or_key="J2BTST",
            new_name=NEW_NAME,
            expected_project_id="10033",
            expected_project_key="J2BTST",
            expected_project_name=OLD_NAME,
        )
    )

    assert (
        put_called
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

    assert (
        "stale"
        in result[
            "error"
        ].lower()
    )


def test_update_timeout_reconciles_exact_new_state():

    project_get_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal project_get_count

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10033"
        ):
            project_get_count += 1

            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            (
                                OLD_NAME
                                if project_get_count == 1
                                else NEW_NAME
                            )
                        )
                    ),
                )
            )

        validation = (
            _validation_response(
                request
            )
        )

        if validation is not None:
            return (
                validation
            )

        if (
            request.method
            == "PUT"
            and request.url.path
            == "/rest/api/3/project/10033"
        ):
            raise (
                httpx.ReadTimeout(
                    "simulated timeout",
                    request=(
                        request
                    ),
                )
            )

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .update_project(
            project_id_or_key="J2BTST",
            new_name=NEW_NAME,
            expected_project_id="10033",
            expected_project_key="J2BTST",
            expected_project_name=OLD_NAME,
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
