import httpx

from services.jira.project_mutations import (
    JiraProjectMutationService,
)


PROJECT_ID = "10040"
PROJECT_KEY = "J2DDEL"
PROJECT_NAME = "J2D Delete Proof"


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
    *,
    name=PROJECT_NAME,
):
    return {
        "id":
            PROJECT_ID,

        "key":
            PROJECT_KEY,

        "name":
            name,
    }


def _search_response(
    *,
    include: bool,
):
    return {
        "values":
            (
                [
                    _project()
                ]
                if include
                else []
            )
    }


def test_prepare_delete_binds_exact_live_snapshot():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/J2DDEL"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project()
                    ),
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10040"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project()
                    ),
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/search"
        ):
            status = (
                request.url.params.get(
                    "status"
                )
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        _search_response(
                            include=(
                                status
                                == "live"
                            )
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
        .prepare_delete_project(
            project_id_or_key=(
                PROJECT_KEY
            )
        )
    )

    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["risk"] == "high"
    assert result["requires_approval"] is True

    assert (
        result[
            "execution_arguments"
        ]
        == {
            "project_id_or_key":
                PROJECT_KEY,

            "expected_project_id":
                PROJECT_ID,

            "expected_project_key":
                PROJECT_KEY,

            "expected_project_name":
                PROJECT_NAME,
        }
    )


def test_prepare_delete_rejects_archived_project():

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        if (
            request.url.path
            == "/rest/api/3/project/J2DDEL"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project()
                    ),
                )
            )

        if (
            request.url.path
            == "/rest/api/3/project/10040"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project()
                    ),
                )
            )

        if (
            request.url.path
            == "/rest/api/3/project/search"
        ):
            status = (
                request.url.params.get(
                    "status"
                )
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        _search_response(
                            include=(
                                status
                                == "archived"
                            )
                        )
                    ),
                )
            )

        raise AssertionError(
            f"Unexpected request: {request.url}"
        )

    result = (
        _service(
            handler
        )
        .prepare_delete_project(
            project_id_or_key=(
                PROJECT_KEY
            )
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"

    assert (
        "archived"
        in result["error"].lower()
    )


def test_delete_uses_undo_enabled_and_verifies_absence():

    deleted = False
    delete_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal deleted
        nonlocal delete_count

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10040"
        ):
            return (
                httpx.Response(
                    404
                )
                if deleted
                else httpx.Response(
                    200,
                    json=(
                        _project()
                    ),
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/search"
        ):
            status = (
                request.url.params.get(
                    "status"
                )
            )

            include = (
                (
                    not deleted
                )
                and status
                == "live"
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        _search_response(
                            include=(
                                include
                            )
                        )
                    ),
                )
            )

        if (
            request.method
            == "DELETE"
            and request.url.path
            == "/rest/api/3/project/10040"
        ):
            delete_count += 1

            assert (
                request.url.params.get(
                    "enableUndo"
                )
                == "true"
            )

            deleted = True

            return (
                httpx.Response(
                    204
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
        .delete_project(
            project_id_or_key=(
                PROJECT_KEY
            ),
            expected_project_id=(
                PROJECT_ID
            ),
            expected_project_key=(
                PROJECT_KEY
            ),
            expected_project_name=(
                PROJECT_NAME
            ),
        )
    )

    assert delete_count == 1
    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["operation"] == "delete"
    assert result["deleted"] is True
    assert result["mutation_performed"] is True
    assert result["verification_ok"] is True
    assert result["reconciled"] is False


def test_delete_fails_closed_on_stale_snapshot():

    delete_called = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal delete_called

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10040"
        ):
            return (
                httpx.Response(
                    200,
                    json=(
                        _project(
                            name=(
                                "Someone Changed It"
                            )
                        )
                    ),
                )
            )

        if request.method == "DELETE":
            delete_called = True

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .delete_project(
            project_id_or_key=(
                PROJECT_KEY
            ),
            expected_project_id=(
                PROJECT_ID
            ),
            expected_project_key=(
                PROJECT_KEY
            ),
            expected_project_name=(
                PROJECT_NAME
            ),
        )
    )

    assert delete_called is False
    assert result["ok"] is False
    assert result["status"] == "denied"

    assert (
        "stale"
        in result["error"].lower()
    )


def test_delete_timeout_reconciles_deleted_state():

    deleted = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal deleted

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10040"
        ):
            return (
                httpx.Response(
                    404
                )
                if deleted
                else httpx.Response(
                    200,
                    json=(
                        _project()
                    ),
                )
            )

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/search"
        ):
            status = (
                request.url.params.get(
                    "status"
                )
            )

            include = (
                (
                    not deleted
                )
                and status
                == "live"
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        _search_response(
                            include=(
                                include
                            )
                        )
                    ),
                )
            )

        if (
            request.method
            == "DELETE"
            and request.url.path
            == "/rest/api/3/project/10040"
        ):
            assert (
                request.url.params.get(
                    "enableUndo"
                )
                == "true"
            )

            deleted = True

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
        .delete_project(
            project_id_or_key=(
                PROJECT_KEY
            ),
            expected_project_id=(
                PROJECT_ID
            ),
            expected_project_key=(
                PROJECT_KEY
            ),
            expected_project_name=(
                PROJECT_NAME
            ),
        )
    )

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["deleted"] is True
    assert result["mutation_performed"] is True
    assert result["verification_ok"] is True
    assert result["reconciled"] is True
