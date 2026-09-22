import httpx

from services.jira.project_mutations import (
    JiraProjectMutationService,
)


PROJECT_ID = "10033"
PROJECT_KEY = "J2BTST"
PROJECT_NAME = "J2C Rename Proof"


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


def _project():
    return {
        "id":
            PROJECT_ID,

        "key":
            PROJECT_KEY,

        "name":
            PROJECT_NAME,

        "projectTypeKey":
            "software",
    }


def _search_response(
    *,
    include: bool,
):
    return {
        "startAt":
            0,

        "maxResults":
            2,

        "total":
            (
                1
                if include
                else 0
            ),

        "isLast":
            True,

        "values":
            (
                [
                    _project(),
                ]
                if include
                else []
            ),
    }


def test_prepare_archive_binds_exact_live_project():

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
        .prepare_archive_project(
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


def test_prepare_archive_rejects_already_archived_project():

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
            f"Unexpected request: "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .prepare_archive_project(
            project_id_or_key=(
                PROJECT_KEY
            )
        )
    )

    assert result["ok"] is False
    assert result["status"] == "denied"

    assert (
        "already archived"
        in result[
            "error"
        ].lower()
    )


def test_archive_revalidates_then_verifies_archived_state():

    archived = False
    post_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal archived
        nonlocal post_count

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
                archived
                if status
                == "archived"
                else not archived
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
            == "POST"
            and request.url.path
            == (
                "/rest/api/3/project/"
                "10033/archive"
            )
        ):

            post_count += 1
            archived = True

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
        .archive_project(
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

    assert post_count == 1
    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["archived"] is True
    assert result["mutation_performed"] is True
    assert result["verification_ok"] is True
    assert result["reconciled"] is False


def test_archive_fails_closed_if_snapshot_is_stale():

    post_called = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal post_called

        if (
            request.method
            == "GET"
            and request.url.path
            == "/rest/api/3/project/10033"
        ):

            payload = (
                _project()
            )

            payload[
                "name"
            ] = (
                "Someone Changed It"
            )

            return (
                httpx.Response(
                    200,
                    json=(
                        payload
                    ),
                )
            )

        if request.method == "POST":
            post_called = True

        raise AssertionError(
            f"Unexpected request: "
            f"{request.method} "
            f"{request.url}"
        )

    result = (
        _service(
            handler
        )
        .archive_project(
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

    assert post_called is False
    assert result["ok"] is False
    assert result["status"] == "denied"

    assert (
        "stale"
        in result[
            "error"
        ].lower()
    )


def test_archive_timeout_reconciles_archived_state():

    archive_attempted = False

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        nonlocal archive_attempted

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

            if not archive_attempted:
                include = (
                    status
                    == "live"
                )
            else:
                include = (
                    status
                    == "archived"
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
            == "POST"
            and request.url.path
            == (
                "/rest/api/3/project/"
                "10033/archive"
            )
        ):

            archive_attempted = True

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
        .archive_project(
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
    assert result["archived"] is True
    assert result["verification_ok"] is True
    assert result["reconciled"] is True
