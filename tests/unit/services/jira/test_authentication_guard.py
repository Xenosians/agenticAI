"""Jira Cloud anonymous fallback must surface as an authentication failure."""

import httpx
import pytest

from services.jira.authentication import AUTH_EVENT_HOOKS, failed_login_reason


def client(handler):
    return httpx.Client(
        base_url="https://example.atlassian.net",
        auth=("user@example.test", "invalid-token"),
        transport=httpx.MockTransport(handler),
        event_hooks=AUTH_EVENT_HOOKS,
    )


def anonymous_project_search(request):
    # Jira Cloud serves invalid Basic credentials anonymously: HTTP 200, no data.
    return httpx.Response(200, headers={"X-Seraph-LoginReason": "AUTHENTICATED_FAILED"},
                          json={"values": [], "total": 0})


def test_failed_login_becomes_401_before_callers_read_data():
    with client(anonymous_project_search) as jira:
        response = jira.get("/rest/api/3/project/search")
    assert response.status_code == 401


@pytest.mark.parametrize("header", ["OK", "", "AUTHORISATION_FAILED"])
def test_successful_or_unrelated_login_reasons_are_untouched(header):
    def handler(request):
        headers = {"X-Seraph-LoginReason": header} if header else {}
        return httpx.Response(200, headers=headers, json={"values": [{"key": "OPS"}]})

    with client(handler) as jira:
        response = jira.get("/rest/api/3/project/search")
    assert response.status_code == 200
    assert response.json()["values"] == [{"key": "OPS"}]


def test_existing_error_status_is_preserved():
    def handler(request):
        return httpx.Response(403, headers={"X-Seraph-LoginReason": "AUTHENTICATED_FAILED"})

    with client(handler) as jira:
        assert jira.get("/rest/api/3/myself").status_code == 403


def test_multiple_reasons_are_parsed():
    response = httpx.Response(200, headers={"X-Seraph-LoginReason": "OK, AUTHENTICATED_FAILED"})
    assert failed_login_reason(response) == "AUTHENTICATED_FAILED"


def test_project_read_service_reports_authentication_failure():
    from services.jira.projects import JiraProjectReadService

    reader = JiraProjectReadService(
        base_url="https://example.atlassian.net", email="user@example.test",
        api_token="invalid-token", transport=httpx.MockTransport(anonymous_project_search),
    )
    result = reader.list_projects(limit=1)
    assert result["ok"] is False
    assert "authentication" in str(result).lower()
    assert result.get("projects") in (None, [])
