import json

import httpx
import pytest

from services.atlassian.core import (
    ATLASSIAN_API_ORIGIN,
    AtlassianAdminClient,
    AtlassianAdminService,
    AtlassianCredentialBroker,
    AtlassianProviderError,
    bounded_limit,
    extract_next_cursor,
)


def _service(handler, *, default_org_id="org-123"):
    credentials = AtlassianCredentialBroker(
        admin_api_key="secret-admin-key",
        default_org_id=default_org_id,
        jira_email="user@example.com",
        jira_api_token="secret-jira-token",
    )
    client = AtlassianAdminClient(
        credentials=credentials,
        transport=httpx.MockTransport(handler),
    )
    return AtlassianAdminService(
        credentials=credentials,
        client=client,
    )


def test_redacted_status_never_exposes_secrets():
    admin_secret = "super-secret-admin-key"
    jira_secret = "super-secret-jira-token"
    broker = AtlassianCredentialBroker(
        admin_api_key=admin_secret,
        default_org_id="org-123",
        jira_email="user@example.com",
        jira_api_token=jira_secret,
    )

    result = broker.redacted_status()
    serialized = repr(result)

    assert admin_secret not in serialized
    assert jira_secret not in serialized
    assert result["admin_api_key_configured"] is True
    assert result["jira_basic_configured"] is True
    assert result["secret_values_exposed"] is False


def test_bounded_pagination_and_cursor_parsing():
    assert bounded_limit(None) == 25
    assert bounded_limit(50) == 50

    with pytest.raises(ValueError):
        bounded_limit(51)

    assert extract_next_cursor(
        "https://api.atlassian.com/admin/v1/orgs?cursor=next-123"
    ) == "next-123"
    assert extract_next_cursor("opaque-cursor") == "opaque-cursor"
    assert extract_next_cursor("/unsafe/path?cursor=x") is None


def test_client_uses_fixed_origin_and_bearer_secret():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["url"] = str(request.url)
        observed["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": [], "links": {}})

    client = AtlassianAdminClient(
        credentials=AtlassianCredentialBroker(admin_api_key="secret"),
        transport=httpx.MockTransport(handler),
    )
    client.get_json("/admin/v1/orgs", operation="test")

    assert observed["url"] == f"{ATLASSIAN_API_ORIGIN}/admin/v1/orgs"
    assert observed["authorization"] == "Bearer secret"


def test_client_rejects_full_url_and_sanitizes_provider_error():
    secret = "do-not-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text=f"body contains {secret}")

    client = AtlassianAdminClient(
        credentials=AtlassianCredentialBroker(admin_api_key=secret),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ValueError):
        client.get_json("https://evil.example/path", operation="test")

    with pytest.raises(AtlassianProviderError) as captured:
        client.get_json("/admin/v1/orgs", operation="test")

    assert secret not in str(captured.value)


def test_org_discovery():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/admin/v1/orgs"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "org-123",
                        "type": "orgs",
                        "attributes": {"name": "Example Org"},
                    }
                ],
                "links": {},
            },
        )

    result = _service(handler).list_organizations()

    assert result["ok"] is True
    assert result["organizations"][0]["name"] == "Example Org"


def test_workspace_discovery_uses_default_org_and_structured_query():
    observed = {}

    def handler(request: httpx.Request) -> httpx.Response:
        observed["path"] = request.url.path
        observed["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "ari:cloud:jira::site/1",
                        "attributes": {
                            "name": "acme",
                            "type": "Jira",
                            "typeKey": "jira-software",
                            "status": "online",
                            "hostUrl": "https://acme.atlassian.net",
                            "realm": "US",
                            "regions": ["us-east-1"],
                        },
                    }
                ],
                "links": {},
                "meta": {},
            },
        )

    result = _service(handler).list_workspaces(name="acme")

    assert observed["path"] == "/admin/v2/orgs/org-123/workspaces"
    assert observed["body"]["query"]["field"]["name"] == "attributes.name"
    assert observed["body"]["query"]["field"]["values"] == ["acme"]
    assert result["workspaces"][0]["host_url"] == "https://acme.atlassian.net"


def test_api_token_metadata_never_contains_secret_value():
    provider_secret = (
        "SHOULD-NOT-BE-EXPOSED"
    )

    observed_paths = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:

        observed_paths.append(
            request.url.path
        )

        assert (
            request.url.path.startswith(
                "/admin/api-access/v1/orgs/"
            )
        )

        assert (
            request.url.path.endswith(
                "/api-tokens"
            )
        )

        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id":
                            "token-id-1",

                        "label":
                            "automation",

                        "status":
                            "ALLOWED",

                        "createdAt":
                            "2026-01-01T00:00:00Z",

                        "expiresAt":
                            "2026-12-31T00:00:00Z",

                        "lastActiveAt":
                            "2026-09-22T00:00:00Z",

                        "user": {
                            "id":
                                "account-1",
                        },

                        "scopes": [
                            "read:jira-work",
                        ],

                        # Unexpected provider field.
                        # Trusted parsing must never expose it.
                        "token":
                            provider_secret,
                    }
                ],
            },
        )

    result = (
        _service(
            handler
        )
        .list_api_token_metadata()
    )

    assert (
        len(
            observed_paths
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
            "organization_id"
        ]
    )

    assert (
        result[
            "count"
        ]
        == 1
    )

    assert (
        result[
            "tokens"
        ][
            0
        ][
            "id"
        ]
        == "token-id-1"
    )

    assert (
        result[
            "tokens"
        ][
            0
        ][
            "user_id"
        ]
        == "account-1"
    )

    assert (
        result[
            "secret_values_exposed"
        ]
        is False
    )

    assert (
        "account_id"
        not in result
    )

    assert (
        provider_secret
        not in repr(
            result
        )
    )
