from __future__ import annotations

import json
import httpx

from services.atlassian import (
    AtlassianAdminClient,
    AtlassianAdminService,
    AtlassianCredentialBroker,
)


def test_jira_user_invite_uses_trusted_org_and_resource() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["authorization"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "invite-1",
                        "email": "maya@example.com",
                        "results": [],
                    }
                ]
            },
        )

    credentials = AtlassianCredentialBroker(
        admin_api_key="secret-admin-key",
        default_org_id="org-123",
    )
    client = AtlassianAdminClient(
        credentials=credentials,
        transport=httpx.MockTransport(handler),
    )
    service = AtlassianAdminService(
        credentials=credentials,
        client=client,
        jira_resource_ari=(
            "ari:cloud:jira::site/70ef3a32-d0da-4e09-b35e-0109f91969c3"
        ),
    )

    result = service.invite_jira_user(email="maya@example.com")

    assert result["ok"] is True
    assert result["status"] == "success"
    assert result["email"] == "maya@example.com"
    assert result["invitation_id"] == "invite-1"
    assert seen["method"] == "POST"
    assert seen["path"] == "/admin/v2/orgs/org-123/users/invite"
    assert seen["authorization"] == "Bearer secret-admin-key"
    assert seen["body"]["emails"] == ["maya@example.com"]
    assert seen["body"]["permissionRules"] == [
        {
            "resource": (
                "ari:cloud:jira::site/70ef3a32-d0da-4e09-b35e-0109f91969c3"
            ),
            "role": "atlassian/user",
        }
    ]
    assert seen["body"]["sendNotification"] is True


def test_jira_user_invite_fails_closed_without_resource_ari() -> None:
    credentials = AtlassianCredentialBroker(
        admin_api_key="secret-admin-key",
        default_org_id="org-123",
    )

    def should_not_call(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("provider call must not occur")

    client = AtlassianAdminClient(
        credentials=credentials,
        transport=httpx.MockTransport(should_not_call),
    )
    service = AtlassianAdminService(
        credentials=credentials,
        client=client,
        jira_resource_ari=None,
    )

    result = service.invite_jira_user(email="maya@example.com")
    assert result["ok"] is False
    assert result["status"] == "denied"
    assert "ATLASSIAN_JIRA_RESOURCE_ARI" in result["error"]
