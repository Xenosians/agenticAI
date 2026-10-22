"""Documented invitation results are authority; a 2xx envelope alone is not."""
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from services.atlassian import AtlassianAdminClient, AtlassianAdminService, AtlassianCredentialBroker
from tools.atlassian.catalog import ATLASSIAN_TOOLS
from tools.atlassian.mcp import register_atlassian_tools


RESOURCE = "ari:cloud:jira::site/test-site"
EMAIL = "requested@example.test"


def receipt():
    # OpenAPI: AdminInviteResponse.id is an account ID; role/group status is
    # INVITED or ERROR. No notification delivery receipt exists in this schema.
    return {"data": [{"id": "account-123", "email": EMAIL, "results": [{
        "roleAssignmentResult": [{"resource": RESOURCE, "role": "atlassian/user", "status": "INVITED"}],
        "groupAssignmentResult": [],
    }]}]}


def service(handler):
    credentials = AtlassianCredentialBroker(admin_api_key="local-test-only", default_org_id="test-org")
    client = AtlassianAdminClient(credentials=credentials, transport=httpx.MockTransport(handler))
    return AtlassianAdminService(credentials=credentials, client=client, jira_resource_ari=RESOURCE)


def invoke(response):
    calls = []
    def handler(request):
        calls.append(request)
        if isinstance(response, Exception):
            raise response
        return response
    adapter = service(handler)
    try:
        result = adapter.invite_jira_user(email=EMAIL)
    finally:
        adapter.client.client.close()
    assert len(calls) == 1  # Never resubmit an uncertain mutation.
    return result


def test_verified_receipt_distinguishes_account_identity_and_notification_request():
    result = invoke(httpx.Response(200, json=receipt()))
    assert result["ok"] is True and result["status"] == "success"
    assert result["account_id"] == "account-123" and result["invitation_id"] is None
    assert result["verification_scope"] == "provider_invitation_receipt"
    assert result["verification_ok"] is True and result["role_assignment_status"] == "INVITED"
    assert result["notification_requested"] is True
    assert result["notification_sent_by_provider"] is None
    assert result["notification_delivery_verified"] is False
    text = ATLASSIAN_TOOLS["atlassian_user_invite"]["result_formatter"](result)
    assert "sending and delivery are not confirmed" in text


def test_documented_canonical_product_member_role_is_accepted():
    payload = receipt()
    payload["data"][0]["results"][0]["roleAssignmentResult"][0]["role"] = "ari:cloud:jira::role/product/member"
    result = invoke(httpx.Response(200, json=payload))
    assert result["ok"] is True


@pytest.mark.parametrize("case", ["empty_item", "no_account_id", "wrong_email", "duplicate_recipient", "empty_results", "wrong_resource", "admin_role", "role_error", "group_error", "missing_role_status", "unknown_result_shape"])
def test_unverified_receipt_is_uncertain_instead_of_success(case):
    payload = receipt()
    item = payload["data"][0]
    assignments = item["results"][0]
    role = assignments["roleAssignmentResult"][0]
    if case == "empty_item":
        payload["data"] = [{}]
    elif case == "no_account_id":
        del item["id"]
    elif case == "wrong_email":
        item["email"] = "different@example.test"
    elif case == "duplicate_recipient":
        payload["data"].append(deepcopy(item))
    elif case == "empty_results":
        item["results"] = []
    elif case == "wrong_resource":
        role["resource"] = "ari:cloud:jira::site/other-site"
    elif case == "admin_role":
        role["role"] = "atlassian/admin"
    elif case == "role_error":
        role["status"] = "ERROR"
        role["statusReason"] = "private provider diagnostic must not leak"
    elif case == "group_error":
        assignments["groupAssignmentResult"] = [{"group": "group-123", "status": "ERROR"}]
    elif case == "missing_role_status":
        del role["status"]
    elif case == "unknown_result_shape":
        item["results"] = [{"status": "ERROR"}]
    result = invoke(httpx.Response(200, json=payload))
    assert result["ok"] is False and result["status"] == "outcome_unknown"
    assert result["verification_ok"] is False and result["retry_safe"] is False
    assert result["changed"] is None and result["mutation_performed"] is None
    assert result["notification_sent_by_provider"] is None
    assert "private provider diagnostic" not in str(result)


@pytest.mark.parametrize("status", [201, 202, 204, 206, 302, 408, 500, 502, 503])
def test_partial_unexpected_success_redirect_and_server_errors_do_not_confirm_mutation(status):
    result = invoke(httpx.Response(status, json=receipt()))
    assert result["ok"] is False and result["status"] == "outcome_unknown"
    assert result["retry_safe"] is False and result["changed"] is None


def test_invalid_json_after_200_is_unknown_not_ordinary_error():
    result = invoke(httpx.Response(200, content=b"invalid receipt"))
    assert result["status"] == "outcome_unknown" and result["retry_safe"] is False


def test_all_assignment_outcomes_must_succeed_even_when_target_role_is_invited():
    payload = receipt()
    payload["data"][0]["results"][0]["roleAssignmentResult"].append({
        "resource": RESOURCE, "role": "atlassian/user", "status": "ERROR",
    })
    result = invoke(httpx.Response(200, json=payload))
    assert result["status"] == "outcome_unknown" and result["verification_ok"] is False


def test_positive_group_assignment_outcomes_do_not_hide_role_verification():
    payload = receipt()
    payload["data"][0]["results"][0]["groupAssignmentResult"] = [{"group": "default-group", "status": "INVITED"}]
    assert invoke(httpx.Response(200, json=payload))["verification_ok"] is True


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 409, 429])
def test_explicit_provider_refusal_is_not_reported_as_success_or_uncertain(status):
    result = invoke(httpx.Response(status, json={"errors": [{"message": "private raw provider refusal"}]}))
    assert result["ok"] is False and result["status"] != "outcome_unknown"
    assert result["changed"] is False and result["mutation_performed"] is False
    assert "private raw provider refusal" not in str(result)


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("response lost"), httpx.ReadError("response lost"), httpx.WriteError("partial write")])
def test_ambiguous_transport_errors_disable_retry(failure):
    result = invoke(failure)
    assert result["status"] == "outcome_unknown" and result["retry_safe"] is False
    assert result["mutation_performed"] is None
    assert result["notification_requested"] is None and result["request_dispatched"] is None


@pytest.mark.parametrize("failure", [httpx.ConnectError("not connected"), httpx.ConnectTimeout("not connected"), httpx.PoolTimeout("not dispatched")])
def test_pre_dispatch_errors_remain_known_not_sent(failure):
    result = invoke(failure)
    assert result["ok"] is False and result["status"] == "error"
    assert result["retry_safe"] is True and result["mutation_performed"] is False
    assert result["notification_requested"] is False and result["request_dispatched"] is False


def test_mcp_preserves_nullable_delivery_and_unknown_mutation_outcomes():
    functions = {}
    server = SimpleNamespace(tool=lambda: lambda function: functions.setdefault(function.__name__, function))
    adapter = service(lambda request: httpx.Response(200, json={"data": [{}]}))
    try:
        register_atlassian_tools(server, adapter)
        result = functions["atlassian_user_invite"](EMAIL)
    finally:
        adapter.client.client.close()
    assert result.status == "outcome_unknown" and result.changed is None
    assert result.notification_sent_by_provider is None
    assert result.verification_ok is False and result.retry_safe is False
    text = ATLASSIAN_TOOLS["atlassian_user_invite"]["result_formatter"](result.model_dump())
    assert "Reconcile" in text and "before retrying" in text
