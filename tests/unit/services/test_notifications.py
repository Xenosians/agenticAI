import json

import httpx
import pytest

from config import Settings
from services.notifications import (
    NOTIFICATION_PATH,
    UserNotificationService,
    expected_outbox_id,
    new_dedupe_key,
)
from subagents.core.tooling.gateway import resolve_policy_execution_arguments
from tools.operations import policy as operations_policy
from tools.operations.catalog import OPERATIONS_TOOLS, format_notify_user_approval, format_notify_user_result
from tools.registry import TOOLS


TOKEN = "internal-secret-token"
EMAIL = "matrix.user@itsm.test"
KEY = "a" * 64


def settings(**overrides):
    values = {"phoenix_base_url": "http://127.0.0.1:4000", "itsm_internal_job_token": TOKEN}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def service(handler, **overrides):
    seen = []

    def record(request):
        seen.append(request)
        return handler(request)

    return UserNotificationService(settings(**overrides), transport=httpx.MockTransport(record)), seen


def accepted(request):
    body = json.loads(request.content)
    return httpx.Response(202, json={"queued": True, "outbox_id": expected_outbox_id(body["dedupe_key"]),
                                   "delivery_status": "pending", "reconciliation_required": False})


def test_notify_posts_exact_payload_with_internal_token_and_reports_queued():
    svc, seen = service(accepted)
    result = svc.notify(email=EMAIL, message="Backup finished.", subject="Status", dedupe_key=KEY)
    assert result["ok"] is True and result["status"] == "executed" and result["queued"] is True
    assert result["outbox_id"] == expected_outbox_id(KEY)
    request = seen[0]
    assert request.url.path == NOTIFICATION_PATH and request.headers["x-internal-token"] == TOKEN
    assert json.loads(request.content) == {"email": EMAIL, "message": "Backup finished.", "subject": "Status",
                                           "dedupe_key": KEY}
    assert TOKEN not in json.dumps(result)
    text = format_notify_user_result(result)
    assert "queued for delivery" in text and "not confirmed as delivered" in text


def test_notify_omits_absent_subject():
    svc, seen = service(accepted)
    assert svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)["queued"] is True
    assert "subject" not in json.loads(seen[0].content)


@pytest.mark.parametrize("http_status,delivery,queued,ok,status,error", [
    (202, "pending", True, True, "executed", None),
    (202, "processing", False, True, "executed", None),
    (200, "sent", False, True, "executed", None),
    (409, "dead", False, False, "error", "notification_delivery_failed"),
    (409, "outcome_unknown", False, False, "outcome_unknown", "notification_delivery_unknown"),
])
def test_replayed_outbox_states_remain_truthful(http_status, delivery, queued, ok, status, error):
    body = {"outbox_id": expected_outbox_id(KEY), "queued": queued, "delivery_status": delivery,
            "reconciliation_required": delivery == "outcome_unknown"}
    if error:
        body["error"] = error
    svc, seen = service(lambda _: httpx.Response(http_status, json=body))
    result = svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)
    assert result["ok"] is ok and result["status"] == status
    assert result["delivery_status"] == delivery and result["queued"] is queued
    assert result["retry_safe"] is False and len(seen) == 1
    if delivery == "sent":
        assert "provider acceptance" in format_notify_user_result(result)
        assert "Inbox delivery is not confirmed" in format_notify_user_result(result)
    if delivery in {"dead", "outcome_unknown"}:
        assert "queued for delivery" not in format_notify_user_result(result)


@pytest.mark.parametrize("patch", [
    {"queued": True}, {"delivery_status": "invented"}, {"outbox_id": "wrong"},
    {"reconciliation_required": True}, {"queued": 0},
])
def test_contradictory_or_unbound_receipts_do_not_become_success(patch):
    body = {"outbox_id": expected_outbox_id(KEY), "queued": False, "delivery_status": "sent",
            "reconciliation_required": False, **patch}
    svc, _ = service(lambda _: httpx.Response(200, json=body))
    result = svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)
    assert result["ok"] is False and result["status"] == "outcome_unknown"


def test_typed_tool_result_preserves_reconciliation_state():
    from tools.operations.mcp import NotificationResult
    body = {"outbox_id": expected_outbox_id(KEY), "queued": False, "delivery_status": "outcome_unknown",
            "reconciliation_required": True, "error": "notification_delivery_unknown"}
    svc, _ = service(lambda _: httpx.Response(409, json=body))
    result = NotificationResult(**svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)).model_dump()
    assert result["reconciliation_required"] is True and result["retry_safe"] is False
    assert result["delivery_status"] == "outcome_unknown"


@pytest.mark.parametrize("status, body, expected_status, code", [
    (404, {"error": "recipient_not_found"}, "denied", "recipient_not_found"),
    (404, None, "error", "endpoint_unavailable"),
    (422, {"error": "invalid_request", "field": "message"}, "denied", "invalid_arguments"),
    (409, {"error": "dedupe_conflict"}, "denied", "dedupe_conflict"),
    (401, {"error": "unauthorized"}, "error", "provider_auth_failed"),
    (500, {"error": "notification_enqueue_failed"}, "outcome_unknown", "outcome_unknown"),
    (202, {"queued": True, "outbox_id": "other"}, "outcome_unknown", "outcome_unknown"),
])
def test_backend_answers_map_to_honest_results(status, body, expected_status, code):
    svc, _ = service(lambda request: httpx.Response(status, json=body) if body is not None else httpx.Response(status))
    result = svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)
    assert result["ok"] is False and result["queued"] is False
    assert (result["status"], result["error_code"]) == (expected_status, code)


def test_transport_failures_distinguish_not_sent_from_unknown():
    def refused(request):
        raise httpx.ConnectError("refused")

    def slow(request):
        raise httpx.ReadTimeout("slow")

    not_sent = service(refused)[0].notify(email=EMAIL, message="Hello", dedupe_key=KEY)
    assert (not_sent["status"], not_sent["error_code"]) == ("error", "provider_unavailable")
    unknown = service(slow)[0].notify(email=EMAIL, message="Hello", dedupe_key=KEY)
    assert (unknown["status"], unknown["error_code"]) == ("outcome_unknown", "outcome_unknown")
    assert "may or may not" in format_notify_user_result(unknown) or "unknown" in format_notify_user_result(unknown)


def test_notify_refuses_without_trusted_dedupe_key_or_with_invalid_input():
    svc, seen = service(accepted)
    assert svc.notify(email=EMAIL, message="Hello")["error_code"] == "missing_trusted_preparation"
    assert svc.notify(email="not-an-email", message="Hello", dedupe_key=KEY)["error_code"] == "invalid_arguments"
    assert svc.notify(email=EMAIL, message=" ", dedupe_key=KEY)["error_code"] == "invalid_arguments"
    assert svc.notify(email=EMAIL, message="x", subject="a\nBcc: b@c.test", dedupe_key=KEY)["error_code"] == "invalid_arguments"
    assert seen == []


def test_notify_requires_phoenix_configuration_and_safe_transport():
    svc, seen = service(accepted, phoenix_base_url=None)
    assert svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)["error_code"] == "not_configured"
    svc, seen = service(accepted, phoenix_base_url="http://phoenix.example.test")
    assert svc.notify(email=EMAIL, message="Hello", dedupe_key=KEY)["error_code"] == "not_configured"
    assert seen == []


def test_policy_freezes_a_fresh_dedupe_key_and_keeps_user_arguments(monkeypatch):
    monkeypatch.setattr(operations_policy, "Settings", lambda: settings())
    first = operations_policy.evaluate_notify_user_policy(email=EMAIL, message="Hello")
    second = operations_policy.evaluate_notify_user_policy(email=EMAIL, message="Hello")
    assert first["ok"] is True and first["risk"] == "medium" and first["requires_approval"] is True
    key = first["execution_arguments"]["dedupe_key"]
    assert len(key) == 64 and key != second["execution_arguments"]["dedupe_key"]
    resolved, error = resolve_policy_execution_arguments(
        tool_name="notify_user", tool=TOOLS["notify_user"], original_arguments={"email": EMAIL, "message": "Hello"},
        policy_result=first)
    assert error is None and resolved == {"email": EMAIL, "message": "Hello", "dedupe_key": key}


def test_policy_refuses_invalid_requests_and_missing_configuration(monkeypatch):
    monkeypatch.setattr(operations_policy, "Settings", lambda: settings())
    denied = operations_policy.evaluate_notify_user_policy(email="nobody", message="Hello")
    assert denied["ok"] is False and denied["status"] == "denied"
    monkeypatch.setattr(operations_policy, "Settings", lambda: settings(itsm_internal_job_token=None))
    missing = operations_policy.evaluate_notify_user_policy(email=EMAIL, message="Hello")
    assert missing["ok"] is False and "ITSM_INTERNAL_JOB_TOKEN" in missing["error"]


def test_catalog_entry_is_an_approval_gated_prepared_mutation():
    spec = OPERATIONS_TOOLS["notify_user"]
    assert TOOLS["notify_user"] is spec
    assert spec["risk"] == "medium" and spec["requires_approval"] is True and spec["effect"] == "mutation"
    assert spec["trusted_policy_arguments"] == ["dedupe_key"] and "dedupe_key" not in spec["parameters"]
    assert set(spec["grounded_arguments"]) == set(spec["parameters"]) == {"email", "message", "subject"}
    text = format_notify_user_approval({"email": EMAIL, "message": "Disk check finished.", "dedupe_key": KEY})
    assert EMAIL in text and '"Disk check finished."' in text and "does not confirm delivery" in text
    assert KEY not in text


def test_dedupe_key_binds_the_request_but_differs_per_preparation():
    assert new_dedupe_key(EMAIL, "Hello") != new_dedupe_key(EMAIL, "Hello")
