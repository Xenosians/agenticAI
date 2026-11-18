"""D2: the tailnet policy text is read with its ETag, validated by Tailscale, and written only
with If-Match, so a concurrent edit is refused instead of overwritten (direction note D2).

The policy endpoints answer HuJSON text; the text itself never leaves the service except as the
exact bytes written back. A refused write never repeats Tailscale's answer.
"""
from datetime import datetime, timezone

import httpx
import pytest

from services.tailnet import PolicyConflict, TailnetConfig, TailnetService, TailscaleRefused

NOW = datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)
TEXT = '// policy\n{\n\t"grants": [\n\t\t{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},\n\t],\n}\n'


def service(handler):
    def wrapped(request):
        if request.url.path == "/api/v2/oauth/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        return handler(request)

    return TailnetService(TailnetConfig("client-id", "client-secret"),
                          httpx.Client(transport=httpx.MockTransport(wrapped)), clock=lambda: NOW)


def test_the_policy_text_is_read_as_hujson_with_its_etag():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers.get("accept")))
        return httpx.Response(200, text=TEXT, headers={"ETag": '"abc123"', "Content-Type": "application/hujson"})

    text, etag = service(handler).policy_text()
    assert (text, etag) == (TEXT, '"abc123"')
    assert seen == [("GET", "/api/v2/tailnet/-/acl", "application/hujson")]


def test_a_policy_without_an_etag_cannot_be_changed_safely():
    with pytest.raises(TailscaleRefused, match="ETag"):
        service(lambda request: httpx.Response(200, text=TEXT)).policy_text()


def test_validation_sends_the_exact_text_and_writes_nothing():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.content.decode(), request.headers.get("content-type")))
        return httpx.Response(200, json={})

    assert service(handler).validate_policy(TEXT) is None
    assert seen == [("POST", "/api/v2/tailnet/-/acl/validate", TEXT, "application/hujson")]


def test_a_policy_tailscale_rejects_names_its_reason_without_echoing_the_text():
    def handler(request):
        return httpx.Response(200, json={"message": "test(s) failed", "data": [{"user": "owner@example.com",
                                                                                  "errors": ["x"]}]})

    error = service(handler).validate_policy(TEXT)
    assert error == "Tailscale rejected the changed policy: test(s) failed."
    assert "example.com" not in error


def test_the_write_carries_if_match_and_the_exact_text():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.headers.get("if-match"), request.content.decode()))
        return httpx.Response(200, text=TEXT, headers={"ETag": '"def456"'})

    assert service(handler).write_policy(TEXT, '"abc123"') == '"def456"'
    assert seen == [("POST", "/api/v2/tailnet/-/acl", '"abc123"', TEXT)]


def test_a_concurrent_edit_is_refused_not_overwritten():
    with pytest.raises(PolicyConflict):
        service(lambda request: httpx.Response(412, json={"message": "precondition failed"})).write_policy(
            TEXT, '"stale"')


def test_a_write_needs_an_etag():
    with pytest.raises(ValueError):
        service(lambda request: httpx.Response(200)).write_policy(TEXT, "")
