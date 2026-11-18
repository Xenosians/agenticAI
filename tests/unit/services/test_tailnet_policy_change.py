"""D2: a governed tailnet policy change, from the frozen proposal to the read-back (direction note D2).

Before approval the trusted policy reads the policy text and its ETag, makes the typed edit,
has Tailscale validate the result (nothing is written) and freezes the ETag and both digests.
After approval the change re-reads the policy and refuses if it moved, re-makes the same edit and
checks its digest, writes it with If-Match, then reads it back. A lost answer after the write is an
unknown outcome that must not be retried.
"""
import hashlib

import httpx
import pytest

from services.tailnet import PolicyConflict, TailnetConfig, TailnetService
from services.tailnet_policy_change import apply_grant_change, prepare_grant_change
from services.tailnet_policy_edit import GrantSpec, add_grant, parse_hujson

TEXT = '// policy\n{\n\t"grants": [\n\t\t{"src": ["autogroup:member"], "dst": ["*"], "ip": ["*"]},\n\t],\n}\n'
LAB = GrantSpec(("tag:janus-lab",), ("tag:janus-lab",), ("22",))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Tailnet:
    """Stands in for Tailscale's policy endpoints, keeping one stored policy and its version."""

    def __init__(self, text=TEXT, *, external=False, reject=None, lose_write=False, conflict=False,
                 lose_readback=False):
        self.text, self.version = text, 1
        self.external, self.reject, self.lose_write = external, reject, lose_write
        self.conflict, self.lose_readback = conflict, lose_readback
        self.writes = []
        self.reads_after_write = 0

    def etag(self):
        return f'"v{self.version}"'

    def handler(self, request):
        path = request.url.path
        if path == "/api/v2/oauth/token":
            return httpx.Response(200, json={"access_token": "token", "expires_in": 3600})
        if path == "/api/v2/tailnet/-/settings":
            return httpx.Response(200, json={"aclsExternallyManagedOn": self.external})
        if path == "/api/v2/tailnet/-/acl/validate":
            return httpx.Response(200, json={"message": self.reject} if self.reject else {})
        if path == "/api/v2/tailnet/-/acl" and request.method == "GET":
            if self.writes:
                self.reads_after_write += 1
                if self.lose_readback:
                    raise httpx.ReadTimeout("lost")
            return httpx.Response(200, text=self.text, headers={"ETag": self.etag()})
        if path == "/api/v2/tailnet/-/acl" and request.method == "POST":
            if self.conflict or request.headers.get("if-match") != self.etag():
                return httpx.Response(412, json={"message": "precondition failed"})
            self.writes.append(request.content.decode())
            self.text, self.version = request.content.decode(), self.version + 1
            if self.lose_write:
                raise httpx.ReadTimeout("lost after send")
            return httpx.Response(200, text=self.text, headers={"ETag": self.etag()})
        raise AssertionError(path)

    def service(self):
        return TailnetService(TailnetConfig("id", "secret"), httpx.Client(transport=httpx.MockTransport(self.handler)))


def frozen(tailnet, operation="add", spec=LAB):
    prepared = prepare_grant_change(tailnet.service(), operation, spec)
    assert prepared["ok"] is True, prepared
    return prepared["frozen"]


def test_the_proposal_freezes_the_version_and_both_digests_and_writes_nothing():
    tailnet = Tailnet()
    prepared = prepare_grant_change(tailnet.service(), "add", LAB)
    assert prepared["ok"] is True and tailnet.writes == []
    assert prepared["frozen"] == {"expected_etag": '"v1"', "policy_sha256": sha(TEXT),
                                  "new_policy_sha256": sha(add_grant(TEXT, LAB))}
    assert prepared["grant"] == "tag:janus-lab -> tag:janus-lab : 22"


@pytest.mark.parametrize("tailnet,words", [
    (Tailnet(external=True), "managed outside"),
    (Tailnet(reject="test(s) failed"), "Tailscale rejected"),
])
def test_a_policy_managed_elsewhere_or_rejected_by_tailscale_is_refused_before_approval(tailnet, words):
    prepared = prepare_grant_change(tailnet.service(), "add", LAB)
    assert (prepared["ok"], prepared["status"]) == (False, "denied") and words in prepared["error"]
    assert tailnet.writes == []


def test_an_impossible_edit_is_refused_before_approval():
    prepared = prepare_grant_change(Tailnet().service(), "remove", LAB)
    assert (prepared["ok"], prepared["status"]) == (False, "denied") and "no grant" in prepared["error"]


def test_the_approved_change_is_written_with_if_match_and_read_back():
    tailnet = Tailnet()
    result = apply_grant_change(tailnet.service(), "add", LAB, **frozen(tailnet))
    assert (result["ok"], result["status"], result["changed"], result["verification_ok"]) == \
        (True, "executed", True, True)
    assert result["retry_safe"] is False and result["grant"] == "tag:janus-lab -> tag:janus-lab : 22"
    assert parse_hujson(tailnet.text)["grants"][-1] == {"src": ["tag:janus-lab"], "dst": ["tag:janus-lab"],
                                                        "ip": ["22"]}
    assert tailnet.writes == [add_grant(TEXT, LAB)]
    # And the same grant removed again leaves the policy text exactly as it was.
    removed = apply_grant_change(tailnet.service(), "remove", LAB, **frozen(tailnet, "remove"))
    assert removed["ok"] is True and tailnet.text == TEXT


def test_a_policy_that_moved_since_approval_is_refused_and_nothing_is_written():
    tailnet = Tailnet()
    approved = frozen(tailnet)
    tailnet.text, tailnet.version = TEXT.replace("// policy", "// edited by hand"), 2
    result = apply_grant_change(tailnet.service(), "add", LAB, **approved)
    assert (result["ok"], result["status"], result["changed"]) == (False, "conflict", False)
    assert result["retry_safe"] is True and tailnet.writes == []


def test_a_concurrent_edit_during_the_write_is_refused_not_overwritten():
    tailnet = Tailnet()
    approved = frozen(tailnet)
    tailnet.conflict = True
    result = apply_grant_change(tailnet.service(), "add", LAB, **approved)
    assert (result["ok"], result["status"], result["changed"]) == (False, "conflict", False)
    assert tailnet.writes == []


@pytest.mark.parametrize("lost", ["lose_write", "lose_readback"])
def test_a_lost_answer_after_the_write_is_an_unknown_outcome_never_retried(lost):
    tailnet = Tailnet()
    approved = frozen(tailnet)
    setattr(tailnet, lost, True)
    result = apply_grant_change(tailnet.service(), "add", LAB, **approved)
    assert (result["ok"], result["status"], result["retry_safe"]) == (False, "outcome_unknown", False)


def test_a_tampered_digest_is_refused_before_anything_is_written():
    tailnet = Tailnet()
    approved = {**frozen(tailnet), "new_policy_sha256": "0" * 64}
    result = apply_grant_change(tailnet.service(), "add", LAB, **approved)
    assert (result["ok"], result["status"]) == (False, "conflict") and tailnet.writes == []


def test_the_receipt_never_carries_the_policy_text():
    tailnet = Tailnet()
    result = apply_grant_change(tailnet.service(), "add", LAB, **frozen(tailnet))
    assert "autogroup:member" not in str({key: value for key, value in result.items() if key != "grant"})
    with pytest.raises(PolicyConflict):
        tailnet.service().write_policy(TEXT, '"v0"')
