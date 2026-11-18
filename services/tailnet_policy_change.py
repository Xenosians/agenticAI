"""A governed tailnet policy change: one typed grant added or removed (direction note D2).

`prepare_grant_change` runs before approval and writes nothing: it reads the policy text and its
ETag, refuses a policy managed outside the admin console, makes the typed edit, has Tailscale
validate the result and freezes the ETag and both digests. `apply_grant_change` runs after
approval: it re-reads the policy and refuses if it moved, re-makes the same edit and checks its
digest, writes it with If-Match and reads it back. Once a write has been sent, a lost answer is an
unknown outcome that is never retried. The policy text never leaves this module.
"""
from __future__ import annotations

import hashlib
from typing import Any

import httpx

from services.tailnet import PolicyConflict, TailnetService, TailscaleRefused
from services.tailnet_policy_edit import (
    GrantSpec,
    PolicyEditRefused,
    add_grant,
    grant_text,
    parse_hujson,
    remove_grant,
)

OPERATIONS = {"add": add_grant, "remove": remove_grant}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _edit(operation: str, text: str, spec: GrantSpec) -> str:
    if operation not in OPERATIONS:
        raise PolicyEditRefused("A tailnet policy change adds or removes one grant.")
    return OPERATIONS[operation](text, spec)


def _refused(status: str, error: str) -> dict[str, Any]:
    return {"ok": False, "status": status, "error": error}


def prepare_grant_change(service: TailnetService, operation: str, spec: GrantSpec) -> dict[str, Any]:
    """Read-only: the exact edit, validated by Tailscale, and the facts the approval freezes."""
    try:
        spec = spec.validated()
        settings = service._get("/tailnet/-/settings")
        if settings.get("aclsExternallyManagedOn") is True:
            return _refused("denied", "The tailnet policy is managed outside the admin console (for example "
                                      "GitOps); change it there, or Janus's write would be overwritten.")
        text, etag = service.policy_text()
        edited = _edit(operation, text, spec)
        rejected = service.validate_policy(edited)
        if rejected:
            return _refused("denied", rejected)
    except PolicyEditRefused as error:
        return _refused("denied", str(error))
    return {"ok": True, "status": "ready", "grant": grant_text(spec),
            "frozen": {"expected_etag": etag, "policy_sha256": _digest(text), "new_policy_sha256": _digest(edited)}}


def apply_grant_change(service: TailnetService, operation: str, spec: GrantSpec, *, expected_etag: str,
                       policy_sha256: str, new_policy_sha256: str) -> dict[str, Any]:
    """The approved change: refused if anything moved, written once with If-Match, then read back."""
    base = {"provider": "tailscale", "operation": f"tailnet_grant_{operation}", "changed": False,
            "verification_ok": False, "retry_safe": True}
    try:
        spec = spec.validated()
        label = grant_text(spec)
        base["grant"] = label
        text, etag = service.policy_text()
        if etag != expected_etag or _digest(text) != policy_sha256:
            return {**base, "ok": False, "status": "conflict",
                    "error": "The tailnet policy changed since the approval; nothing was written. Propose the "
                             "change again from the current policy."}
        edited = _edit(operation, text, spec)
        if _digest(edited) != new_policy_sha256:
            return {**base, "ok": False, "status": "conflict",
                    "error": "The change no longer matches what was approved; nothing was written."}
    except PolicyEditRefused as error:
        return {**base, "ok": False, "status": "conflict", "error": f"{error} Nothing was written."}

    try:
        service.write_policy(edited, etag)
    except PolicyConflict:
        return {**base, "ok": False, "status": "conflict",
                "error": "The tailnet policy was edited at the same time; nothing was written or overwritten."}
    except TailscaleRefused as error:
        return {**base, "ok": False, "status": "failed", "error": str(error)}
    except (httpx.TransportError, RuntimeError):
        return {**base, "ok": False, "status": "outcome_unknown", "retry_safe": False,
                "error": "Tailscale did not confirm the policy write. Read the policy before trying again."}

    base.update(changed=True, retry_safe=False)
    try:
        after, _ = service.policy_text()
        grants = parse_hujson(after).get("grants") or []
    except (httpx.TransportError, RuntimeError, PolicyEditRefused):
        return {**base, "ok": False, "status": "outcome_unknown",
                "error": "Tailscale accepted the policy write, but Janus could not read the policy back. "
                         "Read it before changing it again."}
    present = any(spec.matches(rule) for rule in grants)
    verified = present if operation == "add" else not present
    if not verified:
        return {**base, "ok": False, "status": "outcome_unknown",
                "error": "Tailscale accepted the policy write, but the policy read back does not show the "
                         "change. Read it before changing it again."}
    return {**base, "ok": True, "status": "executed", "verification_ok": True, "error": None}
