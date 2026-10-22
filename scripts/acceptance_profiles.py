"""Owner-scoped, reversible model selection for live acceptance.

Uses the authenticated product API. Catalog fingerprints contain hashes rather
than connection URLs, credentials or personal data. The extensions API has no
compare-and-set operation: every write is guarded by a fresh observation and
verified afterward, but callers must still use an exclusively owned acceptance
account. A detected concurrent edit is never overwritten during restoration.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from services.extensions.store import LOCAL_ROLE_MODEL

ADAPTERS = frozenset({"local", "http", "pi", "opencode"})
NETWORKING_ROLE = "networking-specialist"


class ProfileScopeError(RuntimeError):
    """A safe, non-secret explanation of incomplete acceptance setup."""


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class ProfileSnapshot:
    catalog: dict
    selected: dict
    roles: dict
    resources: dict

    def resource(self, identity: str) -> dict:
        resource = self.resources.get(identity)
        if (not isinstance(resource, dict) or resource.get("kind") != "model"
                or resource.get("state") != "ready" or resource.get("backend") not in ADAPTERS
                or not _identifier(resource.get("model"))):
            raise ProfileScopeError("Selected model is missing, unavailable or malformed")
        return resource

    def effective_networking(self) -> str:
        if NETWORKING_ROLE in self.roles:
            return self.roles[NETWORKING_ROLE]
        selected = self.selected["model_id"]
        if selected and self.resource(selected).get("networking_inference") == "selected_model":
            return selected
        return LOCAL_ROLE_MODEL

    def fingerprint(self) -> dict:
        identities = set(self.roles.values()) - {LOCAL_ROLE_MODEL}
        identities.update(value for value in self.selected.values() if value is not None)
        # Public resource payloads are hashed whole so options and URL changes
        # invalidate evidence. Do not return these payloads in a report.
        resources = {identity: self.resources[identity] for identity in sorted(identities)}
        material = {"selected": self.selected, "roles": self.roles, "resources": resources,
                    "local_profiles": self.catalog["local_profiles"],
                    "specialist_roles": self.catalog["specialist_roles"]}
        main = self.selected["model_id"]
        network = self.effective_networking()
        network_backend = "local" if network == LOCAL_ROLE_MODEL else self.resource(network)["backend"]
        return {"schema": "acceptance-inference-profile.v1", "sha256": _digest(material),
                "adapter": self.resource(main)["backend"] if main else "local",
                "model_identity_sha256": _digest(main),
                "networking": {"adapter": network_backend,
                               "model_identity_sha256": _digest(network),
                               "follows_main": bool(main and network == main)},
                "dataset_selected": self.selected["dataset_id"] is not None}


def inspect_profile(client) -> ProfileSnapshot:
    """Read the current authenticated owner's catalog without exposing its data."""
    try:
        response = client.client.get("/api/v1/extensions")
        response.raise_for_status()
        catalog = response.json()
    except Exception:
        raise ProfileScopeError("Unable to inspect authenticated inference selection") from None
    if (not isinstance(catalog, dict) or catalog.get("schema") != "extension-catalog.v1"
            or not isinstance(catalog.get("resources"), list)
            or not isinstance(catalog.get("specialist_roles"), list)
            or not isinstance(catalog.get("local_profiles"), list)
            or not isinstance(catalog.get("selected"), dict)
            or set(catalog["selected"]) != {"model_id", "dataset_id"}
            or not isinstance(catalog.get("role_assignments"), dict)):
        raise ProfileScopeError("Malformed inference catalog")
    selected = copy.deepcopy(catalog["selected"])
    roles = copy.deepcopy(catalog["role_assignments"])
    if any(value is not None and not _identifier(value) for value in selected.values()):
        raise ProfileScopeError("Malformed selected resource identity")
    resources = {}
    for resource in catalog["resources"]:
        if (not isinstance(resource, dict) or not _identifier(resource.get("id"))
                or resource["id"] in resources):
            raise ProfileScopeError("Malformed or duplicate resource identity")
        resources[resource["id"]] = copy.deepcopy(resource)
    role_names = set()
    for role in catalog["specialist_roles"]:
        if (not isinstance(role, dict) or not _identifier(role.get("id"))
                or role["id"] in role_names):
            raise ProfileScopeError("Malformed specialist role catalog")
        role_names.add(role["id"])
    for profile in catalog["local_profiles"]:
        if (not isinstance(profile, dict) or not _identifier(profile.get("id"))
                or not _identifier(profile.get("backend"))):
            raise ProfileScopeError("Malformed local profile catalog")
    if any(role not in role_names or not _identifier(value) for role, value in roles.items()):
        raise ProfileScopeError("Malformed specialist assignment")
    snapshot = ProfileSnapshot(copy.deepcopy(catalog), selected, roles, resources)
    for identity in ([selected["model_id"]] if selected["model_id"] else []) + [
            value for value in roles.values() if value != LOCAL_ROLE_MODEL]:
        snapshot.resource(identity)
    dataset = selected["dataset_id"]
    if dataset and (dataset not in resources or resources[dataset].get("kind") != "dataset"
                    or resources[dataset].get("state") != "ready"
                    or resources[dataset].get("purpose") != "rag"):
        raise ProfileScopeError("Selected dataset is missing or unavailable")
    return snapshot


class AcceptanceProfileScope:
    """Temporarily choose one adapter, preserving the account's original state."""

    def __init__(self, client, *, adapter: str, model_id: str | None = None,
                 networking: str = "preserve"):
        if adapter not in ADAPTERS or networking not in {"preserve", "local", "follow"}:
            raise ProfileScopeError("Explicit supported adapter and networking scope required")
        if (adapter == "local" and model_id is not None) or (adapter != "local" and not _identifier(model_id)):
            raise ProfileScopeError("Local defaults take no model ID; outbound scope requires one")
        self.client, self.adapter, self.model_id = client, adapter, model_id
        self.networking = networking
        self.original: ProfileSnapshot | None = None
        self.expected: ProfileSnapshot | None = None
        self.fingerprint: dict | None = None
        self.restoration = {"status": "not_started"}
        self._uncertain = False

    def _command(self, action: str, payload: dict) -> None:
        self.assert_unchanged()
        self._uncertain = True
        try:
            response = self.client.client.post("/api/v1/extensions",
                headers=self.client._mutation_headers(), json={"action": action, "payload": payload})
            response.raise_for_status()
        except Exception:
            raise ProfileScopeError("Inference selection write is unconfirmed; reconcile before retry") from None
        before = self.expected
        observed = inspect_profile(self.client)
        wanted_selected, wanted_roles = dict(before.selected), dict(before.roles)
        if action == "select":
            wanted_selected = dict(payload)
        elif payload["model_id"] is None:
            wanted_roles.pop(payload["role"], None)
        else:
            wanted_roles[payload["role"]] = payload["model_id"]
        if observed.selected != wanted_selected or observed.roles != wanted_roles:
            raise ProfileScopeError("Inference selection changed concurrently; reconcile before retry")
        # Relevant connection/catalog edits must not be concealed by the write.
        compare_before = ProfileSnapshot(before.catalog, wanted_selected, wanted_roles, before.resources)
        if observed.fingerprint() != compare_before.fingerprint():
            raise ProfileScopeError("Inference configuration changed during selection")
        self.expected, self._uncertain = observed, False

    def assert_unchanged(self) -> dict:
        if self.expected is None or self._uncertain:
            raise ProfileScopeError("Inference scope is not confirmed")
        observed = inspect_profile(self.client)
        if observed.fingerprint() != self.expected.fingerprint():
            raise ProfileScopeError("Inference configuration drifted; stop and reconcile")
        return observed.fingerprint()

    def __enter__(self):
        self.original = self.expected = inspect_profile(self.client)
        if self.model_id and self.original.resource(self.model_id)["backend"] != self.adapter:
            raise ProfileScopeError("Selected model does not match requested adapter")
        role_names = {row["id"] for row in self.original.catalog["specialist_roles"]}
        target_network = (self.original.effective_networking() if self.networking == "preserve"
                          else LOCAL_ROLE_MODEL if self.networking == "local"
                          else self.model_id or LOCAL_ROLE_MODEL)
        desired_roles = {NETWORKING_ROLE: target_network} if NETWORKING_ROLE in role_names else {}
        self.restoration = {"status": "pending", "atomic_compare_and_set": False}
        try:
            for role in sorted(set(self.original.roles) | set(desired_roles)):
                value = desired_roles.get(role)
                if self.expected.roles.get(role) != value:
                    self._command("assign_role", {"role": role, "model_id": value})
            desired = {**self.original.selected, "model_id": self.model_id}
            if self.expected.selected != desired:
                self._command("select", desired)
            self.fingerprint = self.assert_unchanged()
            if self.fingerprint["adapter"] != self.adapter:
                raise ProfileScopeError("Effective adapter differs from requested scope")
            return self
        except Exception:
            self.restore()
            raise

    def restore(self) -> dict:
        if self.original is None or self.expected is None:
            return self.restoration
        try:
            self.assert_unchanged()
            observed = inspect_profile(self.client)
            original_ids = set(self.original.roles.values()) - {LOCAL_ROLE_MODEL}
            original_ids.update(value for value in self.original.selected.values() if value is not None)
            if any(observed.resources.get(identity) != self.original.resources[identity]
                   for identity in original_ids):
                raise ProfileScopeError("Original inference resources changed; do not restore over edits")
            if self.expected.selected != self.original.selected:
                self._command("select", self.original.selected)
            for role in sorted(set(self.expected.roles) | set(self.original.roles)):
                value = self.original.roles.get(role)
                if self.expected.roles.get(role) != value:
                    self._command("assign_role", {"role": role, "model_id": value})
            self.assert_unchanged()
            if self.expected.fingerprint() != self.original.fingerprint():
                raise ProfileScopeError("Original inference configuration is no longer available")
            self.restoration = {"status": "restored", "atomic_compare_and_set": False}
        except ProfileScopeError:
            self.restoration = {"status": "reconciliation_required", "atomic_compare_and_set": False}
        return self.restoration

    def __exit__(self, exc_type, exc_value, traceback):
        restored = self.restore()
        if restored["status"] != "restored" and exc_type is None:
            raise ProfileScopeError("Inference selection could not be safely restored; reconcile account settings")
        return False
