"""Inference acceptance must bind actual selection and preserve concurrent edits."""
import copy
import json
import httpx
import pytest

from scripts.acceptance_profiles import AcceptanceProfileScope, ProfileScopeError, inspect_profile


def catalog():
    return {"schema": "extension-catalog.v1", "selected": {"model_id": "first", "dataset_id": None},
            "role_assignments": {"account-specialist": "local", "jira-specialist": "second",
                                 "networking-specialist": "second"},
            "resources": [{"id": key, "kind": "model", "state": "ready", "backend": "http",
                           "model": "example-model", "url": "https://private.invalid/v1", "name": key}
                          for key in ("first", "second")],
            "specialist_roles": [{"id": role, "local_model": "test-local"} for role in
                                  ("account-specialist", "jira-specialist", "networking-specialist")],
            "local_profiles": [{"id": "test-local", "backend": "transformers"}]}


class ProductAPI:
    def __init__(self, state=None):
        self.state = copy.deepcopy(catalog() if state is None else state)
        self.writes = []
        self.lost_ack = False
        self.bad_get = False
        self.client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(self.handle))

    def _mutation_headers(self):
        return {"x-csrf-token": "synthetic"}

    def handle(self, request):
        assert request.url.path == "/api/v1/extensions"
        if request.method == "GET":
            return httpx.Response(200, json={} if self.bad_get else self.state)
        assert request.headers["x-csrf-token"] == "synthetic"
        payload = json.loads(request.content)
        self.writes.append(payload)
        data = payload["payload"]
        if payload["action"] == "select":
            self.state["selected"] = data
        else:
            assert payload["action"] == "assign_role"
            if data["model_id"] is None:
                self.state["role_assignments"].pop(data["role"], None)
            else:
                self.state["role_assignments"][data["role"]] = data["model_id"]
        return httpx.Response(503 if self.lost_ack else 200, json={"private": "do-not-report"})


@pytest.mark.parametrize("adapter,model,network", [("bad", None, "preserve"),
    ("http", None, "preserve"), ("local", "first", "local"), ("local", None, "bad")])
def test_invalid_scope_is_rejected_without_http_or_configuration_access(adapter, model, network):
    with pytest.raises(ProfileScopeError):
        AcceptanceProfileScope(object(), adapter=adapter, model_id=model, networking=network)


def test_local_scope_removes_mixed_overrides_pins_networking_and_restores_original():
    api = ProductAPI()
    original = copy.deepcopy(api.state)
    with AcceptanceProfileScope(api, adapter="local", networking="local") as scope:
        assert api.state["selected"]["model_id"] is None
        assert api.state["role_assignments"] == {"networking-specialist": "local"}
        assert scope.fingerprint["adapter"] == "local"
        assert scope.fingerprint["networking"]["adapter"] == "local"
        assert scope.assert_unchanged() == scope.fingerprint
    assert api.state == original
    assert scope.restoration["status"] == "restored"


def test_outbound_scope_requires_matching_ready_owned_resource():
    api = ProductAPI()
    for adapter, identity in [("pi", "first"), ("http", "missing")]:
        with pytest.raises(ProfileScopeError):
            with AcceptanceProfileScope(api, adapter=adapter, model_id=identity):
                pytest.fail("invalid model admitted")
    assert api.writes == []
    api.state["resources"][0]["state"] = "unavailable"
    with pytest.raises(ProfileScopeError):
        inspect_profile(api)


def test_outbound_scope_clears_nonnetwork_overrides_and_preserves_network_connection():
    api = ProductAPI()
    with AcceptanceProfileScope(api, adapter="http", model_id="first") as scope:
        assert api.state["role_assignments"] == {"networking-specialist": "second"}
        fingerprint = scope.fingerprint
        assert fingerprint["adapter"] == fingerprint["networking"]["adapter"] == "http"
        assert fingerprint["networking"]["follows_main"] is False
        encoded = json.dumps(fingerprint)
        for private in ("private.invalid", "https", "example-model", "first", "second"):
            assert private not in encoded


@pytest.mark.parametrize("mode,expected", [("preserve", "first"), ("local", "local"), ("follow", "second")])
def test_implicit_selected_networking_is_pinned_before_switching_hub(mode, expected):
    api = ProductAPI()
    api.state["role_assignments"].pop("networking-specialist")
    api.state["resources"][0]["networking_inference"] = "selected_model"
    with AcceptanceProfileScope(api, adapter="http", model_id="second", networking=mode):
        assert api.state["role_assignments"] == {"networking-specialist": expected}
    assert "networking-specialist" not in api.state["role_assignments"]


def test_dataset_is_preserved_and_its_metadata_is_bound():
    api = ProductAPI()
    api.state["resources"].append({"id": "dataset", "kind": "dataset", "purpose": "rag", "state": "ready",
                                   "source": "private reference", "version": "one"})
    api.state["selected"]["dataset_id"] = "dataset"
    scope = AcceptanceProfileScope(api, adapter="local")
    with pytest.raises(ProfileScopeError, match="drifted"):
        with scope:
            assert api.state["selected"]["dataset_id"] == "dataset"
            api.state["resources"][-1]["version"] = "two"
            scope.assert_unchanged()
    assert scope.restoration["status"] == "reconciliation_required"


def test_concurrent_selection_change_is_never_restored_over():
    api = ProductAPI()
    scope = AcceptanceProfileScope(api, adapter="local")
    with pytest.raises(ProfileScopeError, match="safely restored"):
        with scope:
            api.state["selected"]["model_id"] = "second"
            writes_before_exit = len(api.writes)
    assert len(api.writes) == writes_before_exit
    assert api.state["selected"]["model_id"] == "second"
    assert scope.restoration["status"] == "reconciliation_required"


def test_connection_options_and_local_profile_changes_invalidate_scope():
    for change in (lambda state: state["resources"][0].update(reasoning_effort="high"),
                   lambda state: state["local_profiles"][0].update(backend="changed")):
        api = ProductAPI()
        scope = AcceptanceProfileScope(api, adapter="http", model_id="first")
        with pytest.raises(ProfileScopeError, match="drifted"):
            with scope:
                change(api.state)
                scope.assert_unchanged()
        assert scope.restoration["status"] == "reconciliation_required"


def test_original_connection_change_is_detected_before_restoring_selection():
    api = ProductAPI()
    scope = AcceptanceProfileScope(api, adapter="local", networking="local")
    with pytest.raises(ProfileScopeError, match="safely restored"):
        with scope:
            api.state["resources"][0]["url"] = "https://modified.invalid"
            writes_before_exit = len(api.writes)
    assert len(api.writes) == writes_before_exit
    assert api.state["selected"]["model_id"] is None
    assert scope.restoration["status"] == "reconciliation_required"


def test_lost_write_acknowledgement_disables_automatic_restoration_or_retry():
    api = ProductAPI()
    api.lost_ack = True
    scope = AcceptanceProfileScope(api, adapter="local")
    with pytest.raises(ProfileScopeError, match="unconfirmed") as error:
        with scope:
            pytest.fail("uncertain selection entered")
    assert len(api.writes) == 1
    assert "do-not-report" not in str(error.value)
    assert scope.restoration["status"] == "reconciliation_required"


def test_body_exception_is_preserved_after_successful_restoration():
    api = ProductAPI()
    original = copy.deepcopy(api.state)
    scope = AcceptanceProfileScope(api, adapter="local")
    with pytest.raises(ValueError, match="actual failure"):
        with scope:
            raise ValueError("actual failure")
    assert api.state == original and scope.restoration["status"] == "restored"


@pytest.mark.parametrize("change", [lambda state: state.pop("selected"),
    lambda state: state["resources"].append(copy.deepcopy(state["resources"][0])),
    lambda state: state["role_assignments"].update({"unknown-specialist": "local"}),
    lambda state: state["selected"].update(dataset_id="missing")])
def test_malformed_catalogs_fail_closed(change):
    api = ProductAPI()
    change(api.state)
    with pytest.raises(ProfileScopeError):
        inspect_profile(api)
    assert api.writes == []
