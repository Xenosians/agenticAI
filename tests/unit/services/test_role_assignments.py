"""Per-specialist model connections: the hub keeps the main selection while each
specialist role can run on its own connection or its own local model."""
import asyncio
from types import SimpleNamespace

import pytest

from services.extensions import service
from services.extensions.context import (
    execution_roles,
    execution_selection,
    networking_inference_stage,
    owner_execution,
    selected_profile,
    specialist_inference_stage,
    specialist_role,
)
from services.extensions.store import LOCAL_ROLE_MODEL, ExtensionStore
from subagents.llm.runtime.stages import InferenceStage


def _connection(store, owner, model, **extra):
    saved = store.put(owner, "model", {"backend": "http", "model": model, "state": "ready", **extra}, "key-" + model)
    return saved["id"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    store = ExtensionStore(tmp_path)
    monkeypatch.setattr(service, "extension_store", lambda: store)
    return store


def test_assignment_is_owner_scoped_and_requires_an_admitted_connection(store):
    claude = _connection(store, "alice", "claude")
    with pytest.raises(ValueError):
        store.assign_role("bob", "jira-specialist", claude)
    with pytest.raises(ValueError):
        store.assign_role("alice", "Jira Specialist", claude)
    pending = store.put("alice", "model", {"backend": "http", "model": "x", "state": "pending"})
    with pytest.raises(ValueError):
        store.assign_role("alice", "jira-specialist", pending["id"])
    assert store.assign_role("alice", "jira-specialist", claude) == {"jira-specialist": claude}
    assert store.role_assignments("bob") == {}
    assert store.assign_role("alice", "jira-specialist", None) == {}


def test_deleting_a_connection_removes_its_role_assignments(store):
    groq = _connection(store, "alice", "groq")
    store.assign_role("alice", "jira-specialist", groq)
    store.assign_role("alice", "networking-specialist", LOCAL_ROLE_MODEL)
    store.delete("alice", groq)
    assert store.role_assignments("alice") == {"networking-specialist": LOCAL_ROLE_MODEL}


def test_hub_uses_the_main_selection_and_each_role_its_own_connection(store):
    claude = _connection(store, "alice", "claude")
    groq = _connection(store, "alice", "groq")
    store.select("alice", claude, None)
    store.assign_role("alice", "jira-specialist", groq)
    store.assign_role("alice", "account-specialist", LOCAL_ROLE_MODEL)

    with owner_execution("alice", "request"):
        assert selected_profile(stage=InferenceStage.SEMANTIC_ROUTING)["model"] == "claude"
        assert selected_profile(stage=InferenceStage.SYNTHESIS)["model"] == "claude"
        with specialist_role("jira-specialist"):
            profile = selected_profile(stage=InferenceStage.SPECIALIST_REASONING)
            assert profile["model"] == "groq" and profile["api_key"] == "key-groq"
            # A specialist's own repair attempt stays on its connection.
            assert selected_profile(stage=InferenceStage.REPAIR,
                                    repair_of=InferenceStage.SPECIALIST_REASONING)["model"] == "groq"
            # Routing inside a specialist context is still the hub's choice.
            assert selected_profile(stage=InferenceStage.SEMANTIC_ROUTING)["model"] == "claude"
        with specialist_role("account-specialist"):
            assert selected_profile(stage=InferenceStage.SPECIALIST_REASONING) is None
        with specialist_role("asset-specialist"):
            assert selected_profile(stage=InferenceStage.SPECIALIST_REASONING)["model"] == "claude"
    assert execution_roles.get() == {}


def test_networking_assignment_is_explicit_consent_for_that_connection_only(store):
    claude = _connection(store, "alice", "claude")  # main selection did not opt in
    groq = _connection(store, "alice", "groq")
    store.select("alice", claude, None)

    with owner_execution("alice", "network"):
        assert networking_inference_stage() is InferenceStage.NETWORKING
        assert selected_profile(stage=InferenceStage.NETWORKING_SELECTED) is None

    store.assign_role("alice", "networking-specialist", groq)
    with owner_execution("alice", "network"):
        assert specialist_inference_stage("networking-specialist") is InferenceStage.NETWORKING_SELECTED
        assert selected_profile(stage=InferenceStage.NETWORKING_SELECTED)["model"] == "groq"

    store.assign_role("alice", "networking-specialist", LOCAL_ROLE_MODEL)
    opted = _connection(store, "alice", "opted", networking_inference="selected_model")
    store.select("alice", opted, None)
    with owner_execution("alice", "network"):
        # The local pin wins over the main selection's networking opt-in.
        assert networking_inference_stage() is InferenceStage.NETWORKING


def test_continuations_pin_role_ids_and_old_plans_pin_none(store):
    claude = _connection(store, "alice", "claude")
    groq = _connection(store, "alice", "groq")
    store.select("alice", claude, None)
    store.assign_role("alice", "jira-specialist", groq)

    with owner_execution("alice", "first"):
        pinned = dict(execution_selection.get())
    assert pinned == {"model_id": claude, "dataset_id": None, "roles": {"jira-specialist": groq}}
    assert "key-groq" not in repr(pinned)

    store.assign_role("alice", "jira-specialist", None)  # owner changes preference later
    with owner_execution("alice", "continue", selection=pinned), specialist_role("jira-specialist"):
        assert selected_profile(stage=InferenceStage.SPECIALIST_REASONING)["model"] == "groq"

    legacy = {"model_id": claude, "dataset_id": None}
    store.assign_role("alice", "jira-specialist", groq)
    with owner_execution("alice", "continue", selection=legacy), specialist_role("jira-specialist"):
        assert selected_profile(stage=InferenceStage.SPECIALIST_REASONING)["model"] == "claude"


def test_concurrent_owners_and_tasks_do_not_cross(store):
    for owner in ("alice", "bob"):
        main = _connection(store, owner, f"{owner}-main")
        store.select(owner, main, None)
    store.assign_role("alice", "jira-specialist", _connection(store, "alice", "alice-groq"))

    async def check(owner, role, expected):
        with owner_execution(owner, "request"), specialist_role(role):
            await asyncio.sleep(0)
            assert selected_profile(stage=InferenceStage.SPECIALIST_REASONING)["model"] == expected

    async def run():
        await asyncio.gather(check("alice", "jira-specialist", "alice-groq"),
                             check("alice", "ticket-specialist", "alice-main"),
                             check("bob", "jira-specialist", "bob-main"))

    asyncio.run(run())


def test_assign_role_action_accepts_only_registered_specialists(store, tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "jira-specialist.md").write_text(
        "---\nname: jira-specialist\ndescription: Jira work\nmodel: jira-func-trained\ntools:\n  - ticket_get\n---\nBody\n",
        encoding="utf-8",
    )
    settings = SimpleNamespace(model_profiles={}, agents_dir=agents)
    groq = _connection(store, "alice", "groq")
    result = asyncio.run(service.command("alice", "assign_role", {"role": "jira-specialist", "model_id": groq}, settings))
    assert result == {"role_assignments": {"jira-specialist": groq}}
    with pytest.raises(ValueError):
        asyncio.run(service.command("alice", "assign_role", {"role": "hub", "model_id": groq}, settings))
    with pytest.raises(ValueError):
        asyncio.run(service.command("alice", "assign_role",
                                    {"role": "jira-specialist", "model_id": groq, "api_key": "x"}, settings))
    catalog = asyncio.run(service.command("alice", "list", {}, settings))
    assert catalog["role_assignments"] == {"jira-specialist": groq}
    assert catalog["specialist_roles"] == [
        {"id": "jira-specialist", "description": "Jira work", "local_model": "jira-func-trained"}
    ]
