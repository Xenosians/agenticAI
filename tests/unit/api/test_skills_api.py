"""Skill actions through the internal extensions API: owner-scoped, validated, never a credential path."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import extensions
from config.settings import Settings
from services.extensions.service import extension_store
from services.extensions.skills import builtin_skills

SKILL = ("---\nname: password-reset\ndescription: How to answer password reset requests.\n"
         "triggers: [password reset]\n---\nConfirm the user's identity first.\n")


def client_for(tmp_path, monkeypatch):
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    settings = Settings(_env_file=None, itsm_internal_job_token="test-internal-secret")
    monkeypatch.setattr(extensions, "get_settings", lambda: settings)
    extension_store.cache_clear()
    app = FastAPI()
    app.include_router(extensions.router)
    client = TestClient(app)
    headers = {"x-internal-token": "test-internal-secret"}

    def call(owner, action, payload):
        return client.post("/internal/v1/extensions", json={"owner": owner, "action": action, "payload": payload},
                           headers=headers)
    return call


def own(rows):
    """The owner's own skills, without the built-in ones every owner starts with."""
    return [row for row in rows if not row["builtin"]]


def test_skills_are_saved_versioned_listed_toggled_restored_and_deleted_per_owner(tmp_path, monkeypatch):
    call = client_for(tmp_path, monkeypatch)
    try:
        first = call("alice", "skill_save", {"text": SKILL}).json()
        assert first["name"] == "password-reset" and first["version"] == 1 and first["enabled"] is True
        assert first["builtin"] is False
        second = call("alice", "skill_save", {"text": SKILL.replace("first.", "first, then reset.")}).json()
        assert second["version"] == 2
        listed = own(call("alice", "skill_list", {}).json()["skills"])
        assert [row["name"] for row in listed] == ["password-reset"] and listed[0]["version"] == 2
        # Another owner starts with only the built-in skills, marked as such; never alice's.
        bob = call("bob", "skill_list", {}).json()["skills"]
        assert {row["name"] for row in bob} == {skill.name for skill in builtin_skills()}
        assert all(row["builtin"] is True for row in bob)
        assert call("alice", "skill_enable", {"name": "password-reset", "enabled": False}).json()["enabled"] is False
        restored = call("alice", "skill_restore", {"id": first["id"]}).json()
        assert restored["version"] == 3 and restored["body"] == "Confirm the user's identity first."
        assert call("bob", "skill_restore", {"id": first["id"]}).status_code == 422
        assert call("bob", "skill_delete", {"name": "password-reset"}).status_code == 422
        assert call("alice", "skill_delete", {"name": "password-reset"}).json() == {"deleted": "password-reset"}
        assert own(call("alice", "skill_list", {}).json()["skills"]) == []
        # The catalog lists the owner's skills too, so the browser needs one read.
        call("alice", "skill_save", {"text": SKILL})
        assert [row["name"] for row in own(call("alice", "list", {}).json()["skills"])] == ["password-reset"]
    finally:
        extension_store.cache_clear()


def test_malformed_skill_requests_are_refused_with_the_reason(tmp_path, monkeypatch):
    call = client_for(tmp_path, monkeypatch)
    try:
        for payload, action in (({"text": 42}, "skill_save"), ({"text": "no front matter"}, "skill_save"),
                                ({"name": "password-reset", "enabled": "no"}, "skill_enable"),
                                ({"name": 7}, "skill_delete"), ({"id": None}, "skill_restore")):
            response = call("alice", action, payload)
            assert response.status_code == 422, (action, payload)
        detail = call("alice", "skill_save", {"text": SKILL.replace("first.", "password=supersecret1")}).json()
        assert "credentials" in str(detail)
    finally:
        extension_store.cache_clear()
