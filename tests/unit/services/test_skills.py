"""Owner skills: SKILL.md guidance that shapes only the answer, never tools or permissions."""
import pytest

from services.extensions import skills as skills_module
from services.extensions.skills import MAX_BODY_CHARACTERS, SkillLibrary, parse_skill
from services.extensions.store import ExtensionStore


def skill_text(name="password-reset", description="How to answer password reset requests.",
               triggers="[password reset, locked out]", always=None, body="Confirm the user's identity first."):
    lines = ["---", f"name: {name}", f"description: {description}"]
    if triggers is not None:
        lines.append(f"triggers: {triggers}")
    if always is not None:
        lines.append(f"always: {always}")
    return "\n".join([*lines, "---", body]) + "\n"


def library(tmp_path):
    return SkillLibrary(ExtensionStore(tmp_path))


def own(rows):
    """The owner's own skills, without the built-in ones every owner starts with."""
    return [row for row in rows if not row["builtin"]]


def test_a_skill_file_is_read_from_its_front_matter_and_body():
    skill = parse_skill(skill_text(body="# Password resets\r\n\r\nConfirm the user's identity first.\r\n"))
    assert skill.name == "password-reset" and skill.description == "How to answer password reset requests."
    assert skill.triggers == ("password reset", "locked out") and skill.always is False
    assert skill.body == "# Password resets\n\nConfirm the user's identity first."
    assert parse_skill(skill_text(triggers="Password Reset, locked out")).triggers == ("password reset", "locked out")


def test_a_skill_written_for_other_agents_is_accepted_without_their_extra_fields():
    text = ("---\nname: incident-update\ndescription: Write incident updates.\nversion: 1.2.0\n"
            "metadata:\n  hermes:\n    tags: [itsm]\n---\nState impact, next step and the next update time.\n")
    skill = parse_skill(text)
    assert skill.name == "incident-update" and skill.triggers == () and skill.always is False


@pytest.mark.parametrize("text, message", [
    ("no front matter", "front matter"),
    ("---\n- just\n- a list\n---\nbody", "front matter"),
    ("---\nname: Bad Name\ndescription: x\n---\nbody", "name"),
    ("---\nname: ok\n---\nbody", "description"),
    ("---\nname: ok\ndescription: x\n---\n", "guidance"),
    ("---\nname: ok\ndescription: x\ntriggers: [1, 2]\n---\nbody", "triggers"),
    ("---\nname: ok\ndescription: x\nalways: maybe\n---\nbody", "always"),
    ("---\nname: ok\ndescription: x\n---\n" + "a" * (MAX_BODY_CHARACTERS + 1), "at most"),
    ("---\nname: ok\ndescription: x\n---\npassword=supersecret1", "credentials"),
    ("---\nname: ok\ndescription: x\n---\nmail person@example.test", "credentials"),
    ("---\nname: ok\ndescription: x\n---\nbell \x07 here", "control"),
    (42, "SKILL.md"),
])
def test_malformed_oversized_or_secret_bearing_skills_are_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_skill(text)


def test_saving_again_makes_a_new_version_and_an_old_one_can_be_restored(tmp_path):
    skills = library(tmp_path)
    first = skills.save("alice", skill_text(body="Version one."))
    assert skills.save("alice", skill_text(body="Version one."))["version"] == 1  # unchanged: no new version
    second = skills.save("alice", skill_text(body="Version two."))
    assert (first["version"], second["version"]) == (1, 2)
    [row] = own(skills.list("alice"))
    assert row["name"] == "password-reset" and row["body"] == "Version two." and row["enabled"] is True
    assert row["id"] == second["id"] and len(row["sha256"]) == 64 and row["triggers"] == ["password reset", "locked out"]
    assert [version["version"] for version in row["versions"]] == [2, 1]
    # The file to edit is the saved skill in canonical form, and saving it unchanged adds no version.
    assert parse_skill(row["text"]) == parse_skill(skill_text(body="Version two."))
    assert skills.save("alice", row["text"])["version"] == 2
    tricky = skill_text(description="'Say \"no\": never # guess'", triggers="['a: b', c]")
    assert parse_skill(tricky).description == 'Say "no": never # guess'
    assert parse_skill(skills.save("alice", tricky)["text"]) == parse_skill(tricky)
    restored = skills.restore("alice", first["id"])
    assert restored["version"] == 4 and restored["body"] == "Version one." and restored["sha256"] == first["sha256"]
    assert own(skills.list("bob")) == []
    with pytest.raises(ValueError, match="not found"):
        skills.restore("bob", first["id"])
    with pytest.raises(ValueError):
        skills.restore("alice", "not-a-version")


def test_history_and_library_sizes_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_module, "MAX_VERSIONS_KEPT", 3)
    monkeypatch.setattr(skills_module, "MAX_SKILLS_PER_OWNER", 2)
    skills = library(tmp_path)
    for index in range(5):
        skills.save("alice", skill_text(body=f"Version {index}."))
    [row] = own(skills.list("alice"))
    assert [version["version"] for version in row["versions"]] == [5, 4, 3]
    skills.save("alice", skill_text(name="second-skill"))
    with pytest.raises(ValueError, match="at most 2"):
        skills.save("alice", skill_text(name="third-skill"))
    skills.save("bob", skill_text(name="third-skill"))  # the bound is per owner


def test_a_disabled_skill_stays_disabled_across_new_versions_and_can_be_deleted(tmp_path):
    skills = library(tmp_path)
    skills.save("alice", skill_text())
    assert skills.set_enabled("alice", "password-reset", False)["enabled"] is False
    assert skills.save("alice", skill_text(body="Edited."))["enabled"] is False
    assert skills.applicable("alice", "I need a password reset") == []
    assert skills.set_enabled("alice", "password-reset", True)["enabled"] is True
    with pytest.raises(ValueError):
        skills.set_enabled("alice", "password-reset", "yes")
    skills.delete("alice", "password-reset")
    assert own(skills.list("alice")) == []
    with pytest.raises(ValueError, match="not found"):
        skills.set_enabled("alice", "password-reset", True)
    with pytest.raises(ValueError, match="not found"):
        skills.delete("alice", "password-reset")


def test_a_skill_applies_when_the_request_mentions_a_trigger_or_its_name(tmp_path):
    skills = library(tmp_path)
    skills.save("alice", skill_text())
    skills.save("alice", skill_text(name="incident-update", description="Incident updates.", triggers=None,
                                    body="State impact first."))
    assert [skill["name"] for skill in skills.applicable("alice", "Dana is LOCKED   OUT of email")] == ["password-reset"]
    assert [skill["name"] for skill in skills.applicable("alice", "Write an incident update for the outage")] == [
        "incident-update"]
    assert skills.applicable("alice", "locked outside the building") == []  # whole words only
    assert skills.applicable("alice", "Show the router health") == []
    assert skills.applicable("bob", "password reset please") == []
    [applied] = skills.applicable("alice", "password-reset please")
    assert set(applied) == {"name", "version", "id", "sha256", "body"}


def test_always_on_skills_apply_to_every_request_within_the_bound(tmp_path, monkeypatch):
    skills = library(tmp_path)
    skills.save("alice", skill_text(name="house-style", description="Answer style.", triggers=None, always="true",
                                    body="Be brief."))
    assert [skill["name"] for skill in skills.applicable("alice", "Show the router health")] == ["house-style"]
    skills.save("alice", skill_text())
    skills.save("alice", skill_text(name="account-lockout", description="Lockouts.", triggers="[locked out]",
                                    body="Check the directory account first."))
    # Skills the request names come first; one answer carries at most two.
    assert [skill["name"] for skill in skills.applicable("alice", "password reset: the user is locked out")] == [
        "account-lockout", "password-reset"]
    monkeypatch.setattr(skills_module, "MAX_APPLIED_CHARACTERS", 12)
    assert [skill["name"] for skill in skills.applicable("alice", "password reset: the user is locked out")] == [
        "house-style"]


def test_a_request_carries_the_owners_matching_skills(tmp_path, monkeypatch):
    from services.extensions import service
    from services.extensions.context import (
        answer_skills, owner_execution, private_request, request_skills, skill_guidance,
    )
    from subagents.llm.runtime.stages import InferenceStage
    monkeypatch.setenv("EXTENSION_STORAGE_ROOT", str(tmp_path))
    service.extension_store.cache_clear()
    try:
        SkillLibrary(service.extension_store()).save("alice", skill_text())
        with owner_execution("alice", "password reset for Dana"):
            assert [skill["name"] for skill in request_skills.get()] == ["password-reset"]
            assert answer_skills() == [] and private_request(None, False) is False  # chosen, not yet used
            assert "identity first" in skill_guidance(InferenceStage.SYNTHESIS)
            assert [skill["name"] for skill in answer_skills()] == ["password-reset"]
            # A job guided by the owner's skills is private, like their connection or dataset.
            assert private_request(None, False) is True
        with owner_execution("bob", "password reset for Dana"):
            assert request_skills.get() == [] and private_request(None, False) is False
            assert private_request({"backend": "http"}, False) and private_request(None, True)
        assert request_skills.get() == []
    finally:
        service.extension_store.cache_clear()
