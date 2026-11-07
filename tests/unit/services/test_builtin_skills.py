"""Built-in skills: reviewed SKILL.md files every owner starts with, under the owner's control."""
from pathlib import Path

import pytest

from services.extensions import skills as skills_module
from services.extensions.skills import SkillLibrary, parse_skill
from services.extensions.store import ExtensionStore

DEFAULTS = Path(__file__).resolve().parents[3] / "skills" / "defaults"
EXPECTED = {"itsm-gap-finder", "skill-finder", "chat-tightening", "proxmox-operator",
            "tailscale-operator", "research-with-citations", "senior-engineer"}


def library(tmp_path):
    return SkillLibrary(ExtensionStore(tmp_path))


def test_every_built_in_skill_file_passes_the_same_checks_as_an_owner_skill():
    files = sorted(DEFAULTS.glob("*.md"))
    names = {parse_skill(path.read_text(encoding="utf-8")).name for path in files}
    assert names == EXPECTED
    for path in files:
        assert path.stem == parse_skill(path.read_text(encoding="utf-8")).name


def test_an_owner_starts_with_the_built_in_skills_marked_as_such(tmp_path):
    rows = library(tmp_path).list("alice")
    assert {row["name"] for row in rows} == EXPECTED
    assert all(row["builtin"] is True and row["enabled"] is True and row["version"] == 1 for row in rows)
    assert library(tmp_path).list("bob") and len(library(tmp_path).list("alice")) == len(EXPECTED)  # seeded once


def test_editing_a_built_in_skill_keeps_it_built_in_with_the_owners_version(tmp_path):
    skills = library(tmp_path)
    [row] = [r for r in skills.list("alice") if r["name"] == "chat-tightening"]
    edited = skills.save("alice", row["text"].replace("\n---\n", "\n---\nAlways answer in English.\n", 1))
    assert edited["version"] == 2 and edited["builtin"] is True


def test_a_deleted_built_in_skill_is_not_brought_back(tmp_path):
    skills = library(tmp_path)
    skills.list("alice")
    skills.delete("alice", "senior-engineer")
    assert "senior-engineer" not in {row["name"] for row in skills.list("alice")}
    assert "senior-engineer" not in {row["name"] for row in library(tmp_path).list("alice")}


def test_an_owners_own_skill_with_a_built_in_name_is_kept_as_the_owners(tmp_path):
    skills = library(tmp_path)
    mine = "---\nname: senior-engineer\ndescription: Mine.\n---\nMy own text.\n"
    skills.save("bob", mine)                       # saved before bob's built-ins were seeded
    [row] = [r for r in skills.list("bob") if r["name"] == "senior-engineer"]
    assert row["builtin"] is False and row["body"] == "My own text." and row["version"] == 1
    skills.list("alice")
    skills.delete("alice", "skill-finder")         # after deleting a built-in, its name is free
    saved = skills.save("alice", "---\nname: skill-finder\ndescription: Mine.\n---\nMy own text.\n")
    assert saved["builtin"] is False


def test_built_in_skills_do_not_count_against_the_owner_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_module, "MAX_SKILLS_PER_OWNER", 2)
    skills = library(tmp_path)
    assert len(skills.list("alice")) == len(EXPECTED)  # seeded first: more built-ins than the limit
    skills.save("alice", "---\nname: one\ndescription: x\n---\nbody\n")
    skills.save("alice", "---\nname: two\ndescription: x\n---\nbody\n")
    with pytest.raises(ValueError, match="at most 2"):
        skills.save("alice", "---\nname: three\ndescription: x\n---\nbody\n")


def test_built_in_skills_guide_answers_only_when_a_request_names_them(tmp_path):
    skills = library(tmp_path)
    assert skills.applicable("alice", "Show the router health") == []
    names = [skill["name"] for skill in skills.applicable("alice", "find gaps in our ITSM process")]
    assert names == ["itsm-gap-finder"]
