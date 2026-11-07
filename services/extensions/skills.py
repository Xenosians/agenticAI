"""Owner skills: reusable written guidance (SKILL.md) for the answers Janus writes.

A skill is a Markdown file with YAML front matter, the format other agents use:

    ---
    name: password-reset
    description: How to answer password reset requests.
    triggers: [password reset, locked out]   # optional words that select it
    always: false                             # optional: apply to every request
    ---
    Confirm the user's identity before anything else...

Skills are procedural guidance only. They reach only the steps that write the
answer the user reads, never routing, tool choice, arguments, approvals or
permissions, so a skill can change how an answer is written but not what Janus
does. Each owner's skills are private. Saving a changed skill adds a version
(the newest is in use; an older one can be restored), and a skill can be turned
off without deleting it. Skill text passes the same credential and personal
identifier scan as datasets, and the skills an answer used are reported with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import uuid

import yaml

from services.extensions.store import ExtensionStore, sensitive_content

MAX_SKILL_FILE_CHARACTERS = 20_000
MAX_BODY_CHARACTERS = 4_000
MAX_DESCRIPTION_CHARACTERS = 300
MAX_TRIGGERS = 10
MAX_TRIGGER_CHARACTERS = 60
MAX_SKILLS_PER_OWNER = 50
MAX_VERSIONS_KEPT = 10
# What one answer may carry: at most two skills and this much of their text, so
# guidance never crowds out the request and the evidence the answer is based on.
MAX_APPLIED_SKILLS = 2
MAX_APPLIED_CHARACTERS = 6_000
NAME_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", re.S)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    triggers: tuple[str, ...]
    always: bool
    body: str


class SkillError(ValueError):
    """A skill was refused: a stable skill_* code and a fixed message, never the skill's own text."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def public_detail(self) -> dict:
        return {"code": self.code, "message": self.message}


def _words(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def parse_skill(text) -> Skill:
    """Read and check one SKILL.md file; anything malformed is refused with the reason."""
    if not isinstance(text, str) or not text.strip():
        raise SkillError("skill_format", "Upload a SKILL.md file: YAML front matter, then the guidance.")
    if len(text) > MAX_SKILL_FILE_CHARACTERS:
        raise SkillError("skill_too_large", f"A skill file may hold at most {MAX_SKILL_FILE_CHARACTERS} characters.")
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    if _CONTROL.search(text):
        raise SkillError("skill_format", "A skill cannot contain control characters.")
    match = _FRONT_MATTER.match(text)
    try:
        metadata = yaml.safe_load(match.group(1)) if match else None
    except yaml.YAMLError:
        metadata = None
    if not isinstance(metadata, dict):
        raise SkillError("skill_format", "A skill starts with YAML front matter between --- lines.")
    name = metadata.get("name")
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise SkillError("skill_name", "The skill name must be lowercase letters, digits and dashes, e.g. password-reset.")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > MAX_DESCRIPTION_CHARACTERS:
        raise SkillError("skill_description",
                         f"The skill needs a description of at most {MAX_DESCRIPTION_CHARACTERS} characters.")
    triggers = metadata.get("triggers") or []
    if isinstance(triggers, str):
        triggers = triggers.split(",")
    if (not isinstance(triggers, list) or len(triggers) > MAX_TRIGGERS
            or not all(isinstance(item, str) and 0 < len(item.strip()) <= MAX_TRIGGER_CHARACTERS
                       and _words(item) for item in triggers)):
        raise SkillError("skill_triggers", f"The skill triggers must be up to {MAX_TRIGGERS} short words or phrases.")
    always = metadata.get("always", False)
    if not isinstance(always, bool):
        raise SkillError("skill_always", "The skill setting always must be true or false.")
    body = match.group(2).strip()
    if not body:
        raise SkillError("skill_format", "The skill has no guidance after its front matter.")
    if len(body) > MAX_BODY_CHARACTERS:
        raise SkillError("skill_too_large", f"The skill guidance may hold at most {MAX_BODY_CHARACTERS} characters.")
    if sensitive_content("\n".join([description, body, *triggers])):
        raise SkillError("skill_sensitive", "A skill cannot contain credentials or personal identifiers.")
    return Skill(name=name, description=description.strip(),
                 triggers=tuple(dict.fromkeys(_words(item) for item in triggers)), always=always, body=body)


def skill_file(data: dict) -> str:
    """A saved skill as the SKILL.md text to edit: front matter, then the guidance. Parses back to the same skill."""
    metadata = {"name": data["name"], "description": data["description"]}
    if data["triggers"]:
        metadata["triggers"] = list(data["triggers"])
    if data["always"]:
        metadata["always"] = True
    front = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True, width=1000).strip()
    return f"---\n{front}\n---\n{data['body']}\n"


BUILTIN_DIRECTORY = Path(__file__).resolve().parents[2] / "skills" / "defaults"


@lru_cache(maxsize=1)
def builtin_skills() -> tuple[Skill, ...]:
    """The reviewed built-in skills, parsed and checked like any owner skill (a bad file fails loudly)."""
    return tuple(parse_skill(path.read_text(encoding="utf-8")) for path in sorted(BUILTIN_DIRECTORY.glob("*.md")))


class SkillLibrary:
    """Each owner's skills, versioned, in the extension store's database.

    Every owner starts with the built-in skills shipped in skills/defaults (reviewed
    SKILL.md files, checked by the same parser). They are seeded once, the first
    time the owner's library is read; after that they belong to the owner like any
    other skill: an edit becomes the owner's next version, turning one off keeps it,
    and a deleted built-in skill is not added again. Built-in skills do not count
    against the owner's skill limit.
    """

    def __init__(self, store: ExtensionStore):
        self.store = store
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS skill (
                    owner TEXT NOT NULL, name TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(owner, name));
                CREATE TABLE IF NOT EXISTS skill_version (
                    owner TEXT NOT NULL, id TEXT NOT NULL, name TEXT NOT NULL, version INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(owner, id), UNIQUE(owner, name, version));
                -- Built-in skills each owner has received: present while the owner keeps
                -- the skill, and still present after a delete, so it is not seeded again.
                CREATE TABLE IF NOT EXISTS skill_builtin (
                    owner TEXT NOT NULL, name TEXT NOT NULL, PRIMARY KEY(owner, name));
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(skill)")}
            if "builtin" not in columns:
                db.execute("ALTER TABLE skill ADD COLUMN builtin INTEGER NOT NULL DEFAULT 0")

    def _seed_builtins(self, db, owner: str) -> None:
        """Give the owner each built-in skill not received yet. Runs inside a write transaction."""
        received = {row["name"] for row in db.execute("SELECT name FROM skill_builtin WHERE owner=?", (owner,))}
        for skill in builtin_skills():
            if skill.name in received:
                continue
            exists = db.execute("SELECT 1 FROM skill WHERE owner=? AND name=?", (owner, skill.name)).fetchone()
            db.execute("INSERT INTO skill_builtin(owner, name) VALUES (?,?)", (owner, skill.name))
            if exists is None:  # an owner skill with this name is kept as the owner's own
                db.execute("INSERT INTO skill(owner, name, enabled, builtin) VALUES (?,?,1,1)", (owner, skill.name))
                self._add_version(db, owner, skill)

    def _ensure_builtins(self, owner: str) -> None:
        """Seed only when something is missing, so the per-request read path takes no write lock."""
        wanted = {skill.name for skill in builtin_skills()}
        with self.store.connect() as db:
            received = {row["name"] for row in db.execute("SELECT name FROM skill_builtin WHERE owner=?", (owner,))}
            if wanted <= received:
                return
            db.execute("BEGIN IMMEDIATE")  # re-read under the lock: a concurrent request may have seeded
            self._seed_builtins(db, owner)

    def _add_version(self, db, owner: str, skill: Skill) -> dict:
        data = {"name": skill.name, "description": skill.description, "triggers": list(skill.triggers),
                "always": skill.always, "body": skill.body}
        digest = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        latest = db.execute("SELECT id, version, sha256 FROM skill_version WHERE owner=? AND name=? "
                            "ORDER BY version DESC LIMIT 1", (owner, skill.name)).fetchone()
        if latest is not None and latest["sha256"] == digest:
            return self._row(db, owner, skill.name)
        if latest is None:
            count = db.execute("SELECT COUNT(*) FROM skill WHERE owner=? AND builtin=0", (owner,)).fetchone()[0]
            exists = db.execute("SELECT 1 FROM skill WHERE owner=? AND name=?", (owner, skill.name)).fetchone()
            if exists is None:
                if count >= MAX_SKILLS_PER_OWNER:
                    raise SkillError("skill_limit", f"Keep at most {MAX_SKILLS_PER_OWNER} skills; delete one first.")
                db.execute("INSERT INTO skill(owner, name, enabled) VALUES (?,?,1)", (owner, skill.name))
        version = 1 if latest is None else latest["version"] + 1
        db.execute("INSERT INTO skill_version VALUES (?,?,?,?,?,?)",
                   (owner, uuid.uuid4().hex, skill.name, version, digest, json.dumps(data)))
        db.execute("DELETE FROM skill_version WHERE owner=? AND name=? AND version <= ?",
                   (owner, skill.name, version - MAX_VERSIONS_KEPT))
        return self._row(db, owner, skill.name)

    def save(self, owner: str, text) -> dict:
        skill = parse_skill(text)
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            return self._add_version(db, owner, skill)

    def restore(self, owner: str, version_id) -> dict:
        """Make an older version current again, as a new version with the same content."""
        if not isinstance(version_id, str) or not re.fullmatch(r"[0-9a-f]{32}", version_id):
            raise SkillError("skill_version", "Name a skill version to restore.")
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT data FROM skill_version WHERE owner=? AND id=?", (owner, version_id)).fetchone()
            if row is None:
                raise SkillError("skill_not_found", "Skill version not found.")
            data = json.loads(row["data"])
            return self._add_version(db, owner, Skill(name=data["name"], description=data["description"],
                                                      triggers=tuple(data["triggers"]), always=data["always"],
                                                      body=data["body"]))

    def set_enabled(self, owner: str, name, enabled) -> dict:
        if not isinstance(enabled, bool):
            raise SkillError("skill_enabled", "enabled must be true or false.")
        self._ensure_builtins(owner)
        with self.store.connect() as db:
            if db.execute("UPDATE skill SET enabled=? WHERE owner=? AND name=?",
                          (int(enabled), owner, name)).rowcount == 0:
                raise SkillError("skill_not_found", "Skill not found.")
            return self._row(db, owner, name)

    def delete(self, owner: str, name) -> None:
        self._ensure_builtins(owner)  # a built-in deleted before its first read stays deleted
        with self.store.connect() as db:
            if db.execute("DELETE FROM skill WHERE owner=? AND name=?", (owner, name)).rowcount == 0:
                raise SkillError("skill_not_found", "Skill not found.")
            db.execute("DELETE FROM skill_version WHERE owner=? AND name=?", (owner, name))

    def _row(self, db, owner: str, name: str) -> dict:
        skill = db.execute("SELECT enabled, builtin FROM skill WHERE owner=? AND name=?", (owner, name)).fetchone()
        versions = db.execute("SELECT id, version, sha256, data FROM skill_version WHERE owner=? AND name=? "
                              "ORDER BY version DESC", (owner, name)).fetchall()
        current = json.loads(versions[0]["data"])
        return {**current, "text": skill_file(current), "id": versions[0]["id"], "version": versions[0]["version"],
                "sha256": versions[0]["sha256"], "enabled": bool(skill["enabled"]), "builtin": bool(skill["builtin"]),
                "versions": [{"id": row["id"], "version": row["version"], "sha256": row["sha256"]}
                             for row in versions]}

    def list(self, owner: str) -> list[dict]:
        self._ensure_builtins(owner)
        with self.store.connect() as db:
            names = [row["name"] for row in db.execute(
                "SELECT name FROM skill WHERE owner=? ORDER BY name", (owner,)).fetchall()]
            return [self._row(db, owner, name) for name in names]

    def applicable(self, owner: str, request: str) -> list[dict]:
        """The enabled skills for one request: those it names first, then always-on ones, within the bounds."""
        words = f" {_words(request or '')} "
        named, always = [], []
        for row in self.list(owner):
            if not row["enabled"]:
                continue
            keys = [*row["triggers"], _words(row["name"])]
            if any(f" {key} " in words for key in keys):
                named.append(row)
            elif row["always"]:
                always.append(row)
        chosen, used = [], 0
        for row in [*named, *always]:
            if len(chosen) == MAX_APPLIED_SKILLS:
                break
            if used + len(row["body"]) > MAX_APPLIED_CHARACTERS:
                continue
            used += len(row["body"])
            chosen.append({key: row[key] for key in ("name", "version", "id", "sha256", "body")})
        return chosen
