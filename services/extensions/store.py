from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from cryptography.fernet import Fernet

# A role bound to this value runs on the specialist's own local model.
LOCAL_ROLE_MODEL = "local"
ROLE_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,63}")
NEWER_VERSION_EXISTS = "This dataset already has a newer version; replace the latest version"
# Admission is conservative: these spans make a dataset possibly identifying or secret-bearing.
_SENSITIVE_PATTERNS = tuple(re.compile(pattern) for pattern in (
    r"-----BEGIN .*PRIVATE KEY-----", r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*[^\s]{6,}",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", r"\bsk-[A-Za-z0-9_-]{16,}\b"))


def sensitive_content(text: str) -> str | None:
    """Why text may not be admitted (a credential or personal identifier), or None.

    The one admission check for dataset text: the store applies it to every upload,
    and the crawler applies it page by page so one page cannot block a whole site.
    Detection is not a guarantee of de-identification; manual review remains necessary.
    """
    from learning.evidence.sanitizer import sanitize_text
    if sanitize_text(text) != text:
        return "Dataset failed the shared training privacy/secret scanner"
    if any(pattern.search(text) for pattern in _SENSITIVE_PATTERNS):
        return "Dataset contains possible secrets or personal identifiers; sanitize before admission"
    return None


def _input_key(messages: list) -> str:
    # The same input may never leak across splits, even with different targets.
    return hashlib.sha256(json.dumps(messages[:-1], sort_keys=True).encode()).hexdigest()


def _split_history(previous: dict | None) -> dict[str, str]:
    """Every training input the earlier versions assigned, and the split it went to."""
    if previous is None:
        return {}
    history = dict(previous.get("retired_splits", {}))
    for split, rows in previous["content"].items():
        for row in rows:
            history[_input_key(row["messages"])] = split
    return history


def _training_splits(unique: dict[str, dict], history: dict[str, str]) -> tuple[dict, dict]:
    """Fixed splits: an input keeps the split any earlier version gave it; new inputs
    fill evaluation up to a fifth, in digest order."""
    ordered = sorted(unique)
    validation = {key for key in ordered if history.get(key) == "validation"}
    fresh = [key for key in ordered if key not in history]
    validation.update(fresh[:max(0, max(1, len(ordered) // 5) - len(validation))])
    if not validation:
        raise ValueError("A new version needs at least one evaluation example that was never used for training")
    content = {"train": [unique[key] for key in ordered if key not in validation],
               "validation": [unique[key] for key in ordered if key in validation]}
    # Inputs this version dropped keep their split for any later version that restores them.
    retired = {key: split for key, split in history.items() if key not in unique}
    return content, retired


class ExtensionStore:
    """Owner-scoped durable configuration; credentials never appear in list results."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_path = self.root / "credentials.key"
        try:
            fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as stream:
                stream.write(Fernet.generate_key())
        self.cipher = Fernet(key_path.read_bytes())
        self.path = self.root / "extensions.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS resource (
                    owner TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL,
                    data TEXT NOT NULL, secret BLOB, PRIMARY KEY(owner,id));
                CREATE TABLE IF NOT EXISTS preference (
                    owner TEXT PRIMARY KEY, model_id TEXT, dataset_id TEXT);
                CREATE TABLE IF NOT EXISTS role_assignment (
                    owner TEXT NOT NULL, role TEXT NOT NULL, model_id TEXT NOT NULL,
                    PRIMARY KEY(owner, role));
                -- A dataset version has at most one newer version, so history stays linear.
                CREATE UNIQUE INDEX IF NOT EXISTS dataset_successor
                    ON resource(owner, json_extract(data, '$.previous_id'))
                    WHERE kind = 'dataset' AND json_extract(data, '$.previous_id') IS NOT NULL;
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def put(self, owner: str, kind: str, data: dict, secret: str = "", *, identity: str | None = None) -> dict:
        if not owner or len(owner) > 200:
            raise ValueError("Invalid owner")
        identity = identity or uuid.uuid4().hex
        stored = {**data, "id": identity, "kind": kind}
        encrypted = self.cipher.encrypt(secret.encode()) if secret else None
        with self.connect() as db:
            db.execute("INSERT INTO resource VALUES (?,?,?,?,?)",
                       (owner, identity, kind, json.dumps(stored), encrypted))
        return {**stored, "credential_configured": bool(secret)}

    def get(self, owner: str, identity: str, *, secret=False) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM resource WHERE owner=? AND id=?", (owner, identity)).fetchone()
        if row is None:
            raise ValueError("Resource not found")
        data = json.loads(row["data"])
        data["credential_configured"] = row["secret"] is not None
        if data.get("kind") == "dataset":
            # A dataset saved before versioning is the first version of its own history.
            data.setdefault("version", 1)
            data.setdefault("lineage_id", data["id"])
        if secret:
            data["api_key"] = self.cipher.decrypt(row["secret"]).decode() if row["secret"] else ""
        return data

    def list(self, owner: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT id FROM resource WHERE owner=? ORDER BY rowid", (owner,)).fetchall()
        resources = [self.get(owner, row["id"]) for row in rows]
        # A replaced dataset version names the next version of its history that still exists.
        histories: dict[str, list[dict]] = {}
        for row in resources:
            if row.get("kind") == "dataset":
                histories.setdefault(row["lineage_id"], []).append(row)
        for history in histories.values():
            history.sort(key=lambda row: row["version"])
            for older, newer in zip(history, history[1:]):
                older["superseded_by"] = newer["id"]
        return resources

    def update_model_options(self, owner: str, identity: str, changes: dict) -> dict:
        """Patch only inference options without replacing identity or credentials."""
        from services.extensions.options import MODEL_OPTION_FIELDS, validate_model_options
        if set(changes) - MODEL_OPTION_FIELDS:
            raise ValueError("Unsupported model inference option")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT kind, data FROM resource WHERE owner=? AND id=?", (owner, identity)).fetchone()
            if row is None or row["kind"] != "model":
                raise ValueError("Model connection not found")
            data = json.loads(row["data"])
            for name, value in changes.items():
                if value is None or (name in {"reasoning_effort", "thinking_mode"} and value == ""):
                    data.pop(name, None)
                else:
                    data[name] = value
            validate_model_options(data)
            db.execute("UPDATE resource SET data=? WHERE owner=? AND id=?", (json.dumps(data), owner, identity))
        return self.get(owner, identity)

    def select(self, owner: str, model_id: str | None, dataset_id: str | None):
        for identity, kind in [(model_id, "model"), (dataset_id, "dataset")]:
            if identity:
                resource = self.get(owner, identity)
                if resource["kind"] != kind or resource.get("state") != "ready":
                    raise ValueError("Resource has not passed admission")
                if kind == "dataset" and resource.get("purpose") != "rag":
                    raise ValueError("Training datasets cannot be used as retrieval sources")
        with self.connect() as db:
            db.execute("INSERT INTO preference VALUES (?,?,?) ON CONFLICT(owner) DO UPDATE SET model_id=excluded.model_id,dataset_id=excluded.dataset_id",
                       (owner, model_id, dataset_id))

    def selected(self, owner: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM preference WHERE owner=?", (owner,)).fetchone()
        return {"model_id": row["model_id"], "dataset_id": row["dataset_id"]} if row else {"model_id": None, "dataset_id": None}

    def assign_role(self, owner: str, role: str, model_id: str | None) -> dict:
        """Bind one specialist role to a saved model connection.

        ``None`` removes the binding (the role follows the hub selection);
        ``LOCAL_ROLE_MODEL`` pins the specialist's own local model.
        """
        if not ROLE_PATTERN.fullmatch(role or ""):
            raise ValueError("Invalid role")
        if model_id and model_id != LOCAL_ROLE_MODEL:
            resource = self.get(owner, model_id)
            if resource["kind"] != "model" or resource.get("state") != "ready":
                raise ValueError("Resource has not passed admission")
        with self.connect() as db:
            if model_id:
                db.execute("INSERT INTO role_assignment VALUES (?,?,?) ON CONFLICT(owner, role) "
                           "DO UPDATE SET model_id=excluded.model_id", (owner, role, model_id))
            else:
                db.execute("DELETE FROM role_assignment WHERE owner=? AND role=?", (owner, role))
        return self.role_assignments(owner)

    def role_assignments(self, owner: str) -> dict[str, str]:
        with self.connect() as db:
            rows = db.execute("SELECT role, model_id FROM role_assignment WHERE owner=? ORDER BY role",
                              (owner,)).fetchall()
        return {row["role"]: row["model_id"] for row in rows}

    def delete(self, owner: str, identity: str):
        self.get(owner, identity)
        with self.connect() as db:
            db.execute("DELETE FROM role_assignment WHERE owner=? AND model_id=?", (owner, identity))
            db.execute("UPDATE preference SET model_id=NULL WHERE owner=? AND model_id=?", (owner, identity))
            db.execute("UPDATE preference SET dataset_id=NULL WHERE owner=? AND dataset_id=?", (owner, identity))
            db.execute("DELETE FROM resource WHERE owner=? AND id=?", (owner, identity))

    def dataset(self, owner: str, *, name: str, text: str, source: str, license: str,
                purpose: str, reviewed: bool, replaces: str | None = None,
                provenance: dict | None = None, crawl: dict | None = None) -> dict:
        """Admit a dataset, or a new version of the latest version named by ``replaces``.

        ``provenance`` records how server-side code derived ``text`` (for example a
        Lua transform's script and input digests), and ``crawl`` which pages a site
        crawl read and their digests; neither ever bypasses admission.
        """
        if not name.strip() or not source.strip() or not license.strip():
            raise ValueError("Name, source and permitted-use/license metadata are required")
        if purpose not in {"rag", "training"} or reviewed is not True:
            raise ValueError("Choose a purpose and confirm review/permission to use this dataset")
        if not 0 < len(text.encode()) <= 2_000_000:
            raise ValueError("Dataset must contain 1 to 2,000,000 UTF-8 bytes")
        if len(name) > 120 or len(source) > 500 or len(license) > 200:
            raise ValueError("Dataset metadata exceeds field limits")
        refusal = sensitive_content("\n".join([name, source, license, text]))
        if refusal:
            raise ValueError(refusal)
        previous = self._replaced_version(owner, replaces, purpose) if replaces else None
        digest = hashlib.sha256(text.encode()).hexdigest()
        metadata = {"name": name[:120], "source": source[:500], "license": license[:200]}
        if previous and previous["sha256"] == digest and all(previous[key] == value for key, value in metadata.items()):
            raise ValueError("This version is identical to the one it replaces")
        retired = {}
        if purpose == "training":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            if len(records) < 10:
                raise ValueError("Training admission requires at least 10 unique JSONL examples")
            unique = {}
            for row in records:
                if not isinstance(row, dict) or set(row) != {"messages"}:
                    raise ValueError("Training rows must contain only a messages array")
                messages = row["messages"]
                if not isinstance(messages, list) or len(messages) < 2 or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
                    raise ValueError("Training examples must end with an assistant target")
                for msg in messages:
                    if not isinstance(msg, dict) or set(msg) != {"role", "content"} or msg["role"] not in {"system", "user", "assistant"} or not isinstance(msg["content"], str) or not msg["content"].strip():
                        raise ValueError("Invalid training message")
                key = _input_key(messages)
                if key in unique and unique[key] != row:
                    raise ValueError("Conflicting targets for the same training input")
                unique[key] = row
            if len(unique) < 10:
                raise ValueError("Training admission requires 10 distinct inputs")
            content, retired = _training_splits(unique, _split_history(previous))
            if not content["train"]:
                raise ValueError("A new version needs at least one training example")
        else:
            content = {"text": text}
        identity = uuid.uuid4().hex
        record = {**metadata, "purpose": purpose, "sha256": digest, "state": "ready", "content": content,
                  "version": previous["version"] + 1 if previous else 1,
                  "lineage_id": previous["lineage_id"] if previous else identity}
        if previous:
            record.update(previous_id=previous["id"], previous_sha256=previous["sha256"])
        if retired:
            record["retired_splits"] = retired
        if provenance:
            record["transform"] = provenance
        if crawl:
            record["crawl"] = crawl
        try:
            return self.put(owner, "dataset", record, identity=identity)
        except sqlite3.IntegrityError:
            # Another upload replaced the same version first.
            raise ValueError(NEWER_VERSION_EXISTS) from None

    def _replaced_version(self, owner: str, identity: str, purpose: str) -> dict:
        previous = self.get(owner, identity)
        if previous["kind"] != "dataset":
            raise ValueError("Resource not found")
        if previous.get("purpose") != purpose:
            raise ValueError("A new version must keep the dataset's purpose")
        with self.connect() as db:
            # Only the latest version of a history can be replaced, so history stays linear
            # even after a middle version is deleted.
            newer = db.execute("SELECT 1 FROM resource WHERE owner=? AND kind='dataset' AND "
                               "(json_extract(data, '$.previous_id')=? OR "
                               "(json_extract(data, '$.lineage_id')=? AND json_extract(data, '$.version')>?))",
                               (owner, identity, previous["lineage_id"], previous["version"])).fetchone()
        if newer is not None:
            raise ValueError(NEWER_VERSION_EXISTS)
        return previous

    def search(self, owner: str, identity: str, query: str, limit=5) -> list[dict]:
        from services.retrieval.fts import FtsRecord, search_fts
        from services.extensions.retrieval import dataset_chunks, make_hit
        resource = self._retrieval_resource(owner, identity)
        chunks = dataset_chunks(resource["content"]["text"])
        # Authorization above selects the corpus; the shared FTS engine never
        # receives another owner's records or makes permission decisions.
        matches = search_fts([FtsRecord(identifier=str(i), content=text)
            for i, text in enumerate(chunks, 1)], query,
            limit=min(max(int(limit), 1), 10), max_query_tokens=32)
        return [make_hit(resource, int(index), chunks[int(index)-1]) for index in matches.identifiers]

    def _retrieval_resource(self, owner: str, identity: str) -> dict:
        resource = self.get(owner, identity)
        if resource["kind"] != "dataset" or resource.get("purpose") != "rag" or resource.get("state") != "ready":
            raise ValueError("Not an admitted retrieval dataset")
        return resource

    def resolve(self, owner: str, identity: str, chunk: int, sha256: str) -> dict:
        from services.extensions.retrieval import dataset_chunks, make_hit
        resource = self._retrieval_resource(owner, identity)
        if sha256 != resource["sha256"]:
            raise ValueError("Retrieved reference belongs to a different dataset version")
        chunks = dataset_chunks(resource["content"]["text"])
        if type(chunk) is not int or not 1 <= chunk <= len(chunks):
            raise ValueError("Retrieved chunk does not exist")
        return make_hit(resource, chunk, chunks[chunk-1])
