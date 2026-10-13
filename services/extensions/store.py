from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from pathlib import Path

from cryptography.fernet import Fernet


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
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def put(self, owner: str, kind: str, data: dict, secret: str = "") -> dict:
        if not owner or len(owner) > 200:
            raise ValueError("Invalid owner")
        identity = uuid.uuid4().hex
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
        if secret:
            data["api_key"] = self.cipher.decrypt(row["secret"]).decode() if row["secret"] else ""
        return data

    def list(self, owner: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT id FROM resource WHERE owner=? ORDER BY rowid", (owner,)).fetchall()
        return [self.get(owner, row["id"]) for row in rows]

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

    def delete(self, owner: str, identity: str):
        self.get(owner, identity)
        with self.connect() as db:
            db.execute("UPDATE preference SET model_id=NULL WHERE owner=? AND model_id=?", (owner, identity))
            db.execute("UPDATE preference SET dataset_id=NULL WHERE owner=? AND dataset_id=?", (owner, identity))
            db.execute("DELETE FROM resource WHERE owner=? AND id=?", (owner, identity))

    def dataset(self, owner: str, *, name: str, text: str, source: str, license: str,
                purpose: str, reviewed: bool) -> dict:
        if not name.strip() or not source.strip() or not license.strip():
            raise ValueError("Name, source and permitted-use/license metadata are required")
        if purpose not in {"rag", "training"} or reviewed is not True:
            raise ValueError("Choose a purpose and confirm review/permission to use this dataset")
        if not 0 < len(text.encode()) <= 2_000_000:
            raise ValueError("Dataset must contain 1 to 2,000,000 UTF-8 bytes")
        from learning.evidence.sanitizer import sanitize_text
        admission_text = "\n".join([name, source, license, text])
        if len(name) > 120 or len(source) > 500 or len(license) > 200:
            raise ValueError("Dataset metadata exceeds field limits")
        if sanitize_text(admission_text) != admission_text:
            raise ValueError("Dataset failed the shared training privacy/secret scanner")
        # Admission is conservative. Detection is not a guarantee of de-identification.
        patterns = [r"-----BEGIN .*PRIVATE KEY-----", r"(?i)(api[_-]?key|password|secret|token)\s*[:=]\s*[^\s]{6,}",
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", r"\bsk-[A-Za-z0-9_-]{16,}\b"]
        if any(re.search(pattern, admission_text) for pattern in patterns):
            raise ValueError("Dataset contains possible secrets or personal identifiers; sanitize before admission")
        digest = hashlib.sha256(text.encode()).hexdigest()
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
                # Same input may never leak across splits, even with different targets.
                key = hashlib.sha256(json.dumps(messages[:-1], sort_keys=True).encode()).hexdigest()
                if key in unique and unique[key] != row:
                    raise ValueError("Conflicting targets for the same training input")
                unique[key] = row
            if len(unique) < 10:
                raise ValueError("Training admission requires 10 distinct inputs")
            ordered = sorted(unique)
            validation_keys = set(ordered[:max(1, len(ordered)//5)])
            content = {"train": [unique[k] for k in ordered if k not in validation_keys],
                       "validation": [unique[k] for k in ordered if k in validation_keys]}
        else:
            content = {"text": text}
        return self.put(owner, "dataset", {"name": name[:120], "source": source[:500], "license": license[:200],
                        "purpose": purpose, "sha256": digest, "state": "ready", "content": content})

    def search(self, owner: str, identity: str, query: str, limit=5) -> list[dict]:
        resource = self.get(owner, identity)
        if resource["kind"] != "dataset" or resource.get("purpose") != "rag":
            raise ValueError("Not a retrieval dataset")
        words = re.findall(r"\w+", query)[:32]
        if not words:
            return []
        # FTS runs only over this caller's selected dataset, never a shared corpus.
        chunks = [resource["content"]["text"][i:i+2000] for i in range(0, len(resource["content"]["text"]), 1800)]
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")
            db.executemany("INSERT INTO docs(content) VALUES (?)", [(chunk,) for chunk in chunks])
            rows = db.execute("SELECT rowid,content FROM docs WHERE docs MATCH ? ORDER BY rank LIMIT ?",
                              (" OR ".join('"'+word+'"' for word in words), min(max(int(limit), 1), 10))).fetchall()
        return [{"content": row[1], "source": resource["source"], "dataset_id": identity,
                 "chunk": row[0], "sha256": resource["sha256"], "trust": "untrusted_retrieval"} for row in rows]
