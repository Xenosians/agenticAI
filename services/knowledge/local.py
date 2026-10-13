"""Reviewed public-document retrieval using SQLite FTS5, without model training.

Private documents are rejected because the current provider contract has no
authenticated caller identity. Retrieved text never grants tool authority.
"""
import hashlib
import json
from pathlib import Path

from .base import KnowledgeService
from .types import KnowledgeDocument, KnowledgeLookupResult, KnowledgeSearchHit, KnowledgeSearchQuery, KnowledgeSearchResult
from learning.evidence.sanitizer import sanitize_text
from services.retrieval.fts import FtsRecord, search_fts


class LocalKnowledgeService(KnowledgeService):
    def __init__(self, path: Path, expected_sha256: str):
        raw = path.read_bytes()
        if not expected_sha256 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("Knowledge snapshot does not match the reviewed hash.")
        self.documents = {}
        for line in raw.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Knowledge records must be JSON objects.")
            if row.get("visibility") != "public" or row.get("reviewed") is not True:
                raise ValueError("Only reviewed public documents are admitted.")
            if not row.get("source") or not row.get("license"):
                raise ValueError("Knowledge provenance and license are required.")
            document = KnowledgeDocument.model_validate({**row, "provider": "local-fts5",
                "provenance": {"source": row["source"], "license": row["license"],
                    "snapshot_sha256": expected_sha256, "visibility": "public",
                    "content_sha256": hashlib.sha256(str(row.get("content", "")).encode()).hexdigest()}})
            if document.kind not in {"knowledge", "runbook"} or not document.content.strip():
                raise ValueError("Invalid knowledge kind or empty content.")
            if document.document_id in self.documents:
                raise ValueError("Duplicate document identity.")
            text = "\n".join([document.title, document.summary, document.content, *document.tags])
            if sanitize_text(text) != text:
                raise ValueError("Knowledge contains privacy/secret scanner findings.")
            self.documents[document.document_id] = document
        if not self.documents:
            raise ValueError("Knowledge snapshot is empty.")

    def search(self, query: KnowledgeSearchQuery) -> KnowledgeSearchResult:
        # Only documents that passed this service's public snapshot admission
        # enter the shared search mechanics. Search itself grants no access.
        matches = search_fts(
            (FtsRecord(d.document_id, d.content, kind=d.kind, title=d.title,
                       summary=d.summary, tags=tuple(d.tags))
             for d in self.documents.values()),
            query.query, limit=query.limit, kind=query.kind,
        )
        hits = [KnowledgeSearchHit.model_validate(self.documents[identity].model_dump())
                for identity in matches.identifiers]
        return KnowledgeSearchResult(ok=True, status="success", hits=hits,
                                     count=len(hits), truncated=matches.truncated)

    def _get(self, identity: str, kind: str) -> KnowledgeLookupResult:
        document = self.documents.get(identity)
        if document is None or document.kind != kind:
            return KnowledgeLookupResult(ok=False, status="not_found", error="Document not found.")
        return KnowledgeLookupResult(ok=True, status="success", document=document)

    def get_knowledge(self, document_id: str) -> KnowledgeLookupResult:
        return self._get(document_id, "knowledge")

    def get_runbook(self, runbook_id: str) -> KnowledgeLookupResult:
        return self._get(runbook_id, "runbook")
