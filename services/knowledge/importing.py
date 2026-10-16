"""Publish operator-reviewed knowledge using the existing snapshot admission rules."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

from .local import LocalKnowledgeService

MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024


class KnowledgeImportError(ValueError):
    """A safe operator-facing import failure, without copying document content."""


@dataclass(frozen=True)
class ImportedKnowledgeSnapshot:
    path: Path
    sha256: str
    knowledge_count: int
    runbook_count: int

    def report(self) -> dict:
        return {
            "schema": "knowledge-snapshot-import.v1",
            "snapshot_path": str(self.path),
            "snapshot_sha256": self.sha256,
            "document_count": self.knowledge_count + self.runbook_count,
            "knowledge_count": self.knowledge_count,
            "runbook_count": self.runbook_count,
            "configuration": {
                "KNOWLEDGE_BACKEND": "local",
                "KNOWLEDGE_SNAPSHOT_PATH": str(self.path),
                "KNOWLEDGE_SNAPSHOT_SHA256": self.sha256,
            },
            "environment_modified": False,
            "model_training_performed": False,
        }


def import_knowledge_snapshot(source: Path, output_dir: Path) -> ImportedKnowledgeSnapshot:
    """Validate before publishing exact bytes under a content-addressed filename.

    Public/review/license/source checks and secret scanning stay owned by
    LocalKnowledgeService. The importer neither converts private documents to
    public nor manufactures knowledge or runbook answers.
    """
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    try:
        with source.open("rb") as handle:
            raw = handle.read(MAX_SNAPSHOT_BYTES + 1)
    except OSError:
        raise KnowledgeImportError("The source snapshot could not be read.") from None
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise KnowledgeImportError("The snapshot exceeds the 32 MiB import limit.")
    digest = hashlib.sha256(raw).hexdigest()
    try:
        service = LocalKnowledgeService(source, digest)
    except (OSError, UnicodeError, ValueError, TypeError):
        # Validation errors can contain the original content. Never print them.
        raise KnowledgeImportError("The snapshot failed existing public-document admission or integrity checks.") from None

    target = output_dir / f"{digest}.jsonl"
    counts = {kind: sum(document.kind == kind for document in service.documents.values())
              for kind in ("knowledge", "runbook")}
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".knowledge-import-", dir=output_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(raw)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            # Atomic no-overwrite publication, also when two operators import
            # the same reviewed bytes concurrently. Never replace an old pin.
            os.link(temporary_path, target)
        except FileExistsError:
            if target.is_symlink() or not target.is_file() or target.read_bytes() != raw:
                raise KnowledgeImportError("The content-addressed destination exists with different bytes.") from None
    except OSError:
        raise KnowledgeImportError("The validated snapshot could not be published atomically.") from None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    # Reopen the actual published bytes through the production provider.
    try:
        LocalKnowledgeService(target, digest)
    except (OSError, UnicodeError, ValueError, TypeError):
        raise KnowledgeImportError("The published snapshot did not pass read-back verification.") from None
    return ImportedKnowledgeSnapshot(target, digest, counts["knowledge"], counts["runbook"])
