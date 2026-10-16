import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
from pathlib import Path

import pytest

from services.knowledge.importing import KnowledgeImportError, import_knowledge_snapshot
from services.knowledge.local import LocalKnowledgeService
from services.knowledge.types import KnowledgeSearchQuery


def dataset(tmp_path, **changes):
    row = {"document_id": "KB-IMPORTED", "kind": "knowledge", "title": "Import operator guide",
           "summary": "Reviewed authored dataset text", "content": "The archive is checked with its SHA256 digest before publication.",
           "source": "operator-authored-public-guide", "license": "CC0-1.0", "visibility": "public", "reviewed": True,
           "tags": ["archive"]}
    row.update(changes)
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(row) + "\n")
    return path


def test_import_preserves_exact_operator_content_and_is_retrievable_after_reopen(tmp_path):
    source = dataset(tmp_path)
    runbook = {**json.loads(source.read_text()), "document_id": "RB-IMPORTED", "kind": "runbook",
               "title": "Validate an archive", "content": "1. Compute the archive digest.\n2. Compare it with the reviewed digest."}
    with source.open("a") as stream:
        stream.write(json.dumps(runbook) + "\n")
    imported = import_knowledge_snapshot(source, tmp_path / "snapshots")
    assert imported.path.read_bytes() == source.read_bytes()
    assert imported.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert imported.path.name == imported.sha256 + ".jsonl"
    service = LocalKnowledgeService(imported.path, imported.sha256)
    assert service.get_knowledge("KB-IMPORTED").document.content == json.loads(source.read_text().splitlines()[0])["content"]
    assert service.get_runbook("RB-IMPORTED").document.content == runbook["content"]
    assert service.search(KnowledgeSearchQuery(query="archive")).count == 2
    assert not service.get_knowledge("missing").ok
    assert imported.report()["document_count"] == 2
    assert imported.report()["environment_modified"] is False


def test_import_is_idempotent_and_concurrent_publication_never_overwrites(tmp_path):
    source = dataset(tmp_path)
    destination = tmp_path / "snapshots"
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: import_knowledge_snapshot(source, destination), range(8)))
    assert len({result.path for result in results}) == 1
    assert list(destination.iterdir()) == [results[0].path]
    assert results[0].path.read_bytes() == source.read_bytes()


def test_existing_corrupt_destination_is_refused_instead_of_replaced(tmp_path):
    source = dataset(tmp_path)
    output = tmp_path / "snapshots"
    output.mkdir()
    target = output / (hashlib.sha256(source.read_bytes()).hexdigest() + ".jsonl")
    target.write_bytes(b"operator-existing-data")
    with pytest.raises(KnowledgeImportError, match="different bytes"):
        import_knowledge_snapshot(source, output)
    assert target.read_bytes() == b"operator-existing-data"
    assert list(output.iterdir()) == [target]


@pytest.mark.parametrize("changes", [
    {"visibility": "private"}, {"reviewed": False}, {"source": ""}, {"license": ""},
    {"content": "password=private123"}, {"kind": "unknown"}, {"content": ""},
    {"source": "password=private123"}, {"license": "password=private123"},
    {"document_id": " "}, {"title": " "}, {"source": " "},
])
def test_unadmitted_content_is_never_published_or_echoed(tmp_path, changes):
    source = dataset(tmp_path, **changes)
    output = tmp_path / "snapshots"
    with pytest.raises(KnowledgeImportError) as error:
        import_knowledge_snapshot(source, output)
    assert "private123" not in str(error.value)
    assert not output.exists()


def test_changed_source_creates_new_pin_without_mutating_old_snapshot(tmp_path):
    source = dataset(tmp_path)
    first = import_knowledge_snapshot(source, tmp_path / "snapshots")
    first_bytes = first.path.read_bytes()
    row = json.loads(source.read_text())
    row["content"] = "The revised procedure includes a second operator review."
    source.write_text(json.dumps(row) + "\n")
    second = import_knowledge_snapshot(source, tmp_path / "snapshots")
    assert second.sha256 != first.sha256
    assert first.path.read_bytes() == first_bytes
    assert len(list((tmp_path / "snapshots").glob("*.jsonl"))) == 2


def test_cli_failure_does_not_print_rejected_dataset_text(tmp_path):
    source = dataset(tmp_path, content="password=private123")
    script = Path(__file__).resolve().parents[3] / "scripts/import_knowledge.py"
    result = subprocess.run([sys.executable, str(script), "--source", str(source), "--output-dir", str(tmp_path / "snapshots")], capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stderr)["error"] == "knowledge_import_refused"
    assert "private123" not in result.stdout + result.stderr
