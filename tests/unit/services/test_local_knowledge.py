import hashlib
import json
import pytest
from services.knowledge.local import LocalKnowledgeService
from services.knowledge.types import KnowledgeSearchQuery


def build(tmp_path, **changes):
    row = dict(document_id="KB-NET", kind="knowledge", title="Router release", summary="Inspect firmware",
        content="The board query returns the running firmware release.", source="authored", license="CC0",
        visibility="public", reviewed=True, tags=["networking"])
    row.update(changes)
    raw = json.dumps(row).encode()
    path = tmp_path / "knowledge.jsonl"
    path.write_bytes(raw)
    return LocalKnowledgeService(path, hashlib.sha256(raw).hexdigest())


def test_retrieves_actual_text_and_provenance(tmp_path):
    service = build(tmp_path)
    result = service.search(KnowledgeSearchQuery(query='firmware OR "'))
    assert result.count == 1
    assert result.hits[0].provenance["source"] == "authored"
    assert service.get_knowledge("KB-NET").document.content.startswith("The board")
    assert not service.get_runbook("KB-NET").ok
    assert service.search(KnowledgeSearchQuery(query="nonexistent")).count == 0


@pytest.mark.parametrize("changes", [{"visibility":"private"}, {"reviewed":False}, {"license":""}, {"content":"password=private123"}])
def test_rejects_unadmitted_documents(tmp_path, changes):
    with pytest.raises(ValueError): build(tmp_path, **changes)


def test_rejects_snapshot_change(tmp_path):
    path = tmp_path / "kb"
    path.write_text("{}")
    with pytest.raises(ValueError, match="hash"):
        LocalKnowledgeService(path, "0" * 64)
