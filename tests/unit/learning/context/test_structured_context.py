from pathlib import Path

import pytest

from learning.context import LearningContextRecorder, assemble_context_bundle, load_context_records, record_human_context, record_task_context
from learning.context.models import StructuredContextRecord


def test_task_context_requires_lineage_and_never_grants_authority(tmp_path: Path):
    path = tmp_path / "context.jsonl"
    recorder = LearningContextRecorder(path=path)
    record = record_task_context(
        recorder,
        subject="task",
        original_request="Check jdoe VPN access",
        semantic_contract={"effect": "read", "allowed_tools": ["check_access"]},
        trajectory_id="trajectory-1",
        task_id="task-1",
    )
    assert record is not None
    assert record.scope == "task"
    assert record.authority_grant is False
    assert record.training_eligible is False

    with pytest.raises(ValueError, match="task-scoped"):
        StructuredContextRecord(
            context_id="x",
            observed_at="2026-09-18T00:00:00+00:00",
            kind="task",
            scope="task",
            subject="bad",
            payload={},
        )


def test_assembler_keeps_global_human_and_matching_lineage_only(tmp_path: Path):
    path = tmp_path / "context.jsonl"
    recorder = LearningContextRecorder(path=path)
    record_human_context(recorder, subject="convention", project_conventions=["Use logical repo aliases"])
    record_task_context(recorder, subject="match", original_request="A", trajectory_id="trajectory-1")
    record_task_context(recorder, subject="other", original_request="B", trajectory_id="trajectory-2")

    bundle = assemble_context_bundle(load_context_records(path), trajectory_id="trajectory-1")
    subjects = [item.subject for item in bundle.records]
    assert "convention" in subjects
    assert "match" in subjects
    assert "other" not in subjects
